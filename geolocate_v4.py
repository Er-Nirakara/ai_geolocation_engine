# --- IMPORTS ---
import os
import argparse
from PIL import Image # Make sure Pillow is installed
# Required for HEIC/HEIF: pip install pillow-heif
# Required for AVIF: pip install pillow-avif-plugin
# Pillow will use these automatically if installed. No direct import needed here.
from PIL.ExifTags import TAGS, GPSTAGS
import googlemaps # Make sure google-maps-services-python is installed
from dotenv import load_dotenv # Make sure python-dotenv is installed
import spacy # Make sure spacy and a model (e.g., en_core_web_sm) are installed
import sys
from google.cloud import vision # Make sure google-cloud-vision is installed
import math # Make sure math is imported

# --- Constants ---
# Load the spaCy model once
try:
    NLP = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    NLP = None

LOCATION_ENTITY_LABELS = {"GPE", "LOC", "FAC"}
LANDMARK_CONFIDENCE_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD_KM = 10.0

# --- Helper Functions --- #

# --- EXIF Functions ---
def get_exif_data(image_path):
    """Extracts EXIF data from an image file.
       Tries to handle multiple formats via Pillow and its plugins.
       NOTE: Requires necessary plugins (e.g., pillow-heif, pillow-avif-plugin)
             to be installed in the environment for formats like HEIC, AVIF.
    """
    image = None
    try:
        print(f"Attempting to open image: {image_path}")
        # Pillow automatically uses installed plugins (pillow-heif, pillow-avif) here.
        image = Image.open(image_path)
        image.load() # Force loading data to catch errors early
        print(f"Image format identified by Pillow: {image.format}")
        exif_data = image._getexif()
        if not exif_data: return None
        print(f"EXIF data found (keys: {len(exif_data)})")
        return exif_data

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Image.UnidentifiedImageError:
        print(f"Error: Cannot identify image file format for {image_path}.")
        print("       Ensure it's a valid image and necessary Pillow plugins")
        print("       (e.g., `pip install pillow-heif pillow-avif-plugin`) are installed.")
        return None
    except Exception as e:
        print(f"Error reading image or EXIF data from {image_path}: {e}")
        return None
    finally:
        if image: image.close()

# --- GPS Info Extraction ---
def get_gps_info(exif_data):
    if not exif_data: return None
    gps_info = {}
    gps_tag_code = None
    for k, v in TAGS.items():
        if v == 'GPSInfo': gps_tag_code = k; break
    if gps_tag_code is None or gps_tag_code not in exif_data: return None
    raw_gps_data = exif_data[gps_tag_code]
    for k, v in GPSTAGS.items():
        if k in raw_gps_data: gps_info[v] = raw_gps_data[k]
    return gps_info

# --- DMS to Decimal Conversion ---
def dms_to_decimal(dms, ref):
    if not isinstance(dms, (tuple, list)) or len(dms) < 3: return None
    try:
        degrees = float(dms[0]); minutes = float(dms[1]) / 60.0; seconds = float(dms[2]) / 3600.0
    except (ValueError, TypeError): return None
    decimal = degrees + minutes + seconds
    if ref in ['S', 'W']: decimal = -decimal
    return decimal

# --- Decimal Coordinate Extraction ---
def get_decimal_coordinates(gps_info):
    if not gps_info: return None, None
    lat_dms = gps_info.get('GPSLatitude'); lat_ref = gps_info.get('GPSLatitudeRef')
    lon_dms = gps_info.get('GPSLongitude'); lon_ref = gps_info.get('GPSLongitudeRef')
    if lat_dms and lat_ref and lon_dms and lon_ref:
        lat = dms_to_decimal(lat_dms, lat_ref)
        lon = dms_to_decimal(lon_dms, lon_ref)
        if lat is None or lon is None: return None, None
        return lat, lon
    else: return None, None

# --- Geocoding Functions ---
def reverse_geocode(lat, lon, api_key):
    if not api_key: return None
    if lat is None or lon is None: return None
    try:
        gmaps = googlemaps.Client(key=api_key)
        result = gmaps.reverse_geocode((lat, lon))
        return result[0]['formatted_address'] if result else None
    except Exception as e: print(f"Error (Reverse Geocode): {e}"); return None

def geocode_location_name(location_name, api_key):
    if not api_key: return None, None, None
    if not location_name: return None, None, None
    try:
        gmaps = googlemaps.Client(key=api_key)
        result = gmaps.geocode(location_name)
        if result:
            first = result[0]; lat = first['geometry']['location']['lat']
            lon = first['geometry']['location']['lng']; addr = first['formatted_address']
            print(f"Geocoding success for '{location_name[:50]}...': {addr} ({lat:.4f}, {lon:.4f})")
            return lat, lon, addr
        else:
            print(f"Geocoding no results for '{location_name[:50]}...'.")
            return None, None, None
    except Exception as e: print(f"Error (Geocode for '{location_name[:50]}...'): {e}"); return None, None, None

# --- NLP Function ---
def process_text_with_nlp(text):
    if not NLP: print("Warning: spaCy model not loaded."); return []
    if not text: return []
    doc = NLP(text); locations = []
    print("\n--- NLP Analysis ---"); print(f"Text processed: '{text}'")
    for ent in doc.ents:
        if ent.label_ in LOCATION_ENTITY_LABELS:
            print(f"  Found entity: '{ent.text}' ({ent.label_})"); locations.append(ent.text)
    if not locations: print("  No location entities found.")
    print("--- End NLP Analysis ---"); return locations

# --- Vision API Function ---
def analyze_image_with_vision_api(image_path):
    landmark_result, ocr_texts = None, []
    gcp_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not gcp_credentials_path or not os.path.exists(gcp_credentials_path):
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS invalid. Skipping Vision API.")
        return None, []
    try:
        client = vision.ImageAnnotatorClient()
        if not os.path.exists(image_path): raise FileNotFoundError(f"Image not found: {image_path}")
        with open(image_path, 'rb') as f: content = f.read()
        image = vision.Image(content=content)
        # Landmark
        print("\n--- Vision API: Landmark Detection ---")
        resp_lm = client.landmark_detection(image=image)
        if resp_lm.error.message: print(f"Vision API Error (Landmark): {resp_lm.error.message}")
        elif resp_lm.landmark_annotations:
            l = resp_lm.landmark_annotations[0]
            print(f"  Detected: {l.description} (Conf: {l.score:.2f})")
            if l.locations:
                loc = l.locations[0].lat_lng
                print(f"  Coords: Lat={loc.latitude:.6f}, Lon={loc.longitude:.6f}")
                landmark_result = {'name':l.description,'latitude':loc.latitude,'longitude':loc.longitude,'confidence':l.score}
            else: landmark_result = {'name':l.description,'latitude':None,'longitude':None,'confidence':l.score}
        else: print("  No landmarks detected.")
        print("--- End Landmark Detection ---")
        # Text (OCR)
        print("\n--- Vision API: Text Detection (OCR) ---")
        resp_txt = client.text_detection(image=image)
        if resp_txt.error.message: print(f"Vision API Error (Text): {resp_txt.error.message}")
        elif resp_txt.text_annotations:
            full_text = resp_txt.text_annotations[0].description
            print(f"  Detected (snippet): {full_text[:100].replace(os.linesep, ' ')}...")
            ocr_texts.append(full_text.replace('\n', ' '))
        else: print("  No text detected.")
        print("--- End Text Detection ---")
    except FileNotFoundError as e: print(e)
    except Exception as e: print(f"Unexpected error during Vision API: {e}")
    return landmark_result, ocr_texts

# --- Haversine Distance ---
def haversine_distance(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]: return float('inf')
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2-lon1; dlat = lat2-lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    r = 6371; return 2 * math.asin(math.sqrt(a)) * r

# --- Main Execution ---
def main():
    if not NLP: print("Exiting: spaCy model failed to load."); sys.exit(1)
    load_dotenv()
    gmaps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    gcp_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not gcp_credentials_path or not os.path.exists(gcp_credentials_path): print("Warning: GOOGLE_APPLICATION_CREDENTIALS invalid. Vision API skipped.")
    if not gmaps_api_key: print("Warning: GOOGLE_MAPS_API_KEY not found. Geocoding/Reverse Geocoding will fail.")

    parser = argparse.ArgumentParser(description="Geolocation with Confidence Score.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--text", help="Optional text associated with the image", default=None)
    args = parser.parse_args()
    print(f"\nProcessing image: {args.image_path}")
    if args.text: print(f"Associated text: '{args.text}'")

    # --- Stage 1: Data Extraction ---
    print("\n--- 1a. EXIF Analysis ---")
    exif_data = get_exif_data(args.image_path) # Attempts opening various formats
    gps_info = get_gps_info(exif_data)
    exif_lat, exif_lon = get_decimal_coordinates(gps_info)
    exif_address = None
    if exif_lat is not None:
        print(f"Found EXIF Coords: Lat={exif_lat:.6f}, Lon={exif_lon:.6f}")
        if gmaps_api_key: exif_address = reverse_geocode(exif_lat, exif_lon, gmaps_api_key)
        if exif_address: print(f"Reverse Geocoded Addr: {exif_address}")
        else: print("Could not reverse geocode EXIF.")
    else: print("No usable GPS coords in EXIF.")
    print("--- End EXIF Analysis ---")

    landmark_info, ocr_texts = analyze_image_with_vision_api(args.image_path)
    nlp_locations = process_text_with_nlp(args.text) if args.text else []

    # --- Stage 2: Geocode Text Clues ---
    geocoded_nlp_results, geocoded_ocr_results = [], []
    print("\n--- 2. Geocoding Text Clues ---")
    if not gmaps_api_key: print("Skipping text geocoding (No API Key).")
    else:
        if nlp_locations: print("Geocoding NLP text..."); # ... (loop as before)
        for name in nlp_locations:
            lat, lon, addr = geocode_location_name(name, gmaps_api_key)
            if lat is not None: geocoded_nlp_results.append({'name': name, 'latitude': lat, 'longitude': lon, 'address': addr})
        if ocr_texts: print("Geocoding OCR text..."); # ... (loop as before)
        for text_block in ocr_texts:
            lat, lon, addr = geocode_location_name(text_block, gmaps_api_key)
            if lat is not None: geocoded_ocr_results.append({'text': text_block[:50]+"...", 'latitude': lat, 'longitude': lon, 'address': addr})
    print("--- End Geocoding ---")

    # --- Stage 3: Evidence Fusion & Confidence Scoring ---
    print("\n--- 3. Evidence Fusion & Confidence Calculation ---")
    candidate_locations = []

    # Add candidates from each source (simplified for brevity - use full logic from previous code)
    if exif_lat is not None: candidate_locations.append({"source":"EXIF","latitude":exif_lat,"longitude":exif_lon,"address":exif_address,"confidence_raw":1.0}); print("  Added EXIF candidate.")
    if landmark_info and landmark_info.get('latitude') and landmark_info.get('confidence',0)>=LANDMARK_CONFIDENCE_THRESHOLD: candidate_locations.append({"source":"CV Landmark","name":landmark_info.get('name'),"latitude":landmark_info.get('latitude'),"longitude":landmark_info.get('longitude'),"address":None,"confidence_raw":landmark_info.get('confidence',0)}); print(f"  Added Landmark candidate: '{landmark_info.get('name')}'")
    elif landmark_info: print(f"  Ignoring Landmark candidate: '{landmark_info.get('name')}' (Conf below threshold)")
    all_text_results=geocoded_nlp_results+geocoded_ocr_results
    if all_text_results: first=all_text_results[0]; origin="NLP" if 'name' in first else "OCR"; detail=first.get('name') or first.get('text'); candidate_locations.append({"source":origin,"detail":detail,"latitude":first.get('latitude'),"longitude":first.get('longitude'),"address":first.get('address'),"confidence_raw":0.5}); print(f"  Added Text candidate ({origin}): '{detail}'")


    # --- Determine Final Result based on Priority and Consistency ---
    final_latitude = None
    final_longitude = None
    final_address = None
    final_source = "None"
    final_confidence = 0.0

    if not candidate_locations:
        print("Decision: No usable location candidates found from any source.")
    else:
        # Sort candidates by priority score (EXIF > Landmark > Text)
        def get_priority_score(candidate):
            source = candidate["source"]
            if source == "EXIF": return 3
            if source == "CV Landmark": return 2
            if source in ["NLP", "OCR"]: return 1
            return 0

        candidate_locations.sort(key=get_priority_score, reverse=True)
        best_candidate = candidate_locations[0] # Highest priority candidate

        print(f"  Highest priority candidate source: {best_candidate['source']}")

        # Calculate Base Confidence
        base_confidence = 0.0
        source = best_candidate["source"]
        if source == "EXIF": base_confidence = 0.9
        elif source == "CV Landmark": base_confidence = 0.4 + (best_candidate.get('confidence_raw', 0) * 0.4)
        elif source in ["NLP", "OCR"]: base_confidence = 0.5

        # Consistency Bonus/Penalty
        bonus = 0.0 # Initialize bonus
        if len(candidate_locations) > 1:
            print("  Checking consistency...")
            # ***** CORRECTED LOOP *****
            for other_candidate in candidate_locations[1:]:
                 # Calculate distance (can be kept compact if preferred)
                 dist = haversine_distance(
                     best_candidate["latitude"], best_candidate["longitude"],
                     other_candidate["latitude"], other_candidate["longitude"]
                 )
                 print(f"    Distance to '{other_candidate['source']}' candidate: {dist:.2f} km")
                 # --- Start IF block correctly ---
                 if dist <= CONSISTENCY_THRESHOLD_KM:
                     bonus = 0.15  # Assign bonus on new line
                     print(f"    -> Consistent (within {CONSISTENCY_THRESHOLD_KM} km). Applying bonus.") # Print message on new line
                     break # Exit loop on new line
            # ***** END CORRECTED LOOP SECTION *****

            # Check if bonus was applied after the loop
            if bonus == 0.0: # This check remains outside the loop
                 print("    -> No other candidates found within consistency threshold.")
        else:
             print("  Only one candidate location found, skipping consistency check.")

        # Calculate final confidence
        final_confidence = min(max(base_confidence + bonus, 0.0), 1.0) # Clamp 0-1

        # Assign final details from the chosen best candidate
        final_latitude = best_candidate["latitude"]
        final_longitude = best_candidate["longitude"]
        final_source = best_candidate["source"]
        if best_candidate.get("name"): final_source += f" ('{best_candidate['name']}')"
        elif best_candidate.get("detail"): final_source += f" ('{best_candidate['detail']}')"

        # Get address (either already present or via reverse geocoding)
        final_address = best_candidate.get("address")
        if final_address is None and source == "CV Landmark":
             print(f"  Attempting reverse geocode for chosen landmark '{best_candidate.get('name', '')}'...")
             if gmaps_api_key:
                 lm_address = reverse_geocode(final_latitude, final_longitude, gmaps_api_key)
                 final_address = lm_address if lm_address else best_candidate.get('name', "N/A")
             else:
                 final_address = best_candidate.get('name', "N/A (No API Key)")
        elif final_address is None and source == "EXIF":
             final_address = "N/A (Reverse Geocoding Failed/Skipped)"
        elif final_address is None: # Catch any other case where address might be missing
            final_address = "N/A"

    print("--- End Fusion & Confidence ---")

    # --- Stage 4: Final Output ---
    # ... (Output logic remains the same) ...
    print("\n"+"="*30); print("      FINAL RESULT"); print("="*30)
    print(f"Source of Prediction: {final_source}")
    if final_latitude is not None:
        conf_label = "High" if final_confidence>=0.75 else ("Medium" if final_confidence>=0.4 else "Low")
        print(f"Confidence:           {final_confidence:.2f} ({conf_label})")
        print(f"Predicted Coordinates: Lat={final_latitude:.6f}, Lon={final_longitude:.6f}")
        print(f"Predicted Address:     {final_address}")
        if gmaps_api_key: print(f"Map Link:             https://maps.googleapis.com/maps/api/staticmap?center={final_latitude},{final_longitude}&zoom=13&size=600x300&maptype=roadmap&markers=color:red%7Clabel:P%7C{final_latitude},{final_longitude}&key={gmaps_api_key}")
        else: print("Map Link:             (Google Maps API Key required)")
    else: print("Confidence:           0.0 (Low)"); print("Could not determine location from available sources.")
    print("="*30+"\n")


if __name__ == "__main__":
    # IMPORTANT: Ensure required plugins are installed in the environment:
    # pip install pillow-heif pillow-avif-plugin
    main()