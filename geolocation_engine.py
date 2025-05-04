# geolocation_engine.py (Core processing logic - Reviewed for Syntax/Indentation)

# --- IMPORTS ---
import os
import math
from PIL import Image, ExifTags # Use ExifTags directly
# Required for HEIC/HEIF: pip install pillow-heif
# Required for AVIF: pip install pillow-avif-plugin
import googlemaps
from dotenv import load_dotenv
import spacy
import sys
from google.cloud import vision
import logging

# --- Setup Logging ---
# Configure logging to show messages from the engine
logging.basicConfig(level=logging.INFO, format='%(asctime)s - ENGINE - %(levelname)s - %(message)s')

# --- Constants ---
# Load the spaCy model once
NLP = None # Initialize NLP to None
try:
    NLP = spacy.load("en_core_web_sm")
    logging.info("spaCy model 'en_core_web_sm' loaded.")
except OSError:
    # Log error but allow the script to continue (NLP features will be disabled)
    logging.error("spaCy model 'en_core_web_sm' not found. NLP features disabled.")
    logging.error("To enable NLP, run: python -m spacy download en_core_web_sm")

# Map Pillow's TAGS/GPSTAGS directly for convenience
_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_GPSTAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}

LOCATION_ENTITY_LABELS = {"GPE", "LOC", "FAC"}
LANDMARK_CONFIDENCE_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD_KM = 10.0

# --- Helper Functions --- #

# --- EXIF Functions ---
def get_exif_data(image_path):
    image = None
    try:
        logging.info(f"Attempting to open image: {image_path}")
        image = Image.open(image_path)
        # Force loading the image data and EXIF early to catch decoding errors
        image.load()
        logging.info(f"Image format identified by Pillow: {image.format}")
        # Use the private method for now, can switch to getexif() if needed
        exif_data_raw = image._getexif()
        if not exif_data_raw:
            logging.info("No EXIF metadata found.")
            return None

        # Convert integer keys to tag names for easier debugging (optional)
        exif_data = {}
        for k, v in exif_data_raw.items():
            tag_name = ExifTags.TAGS.get(k, k) # Use tag name if found, else keep integer key
            exif_data[tag_name] = v

        logging.info(f"EXIF data found (keys: {len(exif_data)})")
        return exif_data # Return decoded dictionary

    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return None
    except Image.UnidentifiedImageError:
        # Specific error for unsupported formats
        logging.warning(f"Cannot identify image format for {image_path}. Install plugins (pillow-heif, pillow-avif)?")
        return None
    except Exception as e:
        # Catch other potential errors during image open or EXIF read
        logging.error(f"Error reading image/EXIF from {image_path}: {e}", exc_info=True) # Log traceback for debug
        return None
    finally:
        # Ensure the image file handle is closed if it was successfully opened
        if image:
            image.close()

# --- GPS Info ---
def get_gps_info(exif_data):
    if not exif_data or not isinstance(exif_data, dict):
        return None
    # Use the human-readable tag name 'GPSInfo' which get_exif_data now produces
    gps_info_raw = exif_data.get('GPSInfo')
    if not gps_info_raw:
        return None

    # Decode GPS sub-tags using GPSTAGS mapping
    gps_info = {}
    for k, v in gps_info_raw.items():
        sub_tag_name = ExifTags.GPSTAGS.get(k, k)
        gps_info[sub_tag_name] = v
    return gps_info

# --- DMS to Decimal ---
def dms_to_decimal(dms, ref):
    if not isinstance(dms, (tuple, list)) or len(dms) < 3:
        logging.warning(f"Invalid DMS format received: {dms}")
        return None
    try:
        # Explicitly convert potential Rational objects to float
        degrees = float(dms[0])
        minutes = float(dms[1]) / 60.0
        seconds = float(dms[2]) / 3600.0
    except (ValueError, TypeError, IndexError) as e:
        logging.warning(f"Error converting DMS component to float: {dms} - {e}")
        return None

    decimal = degrees + minutes + seconds
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

# --- Decimal Coords ---
def get_decimal_coordinates(gps_info):
    if not gps_info:
        return None, None

    lat_dms = gps_info.get('GPSLatitude')
    lat_ref = gps_info.get('GPSLatitudeRef')
    lon_dms = gps_info.get('GPSLongitude')
    lon_ref = gps_info.get('GPSLongitudeRef')

    if lat_dms and lat_ref and lon_dms and lon_ref:
        lat = dms_to_decimal(lat_dms, lat_ref)
        lon = dms_to_decimal(lon_dms, lon_ref)
        if lat is not None and lon is not None: # Check conversion success
            return lat, lon
        else:
            logging.warning("DMS to Decimal conversion failed for GPS coordinates.")
            return None, None
    else:
        # This is normal if GPS tags aren't present
        # logging.info("Required GPS tags (Lat/Lon DMS/Ref) not found in GPS Info.")
        return None, None

# --- Geocoding ---
def reverse_geocode(lat, lon, api_key):
    if not api_key:
        logging.warning("Reverse geocode skipped: No Google Maps API Key.")
        return None
    if lat is None or lon is None:
        logging.warning("Reverse geocode skipped: Invalid coordinates.")
        return None
    try:
        gmaps = googlemaps.Client(key=api_key)
        result = gmaps.reverse_geocode((lat, lon))
        if result:
            return result[0]['formatted_address']
        else:
            logging.info(f"Reverse geocode returned no results for ({lat},{lon}).")
            return None
    except Exception as e:
        logging.error(f"Reverse Geocode API Error: {e}", exc_info=False) # Don't need full traceback for API errors usually
        return None

def geocode_location_name(location_name, api_key):
    if not api_key:
        logging.warning("Geocode skipped: No Google Maps API Key.")
        return None, None, None
    if not location_name:
        logging.warning("Geocode skipped: No location name provided.")
        return None, None, None
    try:
        gmaps = googlemaps.Client(key=api_key)
        result = gmaps.geocode(location_name)
        if result:
            first = result[0]
            lat = first['geometry']['location']['lat']
            lon = first['geometry']['location']['lng']
            addr = first['formatted_address']
            logging.info(f"Geocoding success: '{location_name[:50]}...' -> {addr}")
            return lat, lon, addr
        else:
            logging.warning(f"Geocoding returned no results for '{location_name[:50]}...'")
            return None, None, None
    except Exception as e:
        logging.error(f"Geocode API Error ('{location_name[:50]}...'): {e}", exc_info=False)
        return None, None, None

# --- NLP ---
def process_text_with_nlp(text):
    # Check NLP model status *inside* the function
    if not NLP:
        logging.warning("Cannot process text, spaCy model (NLP) is not loaded.")
        return []
    if not text:
        return []

    locations = []
    logging.info(f"--- NLP Analysis on: '{text[:100]}...' ---")
    try:
        doc = NLP(text)
        for ent in doc.ents:
            if ent.label_ in LOCATION_ENTITY_LABELS:
                logging.info(f"  Found NLP entity: '{ent.text}' ({ent.label_})")
                locations.append(ent.text)
        if not locations:
            logging.info("  No location entities found by NLP.")
    except Exception as e:
        logging.error(f"Error during NLP processing: {e}", exc_info=True)

    logging.info("--- End NLP Analysis ---")
    return locations

# --- Vision API ---
def analyze_image_with_vision_api(image_path):
    landmark_result, ocr_texts = None, []
    gcp_cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not gcp_cred or not os.path.exists(gcp_cred):
        logging.warning("GOOGLE_APPLICATION_CREDENTIALS invalid/missing. Skipping Vision API.")
        return None, []

    try:
        client = vision.ImageAnnotatorClient()
        if not os.path.exists(image_path):
             # Raise specific error if file doesn't exist at time of call
             raise FileNotFoundError(f"Image not found for Vision API call: {image_path}")

        with open(image_path, 'rb') as f:
            content = f.read()
        image = vision.Image(content=content)

        # Landmark Detection
        logging.info("--- Vision API: Landmark Detection ---")
        resp_lm = client.landmark_detection(image=image)
        if resp_lm.error.message:
            logging.error(f"Vision API Error (Landmark): {resp_lm.error.message}")
        elif resp_lm.landmark_annotations:
            # Process landmarks if no error
            l = resp_lm.landmark_annotations[0] # Take first/most prominent
            logging.info(f"  Detected Landmark: {l.description} (Conf: {l.score:.2f})")
            lat, lon = None, None
            if l.locations:
                # Extract coordinates if available
                loc = l.locations[0].lat_lng
                lat = loc.latitude
                lon = loc.longitude
                logging.info(f"  Landmark Coords: Lat={lat:.6f}, Lon={lon:.6f}")
            # Store result regardless of coordinates
            landmark_result = {'name': l.description, 'latitude': lat, 'longitude': lon, 'confidence': l.score}
        else:
            # No error, but no landmarks found
            logging.info("  No landmarks detected by Vision API.")
        logging.info("--- End Landmark Detection ---")

        # Text Detection (OCR)
        logging.info("--- Vision API: Text Detection (OCR) ---")
        resp_txt = client.text_detection(image=image)
        if resp_txt.error.message:
            logging.error(f"Vision API Error (Text): {resp_txt.error.message}")
        elif resp_txt.text_annotations:
            # Process text if no error
            full_text = resp_txt.text_annotations[0].description
            logging.info(f"  Detected text (snippet): {full_text[:100].replace(os.linesep, ' ')}...")
            ocr_texts.append(full_text.replace('\n', ' ')) # Replace newlines
        else:
            # No error, but no text found
            logging.info("  No text detected by Vision API.")
        logging.info("--- End Text Detection ---")

    except FileNotFoundError as e:
        # Catch if file disappears just before reading
        logging.error(f"{e}")
        return None, []
    except Exception as e:
        # Catch other potential errors (authentication, network, etc.)
        logging.error(f"Unexpected error during Vision API call: {e}", exc_info=True)
        return None, []

    return landmark_result, ocr_texts

# --- Haversine ---
def haversine_distance(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return float('inf') # Return infinity if coordinates are missing
    # Convert decimal degrees to radians
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

# --- Main Processing Function ---
def process_image_geolocation(image_path, associated_text=None): 
    """Processes an image file and optional text to determine geolocation."""
    logging.info(f"--- Starting Geolocation Engine for: {image_path} ---")
    load_dotenv() # Ensure keys are loaded for this run
    gmaps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    # --- Stage 1: Data Extraction ---
    logging.info("--- Stage 1: Data Extraction ---")
    exif_data = get_exif_data(image_path)
    gps_info = get_gps_info(exif_data)
    exif_lat, exif_lon = get_decimal_coordinates(gps_info)
    exif_address = None
    if exif_lat is not None and exif_lon is not None:
        logging.info(f"EXIF Coords Found: Lat={exif_lat:.6f}, Lon={exif_lon:.6f}")
        exif_address = reverse_geocode(exif_lat, exif_lon, gmaps_api_key) # Attempt reverse geocode
        if exif_address: logging.info(f"EXIF Reverse Geocoded: {exif_address}")
        else: logging.info("Could not reverse geocode EXIF coordinates.")
    else:
        logging.info("No usable GPS coords found in EXIF.")

    landmark_info, ocr_texts = analyze_image_with_vision_api(image_path)
    nlp_locations = process_text_with_nlp(associated_text) if associated_text else []

    # --- Stage 2: Geocode Text Clues ---
    logging.info("--- Stage 2: Geocoding Text Clues ---")
    geocoded_nlp_results, geocoded_ocr_results = [], []
    if not gmaps_api_key:
        logging.warning("Skipping text geocoding (No Google Maps API Key).")
    else:
        if nlp_locations:
            logging.info("Geocoding NLP results...")
            for name in nlp_locations:
                lat, lon, addr = geocode_location_name(name, gmaps_api_key)
                if lat is not None: # Check if geocoding succeeded
                    geocoded_nlp_results.append({'name': name, 'latitude': lat, 'longitude': lon, 'address': addr})
        if ocr_texts:
            logging.info("Geocoding OCR results...")
            for text_block in ocr_texts:
                lat, lon, addr = geocode_location_name(text_block, gmaps_api_key)
                if lat is not None: # Check if geocoding succeeded
                    geocoded_ocr_results.append({'text': text_block[:50]+"...", 'latitude': lat, 'longitude': lon, 'address': addr})

    # --- Stage 3: Evidence Fusion & Confidence ---
    logging.info("--- Stage 3: Evidence Fusion & Confidence ---")
    candidate_locations = []

    # Build candidates list (ensure lat/lon exist before adding)
    if exif_lat is not None and exif_lon is not None:
        candidate_locations.append({"source":"EXIF","latitude":exif_lat,"longitude":exif_lon,"address":exif_address,"confidence_raw":1.0})
        logging.info(" Added EXIF candidate.")

    if landmark_info and landmark_info.get('latitude') is not None and landmark_info.get('longitude') is not None:
         if landmark_info.get('confidence', 0) >= LANDMARK_CONFIDENCE_THRESHOLD:
            candidate_locations.append({
                "source": "CV Landmark", "name": landmark_info.get('name'),
                "latitude": landmark_info.get('latitude'), "longitude": landmark_info.get('longitude'),
                "address": None, "confidence_raw": landmark_info.get('confidence', 0)
            })
            logging.info(f" Added Landmark candidate: '{landmark_info.get('name')}'")
         else:
             logging.info(f" Ignoring Landmark: '{landmark_info.get('name')}' (Confidence below threshold)")

    all_text_results = geocoded_nlp_results + geocoded_ocr_results
    if all_text_results:
        # Add first *successful* text result
        first = all_text_results[0]
        origin = "NLP" if 'name' in first else "OCR"
        detail = first.get('name') or first.get('text')
        if first.get('latitude') is not None and first.get('longitude') is not None:
            candidate_locations.append({
                "source": origin, "detail": detail,
                "latitude": first.get('latitude'), "longitude": first.get('longitude'),
                "address": first.get('address'), "confidence_raw": 0.5
            })
            logging.info(f" Added Text candidate ({origin}): '{detail}'")

    # Determine Final Result
    final_latitude, final_longitude, final_address, final_source, final_confidence = None, None, None, "None", 0.0
    map_url = None

    if not candidate_locations:
        logging.info("Decision: No usable location candidates found.")
    else:
        # Define priority score function
        def get_priority(c):
            if c['source'] == 'EXIF': return 3
            if c['source'] == 'CV Landmark': return 2
            return 1 # NLP/OCR

        candidate_locations.sort(key=get_priority, reverse=True)
        best_candidate = candidate_locations[0]
        logging.info(f" Highest priority source: {best_candidate['source']}")

        # Calculate Base Confidence
        base_conf = 0.0
        if best_candidate['source'] == 'EXIF':
            base_conf = 0.9
        elif best_candidate['source'] == 'CV Landmark':
            base_conf = 0.4 + (best_candidate.get('confidence_raw', 0) * 0.4)
        else: # NLP or OCR
            base_conf = 0.5

        # Calculate Consistency Bonus
        bonus = 0.0
        if len(candidate_locations) > 1:
            logging.info(" Checking consistency...")
            for other in candidate_locations[1:]:
                dist = haversine_distance(best_candidate["latitude"], best_candidate["longitude"],
                                          other["latitude"], other["longitude"])
                logging.info(f"  Distance to '{other['source']}': {dist:.2f} km")
                if dist <= CONSISTENCY_THRESHOLD_KM:
                    bonus = 0.15
                    logging.info(f"   -> Consistent. Bonus applied.")
                    break # Stop after first consistent match
            if bonus == 0.0:
                logging.info("  -> No other candidates within consistency threshold.")
        else:
            logging.info(" Only one candidate found, skipping consistency check.")

        # Final confidence score
        final_confidence = min(max(base_conf + bonus, 0.0), 1.0) # Clamp between 0 and 1

        # Assign final values from the best candidate
        final_latitude = best_candidate["latitude"]
        final_longitude = best_candidate["longitude"]
        final_source = best_candidate["source"]
        if best_candidate.get("name"): final_source += f" ('{best_candidate['name']}')"
        elif best_candidate.get("detail"): final_source += f" ('{best_candidate['detail']}')"

        # Determine final address (potentially reverse geocoding landmark)
        final_address = best_candidate.get("address") # Use existing address if available
        if final_address is None and best_candidate['source'] == "CV Landmark":
            logging.info("Attempting reverse geocode for chosen landmark...")
            final_address = reverse_geocode(final_latitude, final_longitude, gmaps_api_key) or best_candidate.get('name', "N/A")
        elif final_address is None: # Default if still none
            final_address = "N/A"

        # Generate Map URL
        if gmaps_api_key and final_latitude is not None:
            map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={final_latitude},{final_longitude}&zoom=13&size=600x300&maptype=roadmap&markers=color:red%7Clabel:P%7C{final_latitude},{final_longitude}&key={gmaps_api_key}"

    # Format confidence label
    confidence_label = "High" if final_confidence >= 0.75 else ("Medium" if final_confidence >= 0.4 else "Low")
    success_flag = final_latitude is not None

    logging.info(f"--- Geolocation Engine Finished. Success: {success_flag}, Source: {final_source}, Confidence: {final_confidence:.2f} ---")

    # Return results as a dictionary
    return {
        "latitude": final_latitude,
        "longitude": final_longitude,
        "address": final_address,
        "source": final_source,
        "confidence_score": final_confidence,
        "confidence_label": confidence_label,
        "map_url": map_url,
        "success": success_flag
    }

# Optional: Direct execution block for testing the engine module itself
if __name__ == '__main__':
    logging.info("geolocation_engine.py executed directly (intended for import).")
    # Add placeholder test logic if needed:
    # load_dotenv()
    # test_image = "path/to/your/test_image.jpg" # Replace with a real path
    # if os.path.exists(test_image):
    #     results = process_image_geolocation(test_image, associated_text="Optional test text")
    #     logging.info("\nDirect Engine Test Results:")
    #     import json
    #     print(json.dumps(results, indent=4))
    # else:
    #      logging.warning(f"Direct test skipped: Test image '{test_image}' not found.")