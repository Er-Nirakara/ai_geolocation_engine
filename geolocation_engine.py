# geolocation_engine.py
# Core logic for the AI-Powered Image Geolocation Engine

import os
import math
import time
import logging
import json
import sys # Keep for early exit if essential models fail

from PIL import Image, ExifTags
import googlemaps
from dotenv import load_dotenv
import spacy
from google.cloud import vision
import requests

# --- Setup Logging ---
# Using basicConfig for simple logging to console.
# For more advanced logging, consider file handlers, rotation, etc.
logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG for more verbose output
    format='%(asctime)s - GEOLOC_ENGINE - %(levelname)s - %(message)s'
)

# --- Constants ---
# Load spaCy NLP Model
NLP = None # Initialize to None in case loading fails
try:
    NLP = spacy.load("en_core_web_lg") # Using a larger model for better accuracy
    logging.info("spaCy NLP model ('en_core_web_lg') loaded successfully.")
except OSError:
    logging.error(
        "Failed to load 'en_core_web_lg' spaCy model. NLP features will be disabled. "
        "To install: python -m spacy download en_core_web_lg"
    )
    # Script can continue, but process_text_with_nlp will return empty if NLP is None

# Configuration for geolocation logic
LOCATION_ENTITY_LABELS = {"GPE", "LOC", "FAC"}  # spaCy entity labels considered as locations
LANDMARK_CONFIDENCE_THRESHOLD = 0.5             # Minimum confidence for Vision API landmarks
CONSISTENCY_THRESHOLD_KM = 10.0                 # Max distance for candidates to be 'consistent'
VERIFICATION_RADIUS_KM = 1.0                    # Search radius for Places & OSM verification
OVERPASS_URL = "https://overpass-api.de/api/interpreter" # OSM Overpass API endpoint
OVERPASS_TIMEOUT = 30                           # Seconds for Overpass requests
PLACES_API_TIMEOUT = 10                         # Seconds for Google Places API requests
API_RETRY_DELAY = 0.05                          # Brief pause between some API calls

# --- Helper Functions ---

# --- EXIF Data Processing ---
def get_exif_data(image_path):
    """Extracts and decodes EXIF data from an image file."""
    image = None
    try:
        logging.info(f"Opening image for EXIF: {image_path}")
        image = Image.open(image_path)
        image.load() # Force load image data to catch errors early
        logging.info(f"Image format identified by Pillow: {image.format}")

        exif_data_raw = image._getexif() # Get raw EXIF
        if not exif_data_raw:
            logging.info("No EXIF metadata found in image.")
            return None

        # Decode EXIF tags to human-readable names
        exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif_data_raw.items()}
        logging.info(f"EXIF data extracted (keys: {len(exif_data)}).")
        return exif_data

    except FileNotFoundError:
        logging.error(f"Image file not found at: {image_path}")
        return None
    except Image.UnidentifiedImageError:
        logging.warning(
            f"Cannot identify image format for: {image_path}. "
            "Ensure it's a valid image and necessary Pillow plugins (e.g., pillow-heif, pillow-avif-plugin) are installed."
        )
        return None
    except Exception as e:
        logging.error(f"Error reading image or EXIF data from {image_path}: {e}", exc_info=True)
        return None
    finally:
        if image:
            image.close()

def get_gps_info(exif_data):
    """Extracts decoded GPS information from the main EXIF data."""
    if not isinstance(exif_data, dict):
        return None
    gps_info_raw = exif_data.get('GPSInfo') # Uses human-readable tag from decoded EXIF
    if not gps_info_raw:
        return None
    # Decode GPS sub-tags
    return {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info_raw.items()}

def dms_to_decimal(dms, ref):
    """Converts GPS Degrees/Minutes/Seconds to decimal degrees."""
    if not isinstance(dms, (tuple, list)) or len(dms) < 3:
        logging.warning(f"Invalid DMS format for conversion: {dms}")
        return None
    try:
        degrees = float(dms[0])
        minutes = float(dms[1]) / 60.0
        seconds = float(dms[2]) / 3600.0
        decimal = degrees + minutes + seconds
        if ref in ['S', 'W']: # Southern and Western hemispheres are negative
            decimal = -decimal
        return decimal
    except (ValueError, TypeError, IndexError) as e:
        logging.warning(f"Error converting DMS component to float ({dms}): {e}")
        return None

def get_decimal_coordinates(gps_info):
    """Extracts latitude and longitude in decimal format from GPS info."""
    if not gps_info: return None, None
    lat_dms, lon_dms = gps_info.get('GPSLatitude'), gps_info.get('GPSLongitude')
    lat_ref, lon_ref = gps_info.get('GPSLatitudeRef'), gps_info.get('GPSLongitudeRef')

    if all([lat_dms, lat_ref, lon_dms, lon_ref]): # Check all parts are present
        lat = dms_to_decimal(lat_dms, lat_ref)
        lon = dms_to_decimal(lon_dms, lon_ref)
        if lat is not None and lon is not None: # Check conversion success
            return lat, lon
        else:
            logging.warning("DMS to Decimal conversion failed for GPS data.")
            return None, None
    return None, None # Missing necessary GPS tags

# --- Geocoding and Reverse Geocoding (Google Maps) ---
def reverse_geocode(lat, lon, api_key):
    """Converts latitude/longitude to a human-readable address."""
    if not api_key: logging.warning("Reverse geocode skipped: No API Key."); return None
    if lat is None or lon is None: logging.warning("Reverse geocode skipped: Invalid coordinates."); return None
    try:
        gmaps = googlemaps.Client(key=api_key, requests_timeout=PLACES_API_TIMEOUT)
        results = gmaps.reverse_geocode((lat, lon))
        return results[0]['formatted_address'] if results else None
    except Exception as e:
        logging.error(f"Reverse Geocode API Error: {e}", exc_info=False) # exc_info=False for brevity
        return None

def geocode_location_name(location_name, api_key):
    """Converts a location name to latitude/longitude coordinates."""
    if not api_key: logging.warning("Geocode skipped: No API Key."); return None, None, None
    if not location_name: logging.warning("Geocode skipped: No location name provided."); return None, None, None
    try:
        gmaps = googlemaps.Client(key=api_key, requests_timeout=PLACES_API_TIMEOUT)
        results = gmaps.geocode(location_name)
        if results:
            loc = results[0]['geometry']['location']
            logging.info(f"Geocoding success: '{location_name[:50]}...' -> {results[0]['formatted_address']}")
            return loc['lat'], loc['lng'], results[0]['formatted_address']
        else:
            logging.warning(f"Geocoding returned no results for '{location_name[:50]}...'")
            return None, None, None
    except Exception as e:
        logging.error(f"Geocode API Error for '{location_name[:50]}...': {e}", exc_info=False)
        return None, None, None

# --- Natural Language Processing (spaCy) ---
def process_text_with_nlp(text):
    """Extracts location-related named entities from text."""
    if not NLP: logging.warning("NLP processing skipped: spaCy model not loaded."); return []
    if not text: return []
    locations = []
    logging.info(f"--- NLP Analysis on text (snippet): '{text[:100]}...' ---")
    try:
        doc = NLP(text)
        for ent in doc.ents:
            if ent.label_ in LOCATION_ENTITY_LABELS:
                logging.info(f"  Found NLP location entity: '{ent.text}' ({ent.label_})")
                locations.append(ent.text)
        if not locations:
            logging.info("  No location entities identified by NLP.")
    except Exception as e:
        logging.error(f"Error during NLP processing: {e}", exc_info=True)
    logging.info("--- End NLP Analysis ---")
    return locations

# --- Computer Vision (Google Cloud Vision API) ---
def analyze_image_with_vision_api(image_path):
    """Analyzes image for landmarks and OCR text using Google Cloud Vision API."""
    landmark_result, ocr_texts = None, []

    # --- AGGRESSIVE DEBUGGING FOR CREDENTIALS ---
    logging.debug("--- Attempting to load GOOGLE_APPLICATION_CREDENTIALS ---") # Changed to DEBUG level
    gcp_cred_path_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    logging.debug(f"Value from os.getenv('GOOGLE_APPLICATION_CREDENTIALS'): '{gcp_cred_path_env}' (Type: {type(gcp_cred_path_env)})")

    if gcp_cred_path_env:
        logging.debug(f"Path from env var exists? {os.path.exists(gcp_cred_path_env)}")
        if os.path.exists(gcp_cred_path_env):
            try:
                with open(gcp_cred_path_env, 'r') as f_test:
                    snippet = f_test.read(50) # Read only a small part
                    logging.debug(f"Successfully read snippet from credentials file: '{snippet}...'")
            except Exception as e_read:
                logging.error(f"Error trying to read credentials file at '{gcp_cred_path_env}': {e_read}")
    else:
        logging.warning("os.getenv('GOOGLE_APPLICATION_CREDENTIALS') returned None or empty string.")
    # --- END AGGRESSIVE DEBUGGING ---

    # Final check on credentials path before attempting to use it
    if not gcp_cred_path_env or not os.path.exists(gcp_cred_path_env):
        logging.warning("Final Check: GOOGLE_APPLICATION_CREDENTIALS path invalid or missing. Skipping Vision API.")
        return None, [] # Return empty results

    try:
        # Explicitly use credentials file
        client = vision.ImageAnnotatorClient.from_service_account_file(gcp_cred_path_env)
        logging.info(f"Vision API Client created using credentials from: {gcp_cred_path_env}")

        if not os.path.exists(image_path):
             raise FileNotFoundError(f"Image file not found for Vision API: {image_path}")

        with open(image_path, 'rb') as f: content = f.read()
        image_vision = vision.Image(content=content) # Renamed to avoid conflict with PIL.Image

        # Landmark Detection
        logging.info("--- Vision API: Landmark Detection ---")
        resp_lm = client.landmark_detection(image=image_vision)
        if resp_lm.error.message:
            logging.error(f"Vision API Error (Landmark): {resp_lm.error.message}")
        elif resp_lm.landmark_annotations:
            l = resp_lm.landmark_annotations[0]
            logging.info(f"  Detected Landmark: {l.description} (Confidence: {l.score:.2f})")
            lat, lon = None, None
            if l.locations:
                loc = l.locations[0].lat_lng
                lat, lon = loc.latitude, loc.longitude
                logging.info(f"  Landmark Coords: Lat={lat:.6f}, Lon={lon:.6f}")
            landmark_result = {'name': l.description, 'latitude': lat, 'longitude': lon, 'confidence': l.score}
        else:
            logging.info("  No landmarks detected by Vision API.")
        logging.info("--- End Landmark Detection ---")

        # Text Detection (OCR)
        logging.info("--- Vision API: Text Detection (OCR) ---")
        resp_txt = client.text_detection(image=image_vision)
        if resp_txt.error.message:
            logging.error(f"Vision API Error (Text): {resp_txt.error.message}")
        elif resp_txt.text_annotations:
            full_text = resp_txt.text_annotations[0].description
            logging.info(f"  Detected text (snippet): {full_text[:100].replace(os.linesep, ' ')}...")
            ocr_texts.append(full_text.replace('\n', ' '))
        else:
            logging.info("  No text detected by Vision API.")
        logging.info("--- End Text Detection ---")

    except FileNotFoundError as e:
        logging.error(f"File not found during Vision API processing: {e}")
        return None, []
    except Exception as e:
        logging.error(f"Unexpected error during Vision API call: {e}", exc_info=True)
        return None, []
    return landmark_result, ocr_texts

# --- Google Places API ---
def query_places_nearby(lat, lon, api_key, radius_meters=500, keyword=None):
    """Queries Google Places API Nearby Search for contextual place information."""
    if not api_key: logging.warning("Places API skipped: No API Key."); return None
    if lat is None or lon is None: logging.warning("Places API skipped: Invalid coordinates."); return None

    logging.info(f"Querying Places API Nearby Search ({lat:.4f},{lon:.4f}), Keyword: {keyword if keyword else 'General Scan'}")
    try:
        gmaps = googlemaps.Client(key=api_key, requests_timeout=PLACES_API_TIMEOUT)
        places_results_api = gmaps.places_nearby(location=(lat, lon), radius=radius_meters, keyword=keyword)
        # logging.debug(f"RAW Places Nearby Response: {places_results_api}") # For deep debugging
        results = places_results_api.get('results', [])

        if not results:
            logging.info("  -> Places Nearby Search returned no results.")
            return False if keyword else [] # Different return for keyword search failure

        if keyword: # Specific keyword search
            for place in results:
                if keyword.lower() in place.get('name', '').lower():
                    logging.info(f"  -> Places MATCH found for '{keyword}': {place.get('name')}")
                    return True # Found a match
            logging.info(f"  -> Places found, but NO direct match for keyword '{keyword}'.")
            return False # Found places, but not the specific one
        else: # General nearby search
            place_names = [p.get('name', 'N/A') for p in results[:3]] # Top 3 names
            logging.info(f"  -> Places found nearby (general): {', '.join(place_names)}...")
            return place_names # Return list of names for general context

    except Exception as e:
        logging.error(f"Google Places API Error (Nearby Search): {e}", exc_info=False)
        return None # Indicate API error

# --- OpenStreetMap (Overpass API) ---
def query_overpass(query):
    """Sends a query to the Overpass API and returns parsed JSON."""
    logging.info(f"Querying Overpass API...")
    # Replace with your actual contact for User-Agent if using extensively
    headers = {'User-Agent': 'AI_Geolocation_Capstone/1.0 (your_email@example.com)'}
    try:
        response = requests.post(OVERPASS_URL, data={'data': query}, timeout=OVERPASS_TIMEOUT, headers=headers)
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        data = response.json()
        logging.info(f"  -> Overpass query successful. Elements: {len(data.get('elements', []))}")
        # logging.debug(f"RAW Overpass Response JSON: {data}") # For deep debugging
        return data
    except requests.exceptions.Timeout:
        logging.error("Overpass API request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Overpass API request failed: {e}", exc_info=False)
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode Overpass JSON response: {e}. Response text: {response.text[:200]}...", exc_info=False)
        return None
    except Exception as e: # Catch-all for other unexpected errors
        logging.error(f"Unexpected error during Overpass query: {e}", exc_info=True)
        return None

def verify_osm_feature(lat, lon, radius_km, feature_tag, feature_value):
    """Checks if a feature with a specific tag/value exists near coordinates using Overpass."""
    if lat is None or lon is None or not feature_tag or not feature_value: return False

    radius_m = radius_km * 1000
    escaped_value = feature_value.replace('"', '\\"') # Basic escaping for quotes

    # Efficient query to check for existence (count)
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["{feature_tag}"="{escaped_value}"](around:{radius_m},{lat},{lon});
      way["{feature_tag}"="{escaped_value}"](around:{radius_m},{lat},{lon});
      relation["{feature_tag}"="{escaped_value}"](around:{radius_m},{lat},{lon});
    );
    out count;
    """
    result_data = query_overpass(overpass_query)

    if result_data and result_data.get('elements'):
        # 'out count;' returns elements, their number indicates matches.
        # For 'out count;', the elements themselves don't have full data, just counts per type.
        # A simpler check is just if any elements are returned.
        if len(result_data['elements']) > 0:
             # If Overpass returns any element, it means some feature matched.
             # The actual count details are in the 'tags' of these elements, e.g. elements[0]['tags']['total']
             # For simplicity, any element returned means a match for now.
            logging.info(f"  -> OSM SUCCESS: Found feature ['{feature_tag}'='{feature_value}'] near {lat:.4f},{lon:.4f}.")
            return True
    # If result_data is None (query failed) or no elements found
    logging.info(f"  -> OSM FAILED/SKIPPED: No features ['{feature_tag}'='{feature_value}'] near {lat:.4f},{lon:.4f}.")
    return False

# --- Haversine Distance ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance between two lat/lon points in kilometers."""
    if None in [lat1, lon1, lat2, lon2]: return float('inf')
    # Convert to radians
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_r - lon1_r; dlat = lat2_r - lat1_r
    a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)); R_KM = 6371
    return c * R_KM

# --- Main Processing Function ---
def process_image_geolocation(image_path, associated_text=None):
    """Main function to process image and text for geolocation."""
    logging.info(f"--- Starting Geolocation Engine (Advanced Fusion) for: {image_path} ---")
    load_dotenv() # Load .env variables for this execution context
    gmaps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    # --- Stage 1: Data Extraction ---
    logging.info("--- Stage 1: Data Extraction ---")
    exif_data = get_exif_data(image_path)
    gps_info = get_gps_info(exif_data)
    exif_lat, exif_lon = get_decimal_coordinates(gps_info)
    exif_address = None
    if exif_lat is not None and exif_lon is not None:
        logging.info(f"EXIF Coords Found: Lat={exif_lat:.6f}, Lon={exif_lon:.6f}")
        exif_address = reverse_geocode(exif_lat, exif_lon, gmaps_api_key)
        if exif_address: logging.info(f"EXIF Address: {exif_address}")
    else: logging.info("No usable EXIF GPS data found.")

    landmark_info, ocr_texts = analyze_image_with_vision_api(image_path)
    nlp_locations = process_text_with_nlp(associated_text) if associated_text else []

    # --- Stage 2: Geocode Text Clues ---
    logging.info("--- Stage 2: Geocoding Text Clues ---")
    geocoded_nlp_results, geocoded_ocr_results = [], []
    if not gmaps_api_key: logging.warning("Text geocoding skipped (No Google Maps API Key).")
    else:
        if nlp_locations:
            logging.info("Geocoding NLP results...")
            for name in nlp_locations:
                lat, lon, addr = geocode_location_name(name, gmaps_api_key)
                if lat is not None and lon is not None:
                    geocoded_nlp_results.append({'name': name, 'latitude': lat, 'longitude': lon, 'address': addr})
        if ocr_texts:
            logging.info("Geocoding OCR results...")
            for text_block in ocr_texts:
                lat, lon, addr = geocode_location_name(text_block, gmaps_api_key)
                if lat is not None and lon is not None:
                    geocoded_ocr_results.append({'text': text_block[:50]+"...", 'latitude': lat, 'longitude': lon, 'address': addr})

    # --- Stage 3: Advanced Fusion & Verification ---
    logging.info("--- Stage 3: Advanced Fusion & Verification ---")
    candidates = []
    logging.info("  3a: Collecting initial candidates...")
    # Populate candidates list (only add if coordinates are valid)
    if exif_lat is not None and exif_lon is not None:
        candidates.append({"id": "exif_0", "source": "EXIF", "latitude": exif_lat, "longitude": exif_lon, "address": exif_address, "raw_score": 1.0})
        logging.info(f"    + EXIF Candidate Added.")
    if landmark_info and landmark_info.get('latitude') is not None and landmark_info.get('longitude') is not None:
        candidates.append({"id": "lm_0", "source": "CV Landmark", "latitude": landmark_info['latitude'], "longitude": landmark_info['longitude'], "name": landmark_info.get('name'), "raw_score": landmark_info.get('confidence', 0)})
        logging.info(f"    + Landmark Candidate Added: {landmark_info.get('name')}")
    text_idx = 0
    for res in geocoded_nlp_results:
        if res.get('latitude') is not None and res.get('longitude') is not None:
            candidates.append({"id": f"nlp_{text_idx}", "source": "NLP", "latitude": res['latitude'], "longitude": res['longitude'], "name": res.get('name'), "address": res.get('address'), "raw_score": 0.5})
            logging.info(f"    + NLP Candidate Added: {res.get('name')}"); text_idx+=1
    for res in geocoded_ocr_results:
        if res.get('latitude') is not None and res.get('longitude') is not None:
            candidates.append({"id": f"ocr_{text_idx}", "source": "OCR", "latitude": res['latitude'], "longitude": res['longitude'], "text": res.get('text'), "address": res.get('address'), "raw_score": 0.4})
            logging.info(f"    + OCR Candidate Added: {res.get('text')}"); text_idx+=1

    if not candidates:
        logging.warning("  No valid location candidates found after initial collection.")
        return {"success": False, "source": "None", "confidence_score": 0.0, "address":None, "latitude":None, "longitude":None, "map_url":None, "confidence_label":"Low"}

    logging.info(f"  3b: Performing verification checks on {len(candidates)} candidates...")
    for cand in candidates:
        cand['places_nearby_keyword_match'] = None; cand['osm_verified_name'] = None; cand['osm_verified_street'] = None
        lat, lon = cand.get('latitude'), cand.get('longitude')
        # This check is crucial: skip verification if a candidate somehow has no coords
        if lat is None or lon is None:
            logging.warning(f"  Skipping verification for candidate {cand.get('id')} due to missing coordinates.")
            continue

        keyword_to_check = cand.get('name') or cand.get('text') # For Places and OSM name check

        # Places API check for keyword
        if gmaps_api_key and keyword_to_check:
            places_match = query_places_nearby(lat, lon, gmaps_api_key, radius_meters=VERIFICATION_RADIUS_KM*1000, keyword=keyword_to_check)
            cand['places_nearby_keyword_match'] = places_match # True, False, or None
            time.sleep(API_RETRY_DELAY)

        # OSM Name Check
        if keyword_to_check:
            osm_name_match = verify_osm_feature(lat, lon, VERIFICATION_RADIUS_KM, "name", keyword_to_check)
            cand['osm_verified_name'] = osm_name_match
            time.sleep(API_RETRY_DELAY)

        # OSM Street Check (primarily for OCR)
        potential_street = None
        if cand['source'] == 'OCR' and cand.get('text'):
            ocr_lower = cand['text'].lower()
            street_keywords = [' street', ' road', ' avenue', ' st', ' rd', ' ave', ' lane', ' dr', ' blvd', ' way', ' court', ' square']
            if any(kw in ocr_lower for kw in street_keywords):
                # This is a simplistic extraction; more advanced NLP on OCR could be better
                potential_street = cand['text'] # Use the full text block for now
                logging.info(f"    Identified potential street in OCR for {cand['id']}: '{potential_street[:50]}...'")

        if potential_street:
            osm_street_match_hwy = verify_osm_feature(lat, lon, VERIFICATION_RADIUS_KM, "highway", potential_street); time.sleep(API_RETRY_DELAY)
            osm_street_match_name_tag = verify_osm_feature(lat, lon, VERIFICATION_RADIUS_KM, "name", potential_street); time.sleep(API_RETRY_DELAY)
            cand['osm_verified_street'] = osm_street_match_hwy or osm_street_match_name_tag

    logging.info("  3c: Scoring candidates...")
    for cand in candidates:
        score = 0.0
        raw_score = cand.get('raw_score', 0.0)
        # Base score
        if cand['source'] == 'EXIF': score = raw_score * 0.95 # Slightly higher base for EXIF
        elif cand['source'] == 'CV Landmark': score = 0.4 + (raw_score * 0.45) # Max ~0.85 from raw
        elif cand['source'] == 'NLP': score = raw_score * 0.5 # raw_score is 0.5
        elif cand['source'] == 'OCR': score = raw_score * 0.4 # raw_score is 0.4

        # Verification Bonuses
        if cand.get('places_nearby_keyword_match') is True: score += 0.15; logging.info(f"    Score +0.15 (Places Verified) for {cand['id']}")
        if cand.get('osm_verified_name') is True: score += 0.20; logging.info(f"    Score +0.20 (OSM Name Verified) for {cand['id']}")
        if cand.get('osm_verified_street') is True: score += 0.15; logging.info(f"    Score +0.15 (OSM Street Verified) for {cand['id']}")

        # Proximity Bonus
        proximity_bonus = 0.0
        num_close_neighbors = 0
        if len(candidates) > 1 and cand.get('latitude') is not None: # Check coords again
            for other in candidates:
                if cand['id'] == other['id']: continue
                if other.get('latitude') is not None:
                    distance = haversine_distance(cand['latitude'], cand['longitude'], other['latitude'], other['longitude'])
                    if distance <= CONSISTENCY_THRESHOLD_KM:
                        num_close_neighbors += 1
            if num_close_neighbors > 0:
                proximity_bonus = 0.10 # Add small bonus if other candidates are close
                logging.info(f"    Score +{proximity_bonus:.2f} (Proximity: {num_close_neighbors} neighbors) for {cand['id']}")

        cand['final_score'] = min(max(score + proximity_bonus, 0.0), 1.0) # Clamp to 0-1
        logging.info(f"    Candidate {cand['id']} ({cand['source']}) -> Final Score: {cand['final_score']:.3f}")

    # Select best candidate
    candidates.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
    best_candidate = candidates[0]

    # --- Assign Final Values ---
    final_latitude = best_candidate.get("latitude")
    final_longitude = best_candidate.get("longitude")
    final_confidence = best_candidate.get("final_score", 0.0)
    final_source = best_candidate["source"]
    # Append name/text detail to source string for clarity
    if best_candidate.get("name"): final_source += f" ('{best_candidate['name']}')"
    elif best_candidate.get("text"): final_source += f" ('{best_candidate['text']}')"

    # Determine final address
    final_address = best_candidate.get("address")
    if final_address is None and best_candidate['source'] in ["EXIF", "CV Landmark"]:
        if final_latitude is not None and final_longitude is not None:
            logging.info("Attempting final reverse geocode for best candidate...")
            final_address = reverse_geocode(final_latitude, final_longitude, gmaps_api_key)
    # Fallback if address is still None
    if final_address is None: final_address = "N/A"


    # Generate Map URL
    map_url = None
    if gmaps_api_key and final_latitude is not None and final_longitude is not None:
        map_url = (f"https://maps.googleapis.com/maps/api/staticmap?center={final_latitude},{final_longitude}"
                   f"&zoom=13&size=600x300&maptype=roadmap"
                   f"&markers=color:red%7Clabel:P%7C{final_latitude},{final_longitude}&key={gmaps_api_key}")

    confidence_label = "High" if final_confidence >= 0.75 else ("Medium" if final_confidence >= 0.4 else "Low")
    success_flag = final_latitude is not None and final_longitude is not None

    logging.info(f"--- Geolocation Engine Finished. Success: {success_flag}, Source: {final_source}, Confidence: {final_confidence:.2f} ---")

    return {
        "latitude": final_latitude, "longitude": final_longitude, "address": final_address,
        "source": final_source, "confidence_score": final_confidence,
        "confidence_label": confidence_label, "map_url": map_url, "success": success_flag
    }

# --- Optional Direct Execution Block ---
if __name__ == '__main__':
     logging.info("geolocation_engine.py executed directly (intended for import).")
     # To test directly (uncomment and modify):
     # load_dotenv()
     # test_image_path = "path_to_your_test_image.jpg" # IMPORTANT: Replace with a real image path
     # if os.path.exists(test_image_path):
     #     results = process_image_geolocation(test_image_path, associated_text="Optional test caption here")
     #     print("\n--- Direct Engine Test Results ---")
     #     print(json.dumps(results, indent=2))
     # else:
     #     logging.error(f"Test image not found: {test_image_path}")