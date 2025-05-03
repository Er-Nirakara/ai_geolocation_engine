import os
import argparse
import googlemaps # Make sure google-maps-services-python is installed
from dotenv import load_dotenv # Make sure python-dotenv is installed
import sys
from google.cloud import vision # Make sure google-cloud-vision is installed

# --- Constants ---
# Confidence threshold for considering a landmark detection "high confidence"
LANDMARK_CONFIDENCE_THRESHOLD = 0.5 # Adjust as needed (0.0 to 1.0)

# --- Helper Functions --- #

# --- Geocoding Functions (Needed for Vision results) ---

def reverse_geocode(lat, lon, api_key):
    """Gets address from latitude and longitude using Google Maps API."""
    if not api_key:
        print("Error: Google Maps API key not found for reverse geocoding.")
        return None
    if lat is None or lon is None:
        return None

    try:
        gmaps = googlemaps.Client(key=api_key)
        reverse_geocode_result = gmaps.reverse_geocode((lat, lon))
        if reverse_geocode_result:
            return reverse_geocode_result[0]['formatted_address'] # Return only address string
        else:
            print(f"Reverse geocoding returned no results for {lat},{lon}.")
            return None
    except googlemaps.exceptions.ApiError as e:
        print(f"Google Maps API Error (Reverse Geocode): {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred (Reverse Geocode): {e}")
        return None

def geocode_location_name(location_name, api_key):
    """Gets coordinates and address for a location name using Google Maps API."""
    if not api_key:
        print("Error: Google Maps API key not found for geocoding.")
        return None, None, None # lat, lon, address
    if not location_name:
        return None, None, None

    try:
        gmaps = googlemaps.Client(key=api_key)
        # Limit results to potentially improve relevance if OCR text is messy
        geocode_result = gmaps.geocode(location_name)

        if geocode_result:
            # Take the first result
            first_result = geocode_result[0]
            lat = first_result['geometry']['location']['lat']
            lon = first_result['geometry']['location']['lng']
            address = first_result['formatted_address']
            print(f"Geocoding successful for '{location_name[:50]}...': {address} ({lat:.4f}, {lon:.4f})")
            return lat, lon, address
        else:
            print(f"Geocoding returned no results for '{location_name[:50]}...'.")
            return None, None, None
    except googlemaps.exceptions.ApiError as e:
        print(f"Google Maps API Error (Geocode for '{location_name[:50]}...'): {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred (Geocode for '{location_name[:50]}...'): {e}")
        return None, None, None

# --- Vision API Functions ---

def analyze_image_with_vision_api(image_path):
    """Analyzes image for landmarks and text using Google Cloud Vision API."""
    landmark_result = None
    ocr_texts = []

    # Ensure credentials are set before attempting API call
    gcp_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not gcp_credentials_path or not os.path.exists(gcp_credentials_path):
        print("Error: GOOGLE_APPLICATION_CREDENTIALS not set or path invalid. Cannot use Vision API.")
        return None, [] # Return empty results immediately

    try:
        client = vision.ImageAnnotatorClient() # Uses GOOGLE_APPLICATION_CREDENTIALS implicitly
        # Check if file exists before opening
        if not os.path.exists(image_path):
             raise FileNotFoundError(f"Image file not found at path: {image_path}")

        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        # --- Landmark Detection ---
        print("\n--- Vision API: Landmark Detection ---")
        response_landmark = client.landmark_detection(image=image) # Separate response variable

        # Check for API errors specifically for landmark detection
        if response_landmark.error.message:
             print(f"Vision API Error (Landmark Detection): {response_landmark.error.message}")
        else:
            landmarks = response_landmark.landmark_annotations
            if landmarks:
                l = landmarks[0] # Usually, the first landmark is the most prominent
                print(f"  Detected Landmark: {l.description} (Confidence: {l.score:.2f})")
                if l.locations:
                    loc = l.locations[0].lat_lng # Take the first location point
                    print(f"  Landmark Coordinates: Lat={loc.latitude:.6f}, Lon={loc.longitude:.6f}")
                    landmark_result = {
                        'name': l.description,
                        'latitude': loc.latitude,
                        'longitude': loc.longitude,
                        'confidence': l.score
                    }
                else:
                    print("  Landmark detected but no coordinates provided by API.")
                    landmark_result = {
                        'name': l.description,
                        'latitude': None,
                        'longitude': None,
                        'confidence': l.score
                    }
            else:
                print("  No landmarks detected.")
        print("--- End Landmark Detection ---")


        # --- Text Detection (OCR) ---
        print("\n--- Vision API: Text Detection (OCR) ---")
        response_text = client.text_detection(image=image) # Separate response variable

        # Check for API errors specifically for text detection
        if response_text.error.message:
            print(f"Vision API Error (Text Detection): {response_text.error.message}")
        else:
            texts = response_text.text_annotations
            if texts:
                full_text = texts[0].description
                print(f"  Detected text block (snippet): {full_text[:100].replace(os.linesep, ' ')}...")
                ocr_texts.append(full_text.replace('\n', ' ')) # Replace newlines for easier processing
            else:
                print("  No text detected.")
        print("--- End Text Detection ---")

    except FileNotFoundError as e: # Specifically catch file not found
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during Google Cloud Vision API processing: {e}")
        # Could be authentication, quota, network issue etc.

    return landmark_result, ocr_texts


# --- Main Execution ---

def main():
    # Load environment variables (.env file)
    load_dotenv()
    gmaps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    gcp_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # Check essential credentials/keys
    credentials_ok = True
    if not gcp_credentials_path:
         print("Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
         print("         Vision API calls will be skipped.")
         credentials_ok = False
    elif not os.path.exists(gcp_credentials_path):
         print(f"Warning: GOOGLE_APPLICATION_CREDENTIALS path does not exist: {gcp_credentials_path}")
         print("         Vision API calls will be skipped.")
         credentials_ok = False

    if not gmaps_api_key:
        print("Warning: GOOGLE_MAPS_API_KEY not found. Geocoding/Reverse Geocoding will fail.")
        # Allow to continue, but dependent functionality will be broken.

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Geolocation using Vision API (Landmarks, OCR).")
    parser.add_argument("image_path", help="Path to the image file (JPEG/PNG/etc.)")
    # Removed the --text argument
    args = parser.parse_args()

    print(f"\nProcessing image: {args.image_path}")

    # --- Stage 1: Vision API Analysis (Landmark & OCR) ---
    landmark_info = None
    ocr_texts = []
    if credentials_ok: # Only call Vision API if credentials seem okay
        landmark_info, ocr_texts = analyze_image_with_vision_api(args.image_path)
    else:
        print("\nSkipping Vision API analysis due to credential issues.")

    # --- Stage 2: Geocode OCR Text Results ---
    geocoded_ocr_results = [] # List of dicts: {'text': ..., 'latitude': ..., 'longitude': ..., 'address': ...}
    print("\n--- 2. Geocoding Text Clues (OCR) ---")
    if not gmaps_api_key:
         print("Skipping text geocoding (No Google Maps API Key).")
    elif not ocr_texts:
        print("No OCR text found to geocode.")
    else:
        print("Geocoding text found in image (OCR)...")
        for text_block in ocr_texts:
             lat, lon, addr = geocode_location_name(text_block, gmaps_api_key)
             if lat is not None:
                  # Store dict for consistency
                  geocoded_ocr_results.append({'text': text_block[:50]+"...", 'latitude': lat, 'longitude': lon, 'address': addr})
             # else: geocoding failed for this block, ignore

    print("--- End Geocoding ---")


    # --- Stage 3: Evidence Fusion (Landmark > OCR) ---
    print("\n--- 3. Evidence Fusion ---")
    final_latitude = None
    final_longitude = None
    final_address = None
    final_source = "None"

    # Priority 1: High-Confidence Landmark
    if landmark_info and landmark_info.get('latitude') is not None and landmark_info.get('confidence', 0) >= LANDMARK_CONFIDENCE_THRESHOLD:
        final_latitude = landmark_info['latitude']
        final_longitude = landmark_info['longitude']
        # Try reverse geocoding the landmark coords for a better address, if key exists
        lm_address = None
        if gmaps_api_key:
            lm_address = reverse_geocode(final_latitude, final_longitude, gmaps_api_key)
        final_address = lm_address if lm_address else landmark_info['name'] # Fallback to landmark name
        final_source = f"CV Landmark ('{landmark_info['name']}', Conf: {landmark_info['confidence']:.2f})"
        print(f"Decision: Using High-Confidence Landmark '{landmark_info['name']}'.")

    # Priority 2: Geocoded OCR Text (if Landmark not used)
    elif geocoded_ocr_results:
        # This block runs if high-confidence landmark was missing/failed
        print("High-Confidence Landmark not available or below threshold.")
        # Simplistic: Take the first successful geocoding result from OCR
        first_result = geocoded_ocr_results[0]
        final_latitude = first_result['latitude']
        final_longitude = first_result['longitude']
        final_address = first_result['address'] # Address from geocoding
        source_detail = first_result.get('text', 'OCR Text')
        final_source = f"OCR ('{source_detail}')"
        print(f"Decision: Using first geocoded OCR text result.")

    # Final Check: If after all priorities, no source was found
    else:
        # This case occurs if no high-conf landmark AND no successful OCR geocoding
        print("Decision: No usable location found from Vision API (Landmark/OCR).")


    # --- Final Output ---
    print("\n" + "=" * 30)
    print("      FINAL RESULT")
    print("=" * 30)
    print(f"Source of Prediction: {final_source}")

    if final_latitude is not None and final_longitude is not None:
        print(f"Predicted Coordinates: Lat={final_latitude:.6f}, Lon={final_longitude:.6f}")
        if final_address:
            print(f"Predicted Address:     {final_address}")
        else:
            # This might happen if landmark coords are found but reverse geocoding fails/is skipped
            print("Predicted Address:     Could not determine address name.")
    else:
        print("Could not determine location from available sources (Landmark/OCR).")

    print("=" * 30 + "\n")

if __name__ == "__main__":
    main()