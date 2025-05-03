import os
import argparse
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import googlemaps
from dotenv import load_dotenv
import spacy # <-- Added
import sys # <-- Added for error exit

# --- Constants ---
# Load the spaCy model once
try:
    NLP = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    NLP = None # Set to None if loading fails

# Location entity types to consider
LOCATION_ENTITY_LABELS = {"GPE", "LOC", "FAC"} # GPE (Geo-Political Entity), LOC (Location), FAC (Facility)


# --- Helper Functions (Keep existing EXIF functions) ---

def get_exif_data(image_path):
    # (Keep function from Patch 1 - no changes needed)
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            # print(f"No EXIF metadata found in {image_path}") # Less verbose now
            return None
        return exif_data
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error reading image or EXIF data: {e}")
        return None

def get_gps_info(exif_data):
    # (Keep function from Patch 1 - no changes needed)
    if not exif_data:
        return None
    gps_info = {}
    for k, v in TAGS.items():
        if v == 'GPSInfo':
            if k in exif_data:
                for gps_k, gps_v in GPSTAGS.items():
                    if gps_k in exif_data[k]:
                        gps_info[gps_v] = exif_data[k][gps_k]
                return gps_info
            else:
                break
    # print("No GPSInfo tag found in EXIF data.") # Less verbose
    return None

def dms_to_decimal(dms, ref):
    # (Keep function from Patch 1 - no changes needed)
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    decimal = degrees + minutes + seconds
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_decimal_coordinates(gps_info):
    # (Keep function from Patch 1 - no changes needed)
    if not gps_info:
        return None, None
    lat = None
    lon = None
    lat_dms = gps_info.get('GPSLatitude')
    lat_ref = gps_info.get('GPSLatitudeRef')
    lon_dms = gps_info.get('GPSLongitude')
    lon_ref = gps_info.get('GPSLongitudeRef')
    if lat_dms and lat_ref and lon_dms and lon_ref:
        lat = dms_to_decimal(lat_dms, lat_ref)
        lon = dms_to_decimal(lon_dms, lon_ref)
        return lat, lon
    else:
        # print("GPS coordinates missing required tags.") # Less verbose
        return None, None

def reverse_geocode(lat, lon, api_key):
    # (Keep function from Patch 1 - slight modification for clarity)
    if not api_key:
        print("Error: Google Maps API key not found.")
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

# --- New Functions for Patch 2 ---

def process_text_with_nlp(text):
    """Extracts potential location entities from text using spaCy."""
    if not NLP:
        print("spaCy model not loaded, cannot process text.")
        return []
    if not text:
        return []

    doc = NLP(text)
    locations = []
    print("\n--- NLP Analysis ---")
    print(f"Text processed: '{text}'")
    for ent in doc.ents:
        if ent.label_ in LOCATION_ENTITY_LABELS:
            print(f"  Found potential location entity: '{ent.text}' ({ent.label_})")
            locations.append(ent.text)
        # else:
        #     print(f"  Ignoring entity: '{ent.text}' ({ent.label_})") # Optional: for debugging

    if not locations:
        print("  No location-related entities found in text.")
    print("--- End NLP Analysis ---")
    return locations # Return list of names

def geocode_location_name(location_name, api_key):
    """Gets coordinates and address for a location name using Google Maps API."""
    if not api_key:
        print("Error: Google Maps API key not found.")
        return None, None, None # lat, lon, address
    if not location_name:
        return None, None, None

    try:
        gmaps = googlemaps.Client(key=api_key)
        geocode_result = gmaps.geocode(location_name)

        if geocode_result:
            # Take the first result
            first_result = geocode_result[0]
            lat = first_result['geometry']['location']['lat']
            lon = first_result['geometry']['location']['lng']
            address = first_result['formatted_address']
            print(f"Geocoding successful for '{location_name}': {address} ({lat:.4f}, {lon:.4f})")
            return lat, lon, address
        else:
            print(f"Geocoding returned no results for '{location_name}'.")
            return None, None, None
    except googlemaps.exceptions.ApiError as e:
        print(f"Google Maps API Error (Geocode for '{location_name}'): {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred (Geocode for '{location_name}'): {e}")
        return None, None, None


# --- Main Execution (Modified) ---

def main():
    # Check if spaCy model loaded correctly
    if not NLP:
        sys.exit(1) # Exit if spaCy isn't ready

    # Load API Key
    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("Error: GOOGLE_MAPS_API_KEY not found in environment or .env file.")
        print("Please ensure it's set correctly.")
        sys.exit(1) # Exit if API key is missing

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Extract location from image EXIF or associated text.")
    parser.add_argument("image_path", help="Path to the image file (JPEG/PNG)")
    parser.add_argument("--text", help="Optional text associated with the image (e.g., caption)", default=None) # <-- Added optional text argument
    args = parser.parse_args()

    print(f"\nProcessing image: {args.image_path}")
    if args.text:
        print(f"Associated text: '{args.text}'")

    # --- Data Extraction ---
    final_latitude = None
    final_longitude = None
    final_address = None
    final_source = "None" # Track where the final result came from

    # 1. Try EXIF Data First
    print("\n--- EXIF Analysis ---")
    exif_data = get_exif_data(args.image_path)
    gps_info = get_gps_info(exif_data)
    exif_lat, exif_lon = get_decimal_coordinates(gps_info)

    if exif_lat is not None and exif_lon is not None:
        print(f"Found EXIF Coordinates: Lat={exif_lat:.6f}, Lon={exif_lon:.6f}")
        exif_address = reverse_geocode(exif_lat, exif_lon, api_key)
        if exif_address:
            print(f"Reverse Geocoded Address: {exif_address}")
            # Prioritize EXIF
            final_latitude = exif_lat
            final_longitude = exif_lon
            final_address = exif_address
            final_source = "EXIF"
        else:
            # Use coordinates even if reverse geocoding fails
            final_latitude = exif_lat
            final_longitude = exif_lon
            final_source = "EXIF (Coordinates Only)"
            print("Could not reverse geocode EXIF coordinates, but using coordinates.")
    else:
        print("No usable GPS coordinates found in EXIF data.")
    print("--- End EXIF Analysis ---")


    # 2. Try NLP on Text if EXIF didn't yield a result AND text is provided
    if final_source == "None" and args.text:
        potential_locations = process_text_with_nlp(args.text)

        if potential_locations:
            # Try geocoding the first potential location found by NLP
            # (More sophisticated logic could try multiple or rank them later)
            first_location_name = potential_locations[0]
            print(f"\nAttempting to geocode first potential location from NLP: '{first_location_name}'")
            nlp_lat, nlp_lon, nlp_address = geocode_location_name(first_location_name, api_key)

            if nlp_lat is not None and nlp_lon is not None:
                # Use the first successful NLP result
                final_latitude = nlp_lat
                final_longitude = nlp_lon
                final_address = nlp_address # Use the address from geocoding
                final_source = f"NLP ('{first_location_name}')"
            else:
                print(f"Could not geocode the location '{first_location_name}' found via NLP.")
        # else: No locations found by NLP - handled within process_text_with_nlp


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
            print("Predicted Address:     Could not determine address.")
    else:
        print("Could not determine location from available sources (EXIF, Text).")

    print("=" * 30 + "\n")


if __name__ == "__main__":
    main()