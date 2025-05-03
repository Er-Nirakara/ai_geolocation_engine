import os
import argparse
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import googlemaps
from dotenv import load_dotenv

# --- Helper Functions ---

def get_exif_data(image_path):
    """Extracts EXIF data from an image file."""
    try:
        image = Image.open(image_path)
        # image.verify() # Use verify() if you need to check for corrupted images
        exif_data = image._getexif()
        if not exif_data:
            print(f"No EXIF metadata found in {image_path}")
            return None
        return exif_data
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error reading image or EXIF data: {e}")
        return None

def get_gps_info(exif_data):
    """Extracts GPS Info dictionary from raw EXIF data."""
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
                break # Found GPSInfo tag name but no data in image
    print("No GPSInfo tag found in EXIF data.")
    return None

def dms_to_decimal(dms, ref):
    """Converts Degrees/Minutes/Seconds + Ref to Decimal Degrees."""
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    decimal = degrees + minutes + seconds
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_decimal_coordinates(gps_info):
    """Extracts latitude and longitude in decimal format."""
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
        print("GPS coordinates missing required tags (Lat/Lon DMS or Ref).")
        return None, None

def reverse_geocode(lat, lon, api_key):
    """Gets address from latitude and longitude using Google Maps API."""
    if not api_key:
        print("Error: Google Maps API key not found. Set GOOGLE_MAPS_API_KEY in .env file.")
        return None
    if lat is None or lon is None:
        return None

    try:
        gmaps = googlemaps.Client(key=api_key)
        reverse_geocode_result = gmaps.reverse_geocode((lat, lon))

        if reverse_geocode_result:
            # Return the most specific address found
            return reverse_geocode_result[0]['formatted_address']
        else:
            print("Reverse geocoding returned no results.")
            return None
    except googlemaps.exceptions.ApiError as e:
        print(f"Google Maps API Error during reverse geocoding: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during reverse geocoding: {e}")
        return None

# --- Main Execution ---

def main():
    # Load API Key from .env file
    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Extract GPS location from image EXIF data.")
    parser.add_argument("image_path", help="Path to the image file (JPEG/PNG)")
    args = parser.parse_args()

    print(f"Processing image: {args.image_path}")

    # 1. Extract EXIF
    exif_data = get_exif_data(args.image_path)

    # 2. Extract GPS Info
    gps_info = get_gps_info(exif_data)

    # 3. Get Decimal Coordinates
    latitude, longitude = get_decimal_coordinates(gps_info)

    # 4. Reverse Geocode
    address = None
    if latitude is not None and longitude is not None:
        print(f"Found Coordinates: Lat={latitude:.6f}, Lon={longitude:.6f} (Source: EXIF)")
        address = reverse_geocode(latitude, longitude, api_key)
    else:
        print("Could not determine coordinates from EXIF data.")

    # 5. Display Results
    if address:
        print(f"Estimated Address: {address} (Source: EXIF + Reverse Geocoding)")
    elif latitude is not None:
         print("Could not retrieve address from coordinates.")

    print("-" * 20) # Separator

if __name__ == "__main__":
    main()