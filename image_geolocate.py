# image_geolocate.py
# Command-Line Interface for the Image Geolocation Engine

import argparse
import os
import sys
from dotenv import load_dotenv

# Attempt to import the core processing function
try:
    from geolocation_engine import process_image_geolocation
except ImportError:
    # Provide a helpful error message if the engine module isn't found
    print("Error: Could not import 'process_image_geolocation' from 'geolocation_engine.py'.")
    print("Please ensure 'geolocation_engine.py' is in the same directory as this script.")
    sys.exit(1) # Exit if the core component is missing

def run_geolocation_cli():
    """Handles command-line arguments and calls the geolocation engine."""

    # Load environment variables from .env file.
    # This is important for the engine to access API keys.
    load_dotenv()

    # Set up argument parsing for command-line options
    parser = argparse.ArgumentParser(
        description="AI-Powered Image Geolocation Engine: Command-Line Tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Full path to the image file to be geolocated."
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Optional text associated with the image (e.g., a caption or description)."
    )
    args = parser.parse_args()

    # Validate that the provided image path actually exists
    if not os.path.exists(args.image_path):
        print(f"Error: The image file was not found at the specified path: {args.image_path}")
        sys.exit(1)

    print(f"\n--- Geolocation Process Initiated for: {os.path.basename(args.image_path)} ---")
    if args.text:
        print(f"--- Associated Text: '{args.text[:100]}{'...' if len(args.text) > 100 else ''}' ---")

    # Call the main engine function with the image path and any associated text
    # The engine itself will handle internal logging of its steps.
    try:
        results = process_image_geolocation(args.image_path, args.text)
    except Exception as e:
        print(f"\nAn unexpected error occurred while processing the image: {e}")
        # Consider logging the full traceback here for debugging if needed
        # import traceback
        # print(traceback.format_exc())
        sys.exit(1)


    # --- Format and Display the Results ---
    print("\n" + "=" * 35)
    print("      GEOLOCATION RESULT")
    print("=" * 35)

    # Use .get() for dictionary access to provide defaults if keys are missing
    print(f"Source of Prediction: {results.get('source', 'Unavailable')}")

    # Check the 'success' flag returned by the engine
    if results.get('success'):
        confidence_score = results.get('confidence_score', 0.0)
        confidence_label = results.get('confidence_label', 'Low')
        latitude = results.get('latitude', 0.0)
        longitude = results.get('longitude', 0.0)
        address = results.get('address', 'N/A')
        map_url = results.get('map_url')

        print(f"Confidence:           {confidence_score:.2f} ({confidence_label})")
        print(f"Predicted Coordinates: Lat={latitude:.6f}, Lon={longitude:.6f}")
        print(f"Predicted Address:     {address}")

        if map_url:
            print(f"Map Link:             {map_url}")
        else:
            print("Map Link:             (Not generated or API Key missing for map generation)")
    else:
        # Handle cases where the engine ran but couldn't determine a location
        print(f"Confidence:           {results.get('confidence_score', 0.0):.2f} ({results.get('confidence_label', 'Low')})")
        print("Could not determine a specific location from the available sources.")

    print("=" * 35 + "\n")

if __name__ == "__main__":
    run_geolocation_cli()