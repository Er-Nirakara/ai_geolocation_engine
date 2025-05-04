# geolocate_v4.py (Acts as CLI front-end for the engine)

import argparse
import sys
import os
from dotenv import load_dotenv

# Attempt to import the engine function
try:
    from geolocation_engine import process_image_geolocation
except ImportError:
    print("Error: Could not import 'process_image_geolocation' from geolocation_engine.py")
    print("Ensure geolocation_engine.py is in the same directory.")
    sys.exit(1)

def main_cli():
    # Load .env primarily for the engine function to access keys if needed
    # (Depending on where you placed load_dotenv in the engine)
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Command-Line Interface for the AI Geolocation Engine."
    )
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--text",
        help="Optional text associated with the image (e.g., caption)",
        default=None
    )
    args = parser.parse_args()

    # Check if the image file exists before calling the engine
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at path: {args.image_path}")
        sys.exit(1)

    print(f"--- Starting Geolocation CLI for: {args.image_path} ---")

    # Call the engine function
    # The engine handles internal printing/logging
    results = process_image_geolocation(args.image_path, args.text)

    # --- CLI Output Formatting ---
    print("\n" + "=" * 30)
    print("      CLI FINAL RESULT")
    print("=" * 30)
    print(f"Source of Prediction: {results.get('source', 'N/A')}") # Use .get for safety

    if results.get('success'): # Check the success flag from the engine
        print(f"Confidence:           {results.get('confidence_score', 0.0):.2f} ({results.get('confidence_label', 'Low')})")
        print(f"Predicted Coordinates: Lat={results.get('latitude', 0.0):.6f}, Lon={results.get('longitude', 0.0):.6f}")
        print(f"Predicted Address:     {results.get('address', 'N/A')}")
        if results.get('map_url'):
            print(f"Map Link:             {results['map_url']}")
        else:
             print("Map Link:             (Not generated or API Key missing)")
    else:
        # Output even if processing partially succeeded but no location found
        print(f"Confidence:           {results.get('confidence_score', 0.0):.2f} ({results.get('confidence_label', 'Low')})")
        print("Could not determine location from available sources.")

    print("=" * 30 + "\n")

if __name__ == "__main__":
    main_cli()