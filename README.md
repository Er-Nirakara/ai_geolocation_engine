# AI-Powered Image Geolocation Engine


## 1. Introduction

This project is an AI-Powered Image Geolocation Engine designed to determine the geographical location (latitude, longitude, and/or place name) where an image was taken. It leverages a multi-modal approach by integrating information from:

*   Image EXIF metadata (GPS tags).
*   Associated textual descriptions (via Natural Language Processing - NLP with spaCy).
*   Direct visual analysis of the image content (via Google Cloud Vision API for landmark recognition and Optical Character Recognition - OCR).
*   Contextual verification using Google Places API (Nearby Search).
*   Map data validation using OpenStreetMap (via Overpass API).

The system employs an advanced evidence fusion logic to combine these diverse data sources, calculate a confidence score for the prediction, and provide a link to a map of the estimated location. This project demonstrates the application of various AI techniques to solve a complex real-world problem.

## 2. Features

*   **Multi-Modal Geolocation:** Combines EXIF, NLP, CV, Places API, and OSM data.
*   **EXIF Data Extraction:** Reads GPS latitude, longitude from image metadata.
*   **NLP Analysis (spaCy):** Processes user-provided text captions to identify location-related entities.
*   **Computer Vision Analysis (Google Cloud Vision API):**
    *   **Landmark Recognition:** Identifies known geographical landmarks.
    *   **Optical Character Recognition (OCR):** Detects and extracts text from images.
*   **Geocoding & Reverse Geocoding (Google Maps API):** Converts location names to coordinates and vice-versa.
*   **Contextual Verification (Google Places API):** Uses Nearby Search to verify landmarks or text clues against known places.
*   **Map Data Validation (OpenStreetMap via Overpass API):** Verifies features like street names or place names against OSM data.
*   **Advanced Evidence Fusion:**
    *   Collects all potential location candidates.
    *   Applies verification bonuses from Places API and OSM.
    *   Calculates a weighted score for each candidate (source priority, API confidence, verification, proximity).
    *   Selects the location with the highest overall score.
*   **Confidence Scoring:** Outputs an estimated confidence level (Low, Medium, High, and a 0-1 score).
*   **Map Visualization:** Generates a Google Static Maps API link for the predicted location.
*   **Broad Image Format Support:** Handles common image formats (JPEG, PNG, etc.) and attempts to support others like HEIC/AVIF via Pillow plugins (if installed).
*   **Dual Interface:**
    *   **Command-Line Interface (CLI):** For script-based processing (`geolocate_v4.py`).
    *   **Web Interface (Flask):** For easy image upload and visual results display (`app.py`).

## 3. System Architecture (High-Level)

1.  **Input:** Image file + optional text (via CLI or Web UI).
2.  **Data Extraction (`geolocation_engine.py`):**
    *   EXIF, NLP (spaCy), Vision API (Landmarks, OCR).
3.  **Geocoding (`geolocation_engine.py`):**
    *   Google Maps Geocoding API for text clues.
4.  **Verification Modules (`geolocation_engine.py`):**
    *   Google Places API (Nearby Search).
    *   OpenStreetMap (Overpass API).
5.  **Evidence Fusion Logic (`geolocation_engine.py`):**
    *   Candidate collection, scoring, verification bonuses, proximity checks.
6.  **Output:**
    *   Predicted location, source, confidence, map link (via CLI or Web UI).

## 4. Setup and Installation

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   Git (for cloning)
*   A Google Cloud Platform (GCP) Account with billing enabled.

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Er-Nirakara/ai_geolocation_engine.git
    cd ai_geolocation_engine
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy Language Model:**
    The engine currently uses `en_core_web_sm`. For better NLP accuracy (especially for recognizing specific place names like "Colosseum"), consider a larger model:
    ```bash
    python -m spacy download en_core_web_md
    # OR
    # python -m spacy download en_core_web_lg
    ```
    Then, update the model name in `geolocation_engine.py` (e.g., `NLP = spacy.load("en_core_web_md")`).

5.  **Install Optional Pillow Image Format Plugins:**
    For HEIC/HEIF, AVIF support:
    ```bash
    pip install pillow-heif
    pip install pillow-avif-plugin
    ```

6.  **Set up Google Cloud APIs:**
    *   Go to the [Google Cloud Console](https://console.cloud.google.com/).
    *   Select/Create a project and ensure **Billing** is enabled.
    *   Enable the following APIs in "APIs & Services" -> "Library":
        *   **Cloud Vision API**
        *   **Geocoding API**
        *   **Maps Static API**
        *   **Places API**
    *   Create an **API Key** (for Maps, Geocoding, Places, Static Maps).
    *   Create a **Service Account Key (JSON)** (for Vision API) and grant it "Cloud Vision AI User" role (or similar). Download the JSON file.

7.  **Configure Environment Variables:**
    *   **Create `.env` file:** In the project root, create `.env` and add your Google Maps API Key:
        ```plaintext
        # .env
        GOOGLE_MAPS_API_KEY='YOUR_ACTUAL_GOOGLE_MAPS_PLATFORM_API_KEY'
        ```
    *   **Google Application Credentials (for Vision API):**
        Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the **absolute path** of your downloaded service account JSON key file. The `geolocation_engine.py` is set up to read this path.
        *Set this in your terminal session before running the app:*
            *   PowerShell: `$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-file.json"`
            *   CMD: `set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-file.json"`
            *   Bash/Linux/macOS: `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"`

8.  **Important: `.gitignore`:**
    Your `.gitignore` should already be configured to ignore `.env`, `venv/`, `uploads/`, and `*.json` (or your specific key file name).

## 5. How to Run

### a) Web Interface (Flask)

1.  Ensure setup is complete and the virtual environment is active.
2.  Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable in your terminal.
3.  Run:
    ```bash
    python app.py
    ```
4.  Open your browser to: `http://127.0.0.1:5000`

### b) Command-Line Interface (CLI)

1.  Ensure setup is complete and the virtual environment is active.
2.  Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable in your terminal.
3.  Run:
    ```bash
    python geolocate_v4.py "path/to/your/image.jpg" --text "Optional caption"
    ```

## 6. Output Description

*   **Source of Prediction:** Primary data source(s) used (e.g., EXIF, CV Landmark, NLP, OCR).
*   **Confidence Score:** Numerical score (0.0-1.0) and label (Low, Medium, High).
*   **Predicted Coordinates:** Latitude and Longitude.
*   **Predicted Address:** Human-readable address/place name.
*   **Map Link/Image:** Link (CLI) or static map image (Web UI) of the location.

## 7. Known Limitations & Future Work

*   **API Costs & Quotas:** Monitor Google Cloud API usage to stay within free tiers or manage costs.
*   **NLP Model:** The default `en_core_web_sm` spaCy model might not recognize all specific named entities as locations. Using a larger model (`md` or `lg`) is recommended.
*   **Overpass API Robustness:** Overpass API can sometimes be slow or temporarily unavailable. Error handling is basic.
*   **Complex Scenarios:** Highly ambiguous images or those with conflicting clues can still be challenging.

**Future Work Ideas:**

*   Implement an ML-based evidence fusion model.
*   Improve OCR post-processing (e.g., identify and geocode only street names/POIs).
*   More sophisticated error handling and retries for API calls.
*   Caching API results to reduce costs and speed up repeated queries.

## 8. Troubleshooting

*   **`ImportError`**: Ensure all files are in the correct directory and check for syntax errors in the imported file by running it directly (e.g., `python geolocation_engine.py`).
*   **`GOOGLE_APPLICATION_CREDENTIALS invalid/missing`**: This is critical. Verify the environment variable is correctly set to the *full, absolute path* of your service account JSON key file *in the same terminal session* you are running the app from. The `geolocation_engine.py` uses `ImageAnnotatorClient.from_service_account_file()` which depends on this variable being correctly passed or `os.getenv()` retrieving it successfully.
*   **API Errors (403, Authorization, Quota):**
    *   Verify all required APIs are **enabled** in Google Cloud Console.
    *   Check API key validity and restrictions.
    *   Ensure **billing is enabled** on your GCP project.
*   **Image Format Errors**: Install `pillow-heif` and `pillow-avif-plugin` for HEIC/AVIF support.

---
This project was developed with the assistance of AI.
