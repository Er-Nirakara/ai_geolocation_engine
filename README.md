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
*   **NLP Analysis (spaCy):** Processes user-provided text captions to identify location-related entities using the `en_core_web_lg` model.
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
    *   **Command-Line Interface (CLI):** For script-based processing (`image_geolocate.py`).
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
    The engine attempts to load the `en_core_web_lg` spaCy model for Natural Language Processing. Ensure this model is downloaded:
    ```bash
    python -m spacy download en_core_web_lg
    ```
    If this model fails to load, NLP-based geolocation features will be disabled (the engine will log an error message).

5.  **Install Optional Pillow Image Format Plugins:**
    For HEIC/HEIF, AVIF support (as mentioned in `geolocation_engine.py` logging):
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
        Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the **absolute path** of your downloaded service account JSON key file. The `geolocation_engine.py` is set up to read this path (it uses `os.getenv("GOOGLE_APPLICATION_CREDENTIALS")` and then `vision.ImageAnnotatorClient.from_service_account_file()`).
        *Set this in your terminal session before running the app:*
            *   PowerShell: `$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-file.json"`
            *   CMD: `set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-file.json"`
            *   Bash/Linux/macOS: `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"`

8.  **Important: `.gitignore`:**
    Your `.gitignore` should already be configured to ignore `.env`, `venv/`, `uploads/`, and `*.json` (or your specific key file name like `gcloud_key.json`).

## 5. How to Run

### a) Web Interface (Flask)

1.  Ensure setup is complete and the virtual environment is active.
2.  Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable in your terminal (see Step 7 above).
3.  Run:
    ```bash
    python app.py
    ```
4.  Open your browser to: `http://127.0.0.1:5000`

### b) Command-Line Interface (CLI)

1.  Ensure setup is complete and the virtual environment is active.
2.  Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable in your terminal (see Step 7 above).
3.  Run:
    ```bash
    python image_geolocate.py "path/to/your/image.jpg" --text "Optional caption for NLP analysis"
    ```
    Example:
    ```bash
    python image_geolocate.py "./my_images/eiffel_tower.jpg" --text "A photo I took of the Eiffel Tower last summer."
    ```

## 6. Output Description

*   **Source of Prediction:** Primary data source(s) used (e.g., EXIF, CV Landmark, NLP, OCR), potentially with specific names/text if derived from those.
*   **Confidence Score:** Numerical score (0.0-1.0) and label (Low, Medium, High).
*   **Predicted Coordinates:** Latitude and Longitude.
*   **Predicted Address:** Human-readable address/place name.
*   **Map Link/Image:** Link (CLI) or static map image (Web UI) of the location.

## 7. Known Limitations & Future Work

*   **API Costs & Quotas:** Monitor Google Cloud API usage to stay within free tiers or manage costs.
*   **NLP Model Robustness:** The `geolocation_engine.py` uses the `en_core_web_lg` spaCy model. While good, ensure it's properly installed. If it fails to load, NLP features are disabled, which can reduce accuracy for text-based clues.
*   **Overpass API Robustness:** The Overpass API (for OpenStreetMap data) can sometimes be slow or temporarily unavailable. The engine has basic error handling and a timeout (`OVERPASS_TIMEOUT = 30` seconds).
*   **Complex Scenarios:** Highly ambiguous images or those with conflicting clues can still be challenging for the current fusion logic.

**Future Work Ideas:**

*   Implement an ML-based evidence fusion model.
*   Improve OCR post-processing (e.g., more advanced NLP to identify and geocode only relevant street names/POIs from OCR text).
*   More sophisticated error handling and retries for API calls (currently `API_RETRY_DELAY = 0.05`s is a brief pause, not a full retry mechanism).
*   Caching API results to reduce costs and speed up repeated queries for identical inputs.

## 8. Troubleshooting

*   **`ImportError` for `geolocation_engine`**:
    *   Both `app.py` and `image_geolocate.py` will print specific error messages if they cannot import `process_image_geolocation` from `geolocation_engine.py`.
    *   Ensure all files (`app.py`, `image_geolocate.py`, `geolocation_engine.py`) are in the correct root directory.
    *   Check for syntax errors in `geolocation_engine.py` by trying to run it directly (e.g., `python geolocation_engine.py` - it won't do much but will show syntax errors).
*   **`GOOGLE_APPLICATION_CREDENTIALS` invalid/missing**:
    *   This is critical for the Cloud Vision API. The `geolocation_engine.py` includes "AGGRESSIVE DEBUGGING FOR CREDENTIALS" logging, which will output (at DEBUG level) the path it's trying to use.
    *   Verify the environment variable is correctly set to the *full, absolute path* of your service account JSON key file *in the same terminal session* you are running the app from.
    *   The engine explicitly uses `vision.ImageAnnotatorClient.from_service_account_file(gcp_cred_path_env)`. If `gcp_cred_path_env` is invalid or the file is unreadable, Vision API calls will fail.
*   **API Errors (403, Authorization, Quota):**
    *   Verify all required APIs (Vision, Geocoding, Maps Static, Places) are **enabled** in Google Cloud Console.
    *   Check your API key validity and any restrictions (e.g., HTTP referrers, API restrictions).
    *   Ensure **billing is enabled** on your GCP project.
*   **Image Format Errors (e.g., for HEIC/AVIF)**:
    *   Install optional Pillow plugins: `pip install pillow-heif pillow-avif-plugin`. The `geolocation_engine.py` logs a warning if Pillow cannot identify an image format.
*   **spaCy Model Load Error**:
    *   If you see "Failed to load 'en_core_web_lg' spaCy model..." in the logs from `geolocation_engine.py`, ensure you have run `python -m spacy download en_core_web_lg`. NLP features will be disabled otherwise.

---