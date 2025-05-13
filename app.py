# Flask web application for the Image Geolocation Engine

import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Import the core processing logic
try:
    from geolocation_engine import process_image_geolocation
except ImportError:
    print("FATAL ERROR: Could not import 'process_image_geolocation' from 'geolocation_engine.py'.")
    print("Ensure 'geolocation_engine.py' is in the same directory and has no critical errors.")
    raise # Stop Flask from starting if the engine is missing

# --- Application Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'heic', 'heif', 'avif', 'webp', 'tiff', 'bmp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB upload limit
# IMPORTANT: Change secret_key to a strong, random value for any real deployment!
app.secret_key = 'dev-secret-key-change-me'


def is_allowed_file(filename):
    """Checks if the uploaded file has an allowed image extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serves the main page with the image upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def handle_image_upload():
    """
    Handles image upload, calls the geolocation engine, and displays results.
    Accepts POST requests only.
    """
    # Ensure an image file was part of the request
    if 'image' not in request.files:
        flash('No image file selected.')
        return redirect(request.url) # Redirect back to the current page (upload form)

    uploaded_file = request.files['image']
    associated_text = request.form.get('text') # Optional text, defaults to None

    # Check if a file was actually selected by the user
    if uploaded_file.filename == '':
        flash('No image file selected.')
        return redirect(url_for('index'))

    # Process the file if it's present and has an allowed extension
    if uploaded_file and is_allowed_file(uploaded_file.filename):
        # Create a secure, unique filename for the temporary upload
        original_extension = os.path.splitext(uploaded_file.filename)[1].lower()
        unique_filename = secure_filename(f"{uuid.uuid4()}{original_extension}")
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            uploaded_file.save(temp_filepath)
            print(f"INFO: Image temporarily saved: {temp_filepath}")

            # Process the image using the geolocation engine
            results_data = process_image_geolocation(temp_filepath, associated_text)

            # Display the results
            return render_template('results.html', results=results_data)

        except Exception as e:
            # Handle unexpected errors during processing
            print(f"ERROR: Processing upload for {unique_filename} failed: {e}")
            # Consider logging the full traceback for server-side debugging
            # import traceback
            # print(traceback.format_exc())
            flash(f'An error occurred while processing your image: {str(e)[:100]}...')
            return render_template('results.html', error=str(e))
        finally:
            # Ensure the temporary file is deleted after processing
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                    print(f"INFO: Cleaned up temporary file: {temp_filepath}")
                except OSError as e_remove:
                    print(f"ERROR: Failed to remove temporary file {temp_filepath}: {e_remove}")
    else:
        # Handle invalid file types
        allowed_types_str = ', '.join(sorted(list(ALLOWED_EXTENSIONS))) # Sort for consistent display
        flash(f'Invalid file type. Allowed types: {allowed_types_str}')
        return redirect(url_for('index'))


if __name__ == '__main__':
    # Create the 'uploads' directory if it doesn't exist when the app starts
    if not os.path.exists(UPLOAD_FOLDER):
        try:
            os.makedirs(UPLOAD_FOLDER)
            print(f"INFO: Created uploads directory: {UPLOAD_FOLDER}")
        except OSError as e_mkdir:
            print(f"FATAL ERROR: Could not create uploads directory '{UPLOAD_FOLDER}': {e_mkdir}")
            raise # Stop app if uploads folder cannot be created

    # Run the Flask development server
    # debug=True is for development only (enables debugger and auto-reloader)
    app.run(debug=True)