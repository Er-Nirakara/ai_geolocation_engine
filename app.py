import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import uuid # To generate unique filenames

# Import the engine function
from geolocation_engine import process_image_geolocation

# Configure Flask App
UPLOAD_FOLDER = 'uploads' # Make sure this folder exists
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'heic', 'heif', 'avif', 'webp', 'tiff'} # Allowed image types

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB upload limit
app.secret_key = 'your secret key here' # Change this for production!

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Renders the main upload form page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, calls the engine, and renders results."""
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    associated_text = request.form.get('text', None) # Get optional text

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index')) # Redirect back to index if no file

    if file and allowed_file(file.filename):
        # Create a unique filename to prevent collisions/overwrites
        _, extension = os.path.splitext(file.filename)
        filename = secure_filename(f"{uuid.uuid4()}{extension.lower()}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            print(f"File saved temporarily to: {filepath}")

            # --- Call the Geolocation Engine ---
            # Pass the path to the saved file and the text
            results_dict = process_image_geolocation(filepath, associated_text)

            # --- Render Results ---
            # Pass the results dictionary directly to the template
            return render_template('results.html', results=results_dict)

        except Exception as e:
            # Catch potential errors during processing or engine call
            print(f"Error during processing: {e}")
            flash(f'An error occurred during processing: {e}')
            # Optionally render results page with an error message
            return render_template('results.html', error=str(e))
        finally:
            # --- Clean up the uploaded file ---
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Cleaned up temporary file: {filepath}")
                except OSError as e:
                    print(f"Error removing temporary file {filepath}: {e}")

    else:
        flash('Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS))
        return redirect(url_for('index')) # Redirect back to index

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # Run the app (debug=True is helpful during development)
    app.run(debug=True)