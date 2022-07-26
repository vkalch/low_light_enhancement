import os

from app import app

# Create folders to store user uploaded and enhanced images
enhanced_image_folder = os.path.join('static', 'enhanced_data')
upload_folder = os.path.join('static', 'uploads')

# Make sure user upload and enhanced image folders exist
if not (os.path.exists(enhanced_image_folder)):
    os.makedirs(enhanced_image_folder)

if not (os.path.exists(upload_folder)):
    os.makedirs(upload_folder)

app.config['ENHANCED_FOLDER'] = enhanced_image_folder
app.config['UPLOAD_FOLDER'] = upload_folder

# Do not allow uploading files greater than 32 megabytes
app.config['MAX_CONTENT_LENGTH'] = 32 * 1000 * 1000

# Start the flask app
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
