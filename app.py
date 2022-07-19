import os
import threading
import time
from io import BytesIO
from zipfile import ZipFile

import cv2
import flask
from flask import Flask, render_template, request, send_from_directory, send_file
from werkzeug.utils import secure_filename, redirect

import low_signal_detection as lsd
from algs import get_algorithms, get_algorithm, get_algorithm_name
from colormaps import get_colormaps, get_colormap_by_name

import globals

enhanced_image_folder = os.path.join('static', 'enhanced_data')
upload_folder = os.path.join('static', 'uploads')

app = Flask(__name__)
app.config['ENHANCED_FOLDER'] = enhanced_image_folder
app.config['UPLOAD_FOLDER'] = upload_folder

ALLOWED_EXTENSIONS = {'tiff'}

if not (os.path.exists(enhanced_image_folder)):
    os.makedirs(enhanced_image_folder)

if not (os.path.exists(upload_folder)):
    os.makedirs(upload_folder)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_from_directory(directory='/', path=filename)


@app.route('/download_all/<i>')
def download_all(i):
    print(i)
    paths = globals.DOWNLOAD_PATHS_BY_IMAGE[int(i)]
    stream = BytesIO()
    with ZipFile(stream, 'w') as zf:
        for file in paths:
            zf.write(file, os.path.basename(file))
    stream.seek(0)

    return send_file(
        stream,
        as_attachment=True,
        attachment_filename='enhanced_image.zip'
    )


@app.route('/')
def index_page():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/enhance')
def enhance_page():
    return render_template("enhance.html", algorithms=get_algorithms(), colormaps=get_colormaps())


def run_algorithm(radius_denoising, radius_circle, algs, colormap, do_enhance, images_by_filename):
    globals.ORIGINAL_IMAGES = list()
    globals.ENHANCED_IMAGES = list()
    globals.DOWNLOAD_PATHS_BY_IMAGE = list()
    """# list of sequences of images to be analysed
    images = [
        './static/test_data/sample0.tiff'
    ]"""

    start_complete = time.time()

    for img_num in range(len(images_by_filename)):
        globals.ORIGINAL_IMAGES.append(images_by_filename[img_num])
        enhanced_image_by_alg = list()
        for abbr, alg in algs:
            enhanced_image = lsd.detect_signal(images_by_filename[img_num], alg, do_enhance,
                                               radius_denoising, radius_circle)

            path = os.path.join(app.config['ENHANCED_FOLDER'], f'image{img_num}_{abbr}.png')
            enhanced_image = cv2.applyColorMap(enhanced_image, colormap=colormap)
            cv2.imwrite(path, enhanced_image)
            enhanced_image_by_alg.append({"alg_name": f"{get_algorithm_name(abbr)}", "filename": path})

        globals.ENHANCED_IMAGES.append({"img_num": img_num, "enhanced_by_alg": enhanced_image_by_alg})

    # take time for computation duration
    end_complete = time.time()
    # output infos
    print('Finished in ' + str(round(end_complete - start_complete, 2)) + 's')

    time.sleep(0.1)

    paths = [[img['filename'] for img in enhanced_image['enhanced_by_alg']] for enhanced_image in globals.ENHANCED_IMAGES]

    for i in range(len(images_by_filename)):
        globals.DOWNLOAD_PATHS_BY_IMAGE.append(paths[i])

    print("Finished running algorithms...")


@app.route('/algorithm', methods=['POST'])
def algorithm():
    radius_denoising = int(request.form['noiseRadiusInput'])
    radius_circle = int(request.form['circleRadiusInput'])
    algs = [(abbr, get_algorithm(abbr)) for abbr in request.form.getlist('algorithmCheckbox')]
    if algs == list():
        return redirect('/')
    colormap = request.form.get('colormapInput')

    if colormap is None:
        colormap = cv2.COLORMAP_BONE
    else:
        colormap = get_colormap_by_name(colormap)

    do_enhance = True

    images = flask.request.files.getlist('fileUploadInput')
    images_by_filename = list()
    for image in images:
        if image.filename == '':
            return redirect('/')
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            while True:
                try:
                    image.save(path)
                    images_by_filename.append(path)
                    break
                except IsADirectoryError:
                    path += "(1)"

    args = (radius_denoising, radius_circle, algs, colormap, do_enhance, images_by_filename)
    t = threading.Thread(target=run_algorithm, args=args)
    t.start()

    return """<p>Loading...</p><script>
    function redirect() {
        location.replace("/enhanced_images")
    }
    setTimeout(function(){ redirect(); }, 5000);</script>"""


@app.route('/enhanced_images')
def enhanced_images():
    return render_template("algorithm.html", original_images=globals.ORIGINAL_IMAGES, enhanced_images=globals.ENHANCED_IMAGES,
                           download_paths_by_image=globals.DOWNLOAD_PATHS_BY_IMAGE)


port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
