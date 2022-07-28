"""
File: app.py

Author 1: A. Balsam
Author 2: J. Kuehne
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Product

Summary:
    This is the backend for a Flask application which incorporates our unsupervised learning methods and the template
    file we generated.
"""

import logging
import os
import threading
import uuid
from io import BytesIO
from zipfile import ZipFile

import flask
from flask import Flask, render_template, request, send_from_directory, send_file, url_for

from algs import get_algorithms, get_algorithm
from colormaps import get_colormaps, get_colormap_by_name
from low_signal_detector import LowSignalDetector as LSD

app = Flask(__name__)

# Set allowed extensions
ALLOWED_EXTENSIONS = {'tiff', 'tif'}

# Create a new instance of the LSD class to store user data
LOW_SIGNAL_DETECTORS = list()


def find_lsd_by_id(detector_id):
    for detector in LOW_SIGNAL_DETECTORS:
        if detector.get_id() == detector_id:
            return detector
    logging.error(f"Did not find an LSD with ID {detector_id}")
    return None


def allowed_file(filename: str):
    """
    Determines whether a file is allowed.

    :param filename: The filename to check
    :return: Whether the filename is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    """
    Downloads the selected filename.

    :param filename: Filename to download
    :return: Corresponding file
    """
    return send_from_directory(directory='/', path=filename)


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


@app.route('/<detector_id>')
def enhanced_images(detector_id):
    """
    Display the images contained within the user's ImageEnhancer.

    :return Either an error page or user's images
    """
    lsd = find_lsd_by_id(detector_id)

    if lsd is None:
        return render_template('error.html', msg=f"Could not find LSD. Contact us if this keeps happening",
                               btn_href='/enhance', btn_text='Go back', redirect_location='/enhance')

    print(lsd.ENHANCED_IMAGES)
    if lsd.ENHANCED_IMAGES == list():
        if lsd.ERR:
            return render_template('error.html', msg=f"Errored out with error: {lsd.ERR}", btn_href='/enhance',
                                   btn_text='Go back', redirect_location='/enhance')
        else:
            return render_template('error.html', msg="Your images have not yet finished processing. They should load "
                                                     "automatically. If they do not, click on the button below. The "
                                                     "enhancement process may take a couple minutes, depending on "
                                                     "server traffic and the algorithms you chose. Please be patient.",
                                   btn_href=url_for('enhanced_images', detector_id=detector_id),
                                   btn_text="Attempt to Load Images")
    else:
        return render_template("algorithm.html", original_images=lsd.ORIGINAL_IMAGES,
                               enhanced_images=lsd.ENHANCED_IMAGES,
                               download_paths_by_image=lsd.DOWNLOAD_PATHS_BY_IMAGE,
                               user_id=lsd.ID)


@app.route('/download_all/<i>/<detector_id>')
def download_all_by_num(i: int, detector_id: str):
    """
    Downloads all images enhanced from specified source image.

    :param i: Number of original image
    :param detector_id: The unique ID of the current user's LowSignalDetector
    :return: Enhanced images corresponding to original image
    """
    lsd = find_lsd_by_id(detector_id)
    if lsd is None:
        return render_template('error.html', msg=f"Could not find LSD. Contact us if this keeps happening.",
                               btn_href='/enhance', btn_text='Go back', redirect_location='/enhance')
    paths = lsd.DOWNLOAD_PATHS_BY_IMAGE[int(i)]
    stream = BytesIO()
    with ZipFile(stream, 'w') as zf:
        for file in paths:
            zf.write(file, os.path.basename(file))
    stream.seek(0)

    return send_file(
        stream,
        as_attachment=True,
        attachment_filename=f'enhanced_images_{i}.zip'
    )


@app.route('/download_all/<detector_id>')
def download_all(detector_id):
    """
    Downloads all images from self.DOWNLOAD_PATHS_BY_IMAGE as .zip file
    (every file this user's LSD class has generated). Does not pick a specific image to save.

    :param detector_id: The unique ID of the current user's LSD
    :return: Zip file containing all images on page
    """
    lsd = find_lsd_by_id(detector_id)
    if lsd is None:
        render_template('error.html', msg=f"Could not find LSD. Contact us if this keeps happening.",
                        btn_href='/enhance', btn_text='Go back', redirect_location='/enhance')

    paths = list()
    for path_list in lsd.DOWNLOAD_PATHS_BY_IMAGE:
        for path in path_list:
            paths.append(path)
    stream = BytesIO()
    with ZipFile(stream, 'w') as zf:
        for file in paths:
            zf.write(file, os.path.basename(file))
    stream.seek(0)

    return send_file(
        stream,
        as_attachment=True,
        attachment_filename='enhanced_images.zip'
    )


@app.route('/algorithm', methods=['POST'])
def algorithm():
    """
    Creates new LowSignalDetector and a thread to run its run_algorithm() method. Returns a screen which will take the
    user to a screen containing the images he has enhanced.

    :return: The current user's submitted page
    """
    radius_denoising = int(request.form['noiseRadiusInput'])
    radius_circle = int(request.form['circleRadiusInput'])
    autoencoder_image_size = int(request.form['autoencoderImageSizeInput'])
    autoencoder_num_epochs = int(request.form['autoencoderNumEpochsInput'])
    autoencoder_dense_layer_neurons = int(request.form['autoencoderDenseLayerInput'])

    algs = [(abbr, get_algorithm(abbr)) for abbr in request.form.getlist('algorithmCheckbox')]
    if algs == list():
        return render_template('error.html', msg=f"Please select one or more algorithms.", btn_href='/enhance',
                               btn_text='Go back', redirect_location='/enhance')
    colormap = request.form.get('colormapInput')

    colormap = get_colormap_by_name(colormap)

    do_enhance = bool(request.form.get('enhanceCheckbox'))
    show_circle = bool(request.form.get('showCircleCheckbox'))

    images = flask.request.files.getlist('fileUploadInput')
    images_by_filename = list()
    for image in images:
        if image.filename == '':
            return render_template('error.html', msg=f"Please upload one or more files.",
                                   btn_href='/enhance', btn_text='Go back', redirect_location='/enhance')
        if image and allowed_file(image.filename):
            filename = f"{str(uuid.uuid4())}.tiff"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(path)
            images_by_filename.append(path)
        else:
            return render_template('error.html', msg=f"Please upload an image with .tif or .tiff extension.",
                                   btn_href='/enhance', btn_text='Go back', redirect_location='/enhance')

    output_folder = os.path.join(app.config['ENHANCED_FOLDER'])
    lsd = LSD(images_by_filename=images_by_filename, algs=algs, radius_denoising=radius_denoising,
              radius_circle=radius_circle, colormap=colormap, do_enhance=do_enhance,
              show_circle=show_circle, autoencoder_image_size=autoencoder_image_size,
              autoencoder_num_epochs=autoencoder_num_epochs,
              autoencoder_dense_layer_neurons=autoencoder_dense_layer_neurons, output_folder=output_folder)
    LOW_SIGNAL_DETECTORS.append(lsd)

    t = threading.Thread(target=lsd.run_algorithms)
    t.start()

    return render_template('submitted.html', user_id=lsd.ID)
