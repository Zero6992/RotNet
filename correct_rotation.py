from __future__ import print_function

import os
import cv2
import numpy as np
import argparse
import tensorflow as tf

# Adjusted imports for tf.keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model

# Utils import, ensure rotate and crop_largest_rectangle are defined here
from utils import crop_largest_rectangle, angle_error, rotate

def process_images(model, input_path, output_path, crop=False):
    extensions = ['.jpg', '.jpeg', '.bmp', '.png']
    output_is_image = False
    if os.path.isfile(input_path):
        image_paths = [input_path]
        if os.path.splitext(output_path)[1].lower() in extensions:
            output_is_image = True
            output_filename = output_path
            output_path = os.path.dirname(output_filename)
    else:
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in extensions]
        if os.path.splitext(output_path)[1].lower() in extensions:
            print('Output must be a directory!')

    if output_path == '':
        output_path = '.'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224))
        image_processed = preprocess_input(np.expand_dims(image, axis=0))

        prediction = model.predict(image_processed)
        predicted_angle = np.argmax(prediction, axis=1)[0]
        print(f"Predicted angle for {os.path.basename(path)} is: {predicted_angle} degrees")
        rotated_image = rotate(image, -predicted_angle)
        if crop:
            size = (image.shape[0], image.shape[1])
            rotated_image = crop_largest_rectangle(rotated_image, -predicted_angle, *size)
        if not output_is_image:
            output_filename = os.path.join(output_path, os.path.basename(path))
        cv2.imwrite(output_filename, rotated_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to model')
    parser.add_argument('input_path', help='Path to image or directory')
    parser.add_argument('-o', '--output_path', help='Output directory', default='')
    parser.add_argument('-c', '--crop', dest='crop', action='store_true', default=False, help='Crop out black borders after rotating')
    args = parser.parse_args()

    print('Loading model...')
    model_location = load_model(args.model, custom_objects={'angle_error': angle_error})

    print('Processing input image(s)...')
    process_images(model_location, args.input_path, args.output_path, args.crop)
