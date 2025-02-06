import os
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from ultralytics.models.sam import Predictor as SAMPredictor
import cv2
import tensorflow as tf
from tensorflow import keras
import gc
from os import path
from PIL import Image

gc.disable()

tf.experimental.numpy.experimental_enable_numpy_behavior()

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  # Directory where uploaded files will be saved
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = r"\Codes\ACPS Project\Images"
app.config['STATIC_FOLDER'] = r"\Codes\ACPS Project\static"
class_names = ['Jassid and Mite', 'Potassium Deficiency', 'Leaf Miner', 'Mites', 'Nitrogen Deficiency', 'Nitrogen and Potassium Deficiency', 'healthy']


batch_size = 32
img_height = 180
img_width = 180

# Load yolov8 model
model = YOLO(r"C:\Codes\ACPS Project\best.pt")

model_stress = tf.keras.models.load_model('my_model.h5')
icon = 'plantVector.jpg'

@app.route("/")
@app.route("/home")
def hello_world():
    return render_template('home.html', icon_url=icon,num_segmented_images = 0)

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    file_address = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    im = mpimg.imread(file_address)
    results = model(file_address, conf=0.7)
    for result in results:
    # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))

        boxes = result.boxes
        ax.imshow(im)

        for box in boxes:
        # Display the image

            # bbox = np.array([[71.8246, 111.9604, 757.2079, 1270.9786]])
            x_min, y_min, x_max, y_max = box.xyxy[0]

            # Create a Rectangle patch
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=3, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
            
    ax.axis('off')
    image_filename = f'{file.filename}'
    print(image_filename)
    image_path = os.path.join(app.config['STATIC_FOLDER'], image_filename)
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
            
            
    predictions_all = []
    
    segmented_images = []
    overrides = dict(conf=0.7, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
    predictor = SAMPredictor(overrides=overrides)
    for result in results:
        # Create figure and axes
        

        boxes = result.boxes
        input_boxes = result.boxes.xyxy
        predictor.set_image(im)
        segment = predictor(bboxes= input_boxes)
        ax.imshow(im)
        mask_image = segment[0].masks.xy
        length_mask = len(mask_image)
    
        for i in range(0,length_mask):
        
            # Extract the masked leaf
            mask_coordinates = np.array(mask_image[i], dtype=np.int32)
            
            black_image = np.zeros_like(im)
            
            cv2.fillPoly(black_image, [mask_coordinates], (255, 255, 255))
            
            black_image_gray = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
            
            result_image = cv2.bitwise_and(im, im, mask=black_image_gray)
            
            segment_address = os.path.join(app.config['STATIC_FOLDER'], f"segment_{i}_{file.filename}")

            # file.save(os.path.join(app.config['STATIC_FOLDER'], f"segment_{i}_{file.filename}"))
            cv2.imwrite(segment_address,result_image)
            
            segmented_images.append(segment_address)
            # Resize the segmented image
            resized_image = cv2.resize(result_image, (180, 180))

            # Preprocess the image
            img_array = tf.keras.utils.img_to_array(resized_image)
            img_array = tf.expand_dims(img_array, 0)

            # Classify the segmented image
            predictions = model_stress.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predictions_all.append(score)
            
    dir_path = r"C:\Users\praph\runs\segment"

    # Get a list of all directories in the specified path
    entries = os.listdir(dir_path)
    directories = [entry for entry in entries if path.isdir(path.join(dir_path, entry))]

# Get the last created directory
    if directories:
        last_created_dir = max(directories, key=lambda d: os.path.getmtime(path.join(dir_path, d)))
        last_created_dir_path = path.join(dir_path, last_created_dir)
        print(f"The address of the last created directory is: {last_created_dir_path}")
    else:
        print("No directories found in the specified path.")
       
        
    segmented_image_path = path.join(last_created_dir_path, "image0.jpg")
    
    image = Image.open(segmented_image_path)
    
    image.save(os.path.join(app.config['STATIC_FOLDER'], "segment_full_Image.jpg"))
    # Take the average of all predictions
    avg_prediction = np.mean(predictions_all, axis=0)

    # Get the class with the highest prediction
    max_index = np.argmax(avg_prediction)
    max_score = avg_prediction[max_index]
    
    num_segmented_images = len(segmented_images)

    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[max_index], 100 * max_score))

    return render_template('home.html', image_filename=image_filename, score=avg_prediction, class_names=class_names, icon_url=icon,num_segmented_images = num_segmented_images,segmented_image_path = "segment_full_Image.jpg")

@app.route("/History")
def history():
    items = [
        {'id': '1', 'name': 'Tomato leaf 1', 'Stress': 'Potassium', 'Severe': 'High'},
        {'id': '2', 'name': 'Tomato leaf 2', 'Stress': 'Nitrogen', 'Severe': 'Medium'},
        {'id': '3', 'name': 'Tomato leaf 3', 'Stress': 'Sulphur', 'Severe': 'Low'}
    ]
    return render_template('history.html', items=items)

if __name__ == '__main__':
    app.run(debug=True)