from flask import Flask, render_template, request, redirect
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import tensorflow_hub as hub
import os



app = Flask(__name__)

input_image_path = "C:/Users/piyush.sharma/Documents/dog-breed-classification/input"
output_image_path = "C:/Users/piyush.sharma/Documents/dog-breed-classification/static"
output_image_name = "image_with_prediction.jpg"
resized_image_path = "C:/Users/piyush.sharma/Documents/dog-breed-classification/resized image/resized_image.jpg"

# initialize model
model = tf.keras.models.load_model(("models/20221025-094643-full-image-set-mobilenetv2-Adam.h5"), custom_objects={"KerasLayer": hub.KerasLayer})

# load labels
labels_csv = pd.read_csv("C:/Users/piyush.sharma/Documents/dog-breed-classification/labels/labels.csv")
labels = labels_csv["breed"].to_numpy()

# create unique breeds array
unique_breeds = np.unique(labels)


def preprocess_image(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image , size=[224, 224])
    return image

def get_pred_labels(prediction_probabilities):
    return unique_breeds[np.argmax(prediction_probabilities)]

def create_data_batches(X, y=None, batch_size=32, valid_data=False,test_data =False):
  """
    Creates batches of data out of image (X) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle if it's validation data.
    Also accepts test data as input (no labels).
  """
  # IF the data is a test dataset, we probably don't have labels
  if test_data:
    print("Creating testa data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # only filepaths (no labels)
    data_batch = data.map(preprocess_image).batch(batch_size)
    return data_batch

def predict(image):
    image_list = []
    image_list.append(image)
    data = create_data_batches(image_list, test_data=True)
    preds = model.predict(data)
    pred_labels = [get_pred_labels(preds[i]) for i in range(len(preds))]
    return pred_labels, data

def save_prediction_image(image_path, data, pred_labels):
    images =[]
    for image in data.unbatch().as_numpy_iterator():
        images.append(image)
    image = cv2.imread(image_path)

    new_height = 300
    new_width = 200

    resized_image = cv2.resize(image, (new_width, new_height))
    cv2.imwrite(resized_image_path, resized_image)

    image = cv2.imread(resized_image_path)
    # Define the font and other text properties
    font = cv2.FONT_ITALIC
    font_scale = 0.5
    font_color = (0, 255, 0)  # White color in BGR
    font_thickness = 2
    shadow_color = (0, 0, 0)  # Black shadow color in BGR
    # Calculate the position for the text
    text_x = 10  # X-coordinate for the text (adjust as needed)
    text_y = 20  # Y-coordinate for the text (adjust as needed)
    # Draw the text shadow first
    cv2.putText(image, pred_labels[0].upper(), (text_x+3 , text_y+3),
                font, font_scale, shadow_color, font_thickness, cv2.LINE_AA)
    # Draw the main text on top of the shadow
    cv2.putText(image, pred_labels[0].upper(), (text_x, text_y),
                font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    # Save the image with the prediction label
    cv2.imwrite(output_image_path+"/"+output_image_name, image)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle the uploaded file
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            # Save the uploaded file to a specific location
            upload_folder = input_image_path
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            image_path = os.path.join(upload_folder, file.filename)
            file.save(image_path)

            # Make a prediction using your function
            pred_labels, data = predict(image_path)

            save_prediction_image(image_path=image_path, data=data, pred_labels=pred_labels)

            # Render the result template with the prediction
            return render_template("result.html", image_path="static/"+output_image_name, prediction = pred_labels[0])

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)