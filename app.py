from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import tensorflow_hub as hub
#app = Flask(__name__)

# initialize model
model = tf.keras.models.load_model(("models/20221025-094643-full-image-set-mobilenetv2-Adam.h5"), custom_objects={"KerasLayer": hub.KerasLayer})

# load labels
labels_csv = pd.read_csv("D:\dog breed classification\labels\labels.csv")
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

def plot_pred(prediction_probabilities, labels, images, n=1):

    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

    pred_label = get_pred_labels(pred_prob)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    plt.title("{} {:2.0f}%".format(pred_label, np.max(pred_prob)*100, true_label, color=color))

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

def unbatchify(data):
    images = []
    labels = []

    for image, label in data.unbatch().as_numpy_iterator:
      images.append(image)
      labels.append(unique_breeds[np.argmax(label)])
    return images, labels

custom_image_path = ["C:/Users/Piyush Sharma/Desktop/dog breed classification/image/golden.PNG"]
custom_data = create_data_batches(custom_image_path, test_data=True)


custom_preds = model.predict(custom_data)

custom_pred_labels = [get_pred_labels(custom_preds[i]) for i in range(len(custom_preds))]



custom_images = []

for images in custom_data.unbatch().as_numpy_iterator():
   custom_images.append(images)

print(custom_images)

image = cv2.imread(custom_image_path[0])

# Define the font and other text properties
font = cv2.FONT_ITALIC
font_scale = 0.5
font_color = (0, 255, 0)  # White color in BGR
font_thickness = 1
font_line_type = cv2.LINE_AA
shadow_color = (0, 0, 0)  # Black shadow color in BGR


# Calculate the position for the text
text_size, _ = cv2.getTextSize(custom_pred_labels[0].upper(), font, font_scale, font_thickness)
text_x = 10  # X-coordinate for the text (adjust as needed)
text_y = 20  # Y-coordinate for the text (adjust as needed)

# Draw the text shadow first
cv2.putText(image, custom_pred_labels[0].upper(), (text_x+3 , text_y+3),
            font, font_scale, shadow_color, font_thickness, cv2.LINE_AA)

# Draw the main text on top of the shadow
cv2.putText(image, custom_pred_labels[0].upper(), (text_x, text_y),
            font, font_scale, font_color, font_thickness, cv2.LINE_AA)

# Save the image with the prediction label
cv2.imwrite('image_with_prediction.jpg', image)

# plot_pred(prediction_probabilities = custom_preds, labels = unique_breeds, images = custom_image)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/classift", methods=['POST'])
# def classify():
#     try:
#         upload_file = request.files('image')
#         if upload_file.filename !='':
#             image = Image.open(upload_file)
#             image = preprocess_image(image)
#             prediction = model.predict(image)

#             result = {"classification": "Classified Result Here"}
#             return jsonify(result)
#         else:
#             return jsonify({"error": "No file uploaded."})
#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__=='__main__':
#     app.run()