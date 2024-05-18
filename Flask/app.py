from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import h5py
import os

app = Flask(__name__)

# Load the trained model
resnet_model = load_model('model.h5')

# Define class names
class_names = [
    'Ace of Clubs', 'Ace of Diamonds', 'Ace of Hearts', 'Ace of Spades',
    'Eight of Clubs', 'Eight of Diamonds', 'Eight of Hearts', 'Eight of Spades',
    'Five of Clubs', 'Five of Diamonds', 'Five of Hearts', 'Five of Spades',
    'Four of Clubs', 'Four of Diamonds', 'Four of Hearts', 'Four of Spades',
    'Jack of Clubs', 'Jack of Diamonds', 'Jack of Hearts', 'Jack of Spades',
    'joker',
    'King of Clubs', 'King of Diamonds', 'King of Hearts', 'King of Spades',
    'Nine of Clubs', 'Nine of Diamonds', 'Nine of Hearts', 'Nine of Spades',
    'Queen of Clubs', 'Queen of Diamonds', 'Queen of Hearts', 'Queen of Spades',
    'Seven of Clubs', 'Seven of Diamonds', 'Seven of Hearts', 'Seven of Spades',
    'Six of Clubs', 'Six of Diamonds', 'Six of Hearts', 'Six of Spades',
    'Ten of Clubs', 'Ten of Diamonds', 'Ten of Hearts', 'Ten of Spades',
    'Three of Clubs', 'Three of Diamonds', 'Three of Hearts', 'Three of Spades',
    'Two of Clubs', 'Two of Diamonds', 'Two of Hearts', 'Two of Spades'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    # Get the image file from the POST request
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        image_file = request.files['file']

        if image_file.filename == '':
            return redirect(request.url)

        if image_file:
            image_path = os.path.join('static/uploads', image_file.filename)
            image_file.save(image_path)
            # Load and preprocess the image
            img = load_img(image_path, target_size=(500, 500))
            img = img.resize((200, 200))  # Resize to the expected input shape of the model
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Make prediction
            y_pred = resnet_model.predict(x)
            class_idx = np.argmax(y_pred, axis=1)[0]
            predicted_class = class_names[class_idx] if class_idx < len(class_names) else "Unknown"

            return render_template('result.html', predicted_class=predicted_class)

    # Return a default response
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
