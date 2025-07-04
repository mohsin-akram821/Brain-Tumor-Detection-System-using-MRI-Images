from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

# Create Flask app
app = Flask(__name__)

# Set and create upload folder
app.config['UPLOAD_FOLDER'] = './uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
model = load_model('models/VGG16_trained_model.h5')

# Class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Prediction function
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prediction_class_index = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction, axis=1)[0]

    if class_labels[prediction_class_index] == "notumor":
        return 'No Tumor', confidence_score
    else:
        return f'Tumor: {class_labels[prediction_class_index]}', confidence_score

# Route for home
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)
            return render_template('index.html',
                                   result=result,
                                   confidence=f'{confidence * 100:.2f}%',
                                   file_path=f'/uploads/{file.filename}')
    return render_template('index.html')

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
