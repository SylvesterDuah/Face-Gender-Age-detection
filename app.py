from flask import Flask, request, render_template, jsonify
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model for gender detection
gender_model = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
gender_list = ['Male', 'Female']



# Load the model for age detection
age_model = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
age_list = ['0-2', '4-6', '8-12', '15-20', '21-24', '25-32', '33-43', '44-53', '60-100']


# Load some pre-trained data on face frontals from OpenCV (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory for uploads if it doesn't exist
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Process image for face detection, age, and gender prediction
def process_image(image_path):
    img = cv2.imread(image_path)
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    results = []
    for (x, y, w, h) in face_coordinates:
        face_img = img[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = gender_list[gender_preds[0].argmax()]

        age_model.setInput(blob)
        age_preds = age_model.forward()
        age = age_list[age_preds[0].argmax()]

        results.append({'gender': gender, 'age': age, 'x': x, 'y': y, 'w': w, 'h': h})

    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image-upload', methods=['GET'])
def image_upload():
    return render_template('img.html')

@app.route('/video-stream', methods=['GET'])
def video_stream():
    return render_template('video.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(uploads_dir, filename)
    file.save(save_path)
    
    results = process_image(save_path)
    os.remove(save_path)

    return jsonify({
        'message': 'Image uploaded and processed successfully',
        'filename': filename,
        'results': results
    })

@app.route('/upload-webcam-image', methods=['POST'])
def upload_webcam_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(uploads_dir, filename)
    file.save(save_path)
    
    results = process_image(save_path)
    os.remove(save_path)

    return jsonify({
        'message': 'Webcam image uploaded and processed successfully',
        'filename': filename,
        'results': results
    })

if __name__ == '__main__':
    app.run(debug=True)
