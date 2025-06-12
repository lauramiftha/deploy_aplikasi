from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import base64
import io

# Flask app init
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_PERMANENT'] = True

# MySQL config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'cp'
mysql = MySQL(app)

# Model config
IMG_SIZE = (128, 128)
MODEL_PATH = os.path.join(os.getcwd(), "best_model_0.97.h5")
class_labels = ["Bercak Daun", "Embun Jelaga", "Karat Daun", "Sehat"]

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), name='conv1'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', name='conv2'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', name='conv3'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    return model

# Load model
try:
    model = build_model()
    model.load_weights(MODEL_PATH)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

# Image preprocessing
def prepare_image(image):
    try:
        image = image.convert("RGB").resize(IMG_SIZE)
        image = img_to_array(image) / 255.0
        return np.expand_dims(image, axis=0)
    except:
        return None

# Grad-CAM
def generate_gradcam_base64(image_array, model, class_idx, layer_name='conv3'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original = (image_array[0] * 255).astype("uint8")
    superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode('.jpg', superimposed_img)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    return base64_img

# Feature maps
def save_feature_maps_base64(image_array, model):
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(image_array)
    encoded_maps = []

    for i, fmap in enumerate(activations):
        fmap = fmap[0]
        fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
        num_filters = fmap.shape[-1]
        cols = 8
        rows = (num_filters // cols) + int(num_filters % cols != 0)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes = axes.flatten()
        for idx in range(num_filters):
            axes[idx].imshow(fmap_norm[:, :, idx], cmap='gray')
            axes[idx].axis('off')
        for idx in range(num_filters, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_bytes = buf.read()
        base64_img = base64.b64encode(img_bytes).decode('utf-8')
        encoded_maps.append({
            'layer': layer_names[i],
            'image_base64': base64_img
        })

    return encoded_maps

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id, name, password FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        if user and check_password_hash(user[2], password):
            session['logged_in'] = True
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            flash('Login berhasil!', 'success')
            return redirect(url_for('layanan'))
        flash('Email atau password salah.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Password tidak cocok.', 'danger')
            return redirect(url_for('register'))
        hashed = generate_password_hash(password)
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (name, phone, email, password) VALUES (%s, %s, %s, %s)",
                       (name, phone, email, hashed))
        mysql.connection.commit()
        cursor.close()
        flash('Registrasi berhasil!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/layanan')
def layanan():
    if not session.get('logged_in'):
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))  # ✅ ini benar
    return render_template('layanan.html', user_name=session.get('user_name'))  

@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah logout.', 'info')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return jsonify({"error": "Anda harus login terlebih dahulu!"}), 403

    if 'file' not in request.files or request.files['file'].filename == "":
        return jsonify({"error": "File tidak ditemukan atau kosong!"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)

    try:
        with Image.open(file) as image:
            img_array = prepare_image(image)

        if img_array is None or model is None:
            return jsonify({"error": "Gagal memproses gambar!"}), 500

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        predicted_class = class_labels[class_idx]
        confidence = float(round(100 * prediction[0][class_idx], 2))

        gradcam_base64 = generate_gradcam_base64(img_array, model, class_idx)
        feature_maps_base64 = save_feature_maps_base64(img_array, model)

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "gradcam_base64": gradcam_base64,
            "feature_maps": feature_maps_base64
        })

    except Exception as e:
        return jsonify({"error": f"Gagal memproses file: {str(e)}"}), 500

@app.route('/result')
def result():
    if not session.get('logged_in'):
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))

    # Ambil data dari session
    prediction = session.get('prediction')
    confidence = session.get('confidence')
    gradcam_base64 = session.get('gradcam_base64')
    feature_maps = session.get('feature_maps')

    if not prediction or not gradcam_base64 or not feature_maps:
        flash('Data hasil prediksi tidak ditemukan.', 'danger')
        return redirect(url_for('layanan'))

    feature_maps_data = []
    for fmap in feature_maps:
        feature_maps_data.append((fmap['layer'], f"data:image/png;base64,{fmap['image_base64']}"))

    return render_template('result.html',
                           prediction=prediction,
                           confidence=confidence,
                           image_path=f"data:image/jpeg;base64,{gradcam_base64}",
                           feature_maps=feature_maps_data)


@app.route('/about')
def about():
    return render_template('about.html', title="Tentang Kami",
                           description="Aplikasi ini menggunakan CNN untuk mengklasifikasi penyakit daun jambu kristal secara otomatis.")

# Run app
if __name__ == '__main__':
    app.run(debug=True)
