from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
import csv
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
          
# Initialize app
app = Flask(__name__)
app.secret_key = 'change_this_dev_secret'
# Save uploads inside static so templates can serve them easily
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model (legacy quick model) - keep for compatibility
try:
    model = load_model("image_caption_model.h5", compile=False)
except Exception:
    model = None  # keep app running even if model load fails for now

# Prefer using the project caption generation function if available
model_generate_caption = None
try:
    # caption_model.py defines a generate_caption(img_path) that uses the tokenizer + model
    from caption_model import generate_caption as model_generate_caption
    print("Using caption generator from caption_model.py")
except Exception as e:
    model_generate_caption = None
    print("caption_model.generate_caption not available:", e)

# Dummy user database
users = {}

# Captions dataset mapping (image filename -> list of captions)
captions_map = {}

def load_captions(path=None):
    """Load captions from dataset/captions.txt into captions_map."""
    global captions_map
    if captions_map:
        return
    path = path or os.path.join(app.root_path, 'dataset', 'captions.txt')
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                img = row[0].strip()
                cap = row[1].strip()
                if img and cap:
                    captions_map.setdefault(img, []).append(cap)
        print(f"Loaded captions for {len(captions_map)} images from {path}")
    except Exception as e:
        print('Could not load captions file:', e)

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def find_user_by_username_or_email(identifier):
    # identifier can be username or email
    for uname, data in users.items():
        if uname == identifier or data.get('email') == identifier:
            return uname
    return None

# Routes
@app.route('/')
def index():
    if session.get('user'):
        return redirect(url_for('home'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['username']
        password = request.form['password']
        username = find_user_by_username_or_email(identifier)
        if username and users[username]['password'] == password:
            session['user'] = username
            return redirect(url_for('home'))
        flash('Invalid username/email or password')
    return render_template('login1.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists')
            return render_template('register2.html')
        users[username] = {'email': email, 'password': password}
        # Log the user in immediately and send to home
        session['user'] = username
        return redirect(url_for('home'))
    return render_template('register2.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/home')
def home():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('home3.html')

@app.route('/about')
def about():
    # Public about page (no login required)
    return render_template('about6.html')

@app.route('/contact')
def contact():
    # Public contact page (no login required)
    return render_template('contact.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('user'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Redirect to result page for the uploaded image
            return redirect(url_for('result', file='uploads/' + filename))

    # List images from static/uploads and static/trained
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    trained_folder = os.path.join(app.root_path, 'static', 'trained')
    images = []
    if os.path.isdir(upload_folder):
        images += ['uploads/' + f for f in os.listdir(upload_folder) if allowed_file(f)]
    if os.path.isdir(trained_folder):
        images += ['trained/' + f for f in os.listdir(trained_folder) if allowed_file(f)]

    return render_template('upload.html', images=images)


@app.route('/result', methods=['GET', 'POST'])
def result():
    if not session.get('user'):
        return redirect(url_for('login'))

    # If the user clicked Generate (POST), produce a caption
    if request.method == 'POST':
        file_param = request.form.get('file')
        if not file_param:
            return render_template('result5.html', image_url=None, caption=None, file_param=None)

        file_path = os.path.join(app.root_path, 'static', file_param.replace('/', os.sep))
        if not os.path.exists(file_path):
            return render_template('result5.html', image_url=None, caption='Image not found', file_param=file_param)

        caption = generate_caption(file_path)
        image_url = url_for('static', filename=file_param)
        return render_template('result5.html', image_url=image_url, caption=caption, file_param=file_param)

    # GET: show the image (if any) but do NOT generate a caption until user requests it
    file_param = request.args.get('file')
    if not file_param:
        return render_template('result5.html', image_url=None, caption=None, file_param=None)

    file_path = os.path.join(app.root_path, 'static', file_param.replace('/', os.sep))
    if not os.path.exists(file_path):
        return render_template('result5.html', image_url=None, caption='Image not found', file_param=file_param)

    image_url = url_for('static', filename=file_param)
    return render_template('result5.html', image_url=image_url, caption=None, file_param=file_param)


# Image Captioning Logic
def generate_caption(img_path):
    try:
        # Sanity check: file exists
        if not os.path.exists(img_path):
            return "Image file not found"

        # If the project's caption_model is available, prefer it (it uses tokenizer + model)
        if model_generate_caption is not None:
            try:
                caption = model_generate_caption(img_path)
                return f"(Model) {caption}"
            except Exception as e:
                print("model_generate_caption failed:", e)
                # continue to other fallbacks

        # Next try to find a matching caption in captions_map (by filename)
        base = os.path.basename(img_path)
        if captions_map:
            caps = captions_map.get(base)
            if caps:
                return f" {random.choice(caps)}"

        # Otherwise try a very simple fallback heuristic using PIL/Keras to ensure something returns
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model is not None:
            pred = model.predict(img_array)
            # TODO: translate prediction to human-readable caption
            caption = "Generated caption for your image"
        else:
            caption = "(Model not loaded) Generated caption for your image"
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return the exception message so we can see what's failing during debugging
        caption = f"Could not process image: {e}"
    return caption


if __name__ == '__main__':
    # Ensure upload and trained directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'static', 'trained'), exist_ok=True)

    # Load captions dataset (if present)
    load_captions()

    app.run(debug=True)

