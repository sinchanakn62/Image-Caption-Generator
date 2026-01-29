import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("model/image_caption_model.h5", compile=False)

# Load tokenizer
with open("model/tokenizer_fixed.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 34   # your trained max caption length

def generate_caption(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    in_text = "startseq"

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = np.pad(seq, (0, max_len - len(seq)), 'constant')
        seq = np.expand_dims(seq, axis=0)

        yhat = model.predict([img, seq], verbose=0)
        word = tokenizer.index_word.get(np.argmax(yhat), None)

        if word is None:
            break

        in_text += " " + word
        if word == "endseq":
            break

    final_caption = in_text.replace("startseq", "").replace("endseq", "")
    return final_caption.strip()

