import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(
    "model/image_caption_model.h5",
    compile=False,
    custom_objects={"NotEqual": tf.math.not_equal}
)

model.save("model/image_caption_model_FIXED.h5")
print("MODEL FIXED SUCCESSFULLY")
