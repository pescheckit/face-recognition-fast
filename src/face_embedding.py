import cv2
import numpy as np

# No global import of face_recognition or TensorFlow here either.

def get_face_embedding_facerecognition(face_image):
    """
    Uses the face_recognition library to get a 128-d embedding.
    face_image: a numpy array in RGB format
    """
    import face_recognition  # Local import

    encodings = face_recognition.face_encodings(face_image)
    return encodings[0] if encodings else None

def get_feature_extractor():
    """
    Creates and returns a MobileNetV2 feature extractor (requires TensorFlow).
    Local import so that if you don't use MobileNet,
    you won't import TF or see TF logs.
    """
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    # Optionally set GPU memory growth or handle GPU config if you want:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    model = Model(
        inputs=base_model.input,
        outputs=GlobalAveragePooling2D()(base_model.output)
    )
    return model

def preprocess_image(image):
    """
    Resizes the image to 224x224 and applies MobileNetV2 preprocessing.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    image_resized = cv2.resize(image, (224, 224))
    image_resized = np.expand_dims(image_resized, axis=0)
    image_resized = preprocess_input(image_resized)
    return image_resized

def get_face_embedding_mobilenet(model, face_pixels):
    """
    Generates a MobileNet-based embedding (1x1280 vector by default).
    face_pixels: an RGB image (or BGR, but we'll resize + preprocess anyway).
    """
    # We do local import inside get_feature_extractor, so not needed here 
    # unless you want to do more TF stuff directly.
    face_pixels = preprocess_image(face_pixels)
    embedding = model.predict(face_pixels)  # shape (1, 1280)
    return embedding.flatten()
