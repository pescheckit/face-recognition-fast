import cv2
import numpy as np

def get_face_embedding_facerecognition(face_image):
    """
    Uses the face_recognition library to get a 128-d embedding.
    Ensures proper face format before extracting embeddings.
    """
    import face_recognition
    import numpy as np

    if face_image is None or face_image.shape[0] == 0 or face_image.shape[1] == 0:
        return None  # Prevent passing empty image

    # Convert to expected format
    face_image = np.ascontiguousarray(face_image, dtype=np.uint8)

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
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    image_resized = cv2.resize(image, (224, 224))
    image_resized = np.expand_dims(image_resized, axis=0)
    image_resized = preprocess_input(image_resized)
    return image_resized

def get_face_embedding_mobilenet(model, face_pixels):
    face_pixels = preprocess_image(face_pixels)
    embedding = model.predict(face_pixels)  # shape (1, 1280)
    return embedding.flatten()

# ------------------------------------------------------------------
#  ArcFace embedding using InsightFace
# ------------------------------------------------------------------
def get_face_embedding_arcface(face_image):
    """
    Uses InsightFace directly with the ArcFace model for embedding.
    Skips FaceAnalysis and uses ArcFaceONNX directly.
    You need: pip install insightface onnxruntime
    """
    import logging
    import insightface
    import numpy as np
    import cv2
    import os
    import onnxruntime as ort

    logger = logging.getLogger(__name__)

    try:
        # Ensure the image is large enough for ArcFace
        face_resized = cv2.resize(face_image, (112, 112))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        # Convert to the format expected by model
        face_array = np.transpose(face_rgb, (2, 0, 1))  # (3, 112, 112)
        face_array = np.expand_dims(face_array, axis=0)  # (1, 3, 112, 112)
        face_array = face_array.astype(np.float32)
        face_array = (face_array - 127.5) / 127.5  # Normalize

        # Load the ArcFace ONNX model
        from insightface.model_zoo import get_model
        model_path = os.path.expanduser('~/.insightface/models/buffalo_l/w600k_r50.onnx')
        model = get_model(model_path)
        model.prepare(ctx_id=0)

        # Get embedding using forward method
        embedding = model.forward(face_array)
        return embedding.flatten()
    
    except Exception as e:
        logger.warning(f"[ArcFace] Direct model approach failed: {e}")
        logger.info("[ArcFace] Trying with simple normalization...")
        return None


# ------------------------------------------------------------------
#  DeepFace embedding
# ------------------------------------------------------------------
def get_face_embedding_deepface(face_image):
    """
    Using the DeepFace library. We can specify a model_name, e.g. "ArcFace", "Facenet", "VGG-Face", etc.
    pip install deepface
    """
    from deepface import DeepFace
    import logging
    import cv2
    import os
    import tempfile

    logger = logging.getLogger(__name__)

    # DeepFace API may have changed - represent() might not accept img_array param directly
    # Let's try to save the image temporarily and use img_path
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
            temp_path = temp.name
            # Save the face image to the temporary file
            cv2.imwrite(temp_path, face_image)
        
        # Use the temp file path for the DeepFace.represent call
        embedding_objs = DeepFace.represent(
            img_path=temp_path,
            model_name="ArcFace",  # or "Facenet", "Facenet512", "VGG-Face", "SFace", etc.
            enforce_detection=False,
            detector_backend="skip"  # Skip detection since we already have a cropped face
        )
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"[DeepFace] Error in represent(): {e}")
        # Try alternative approach if the first fails
        try:
            logger.info("[DeepFace] Trying alternative approach...")
            
            # Sometimes the newer DeepFace versions use different parameter names
            # Let's try the verify function which returns embeddings internally
            result = DeepFace.verify(
                img1_path=face_image,  # Pass the array directly, newer versions might support this
                img2_path=face_image,  # Same image, we just want the embedding
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="skip"
            )
            
            # Extract embedding from verification result (if available)
            if hasattr(result, 'embedding1') and result.embedding1 is not None:
                embedding_objs = [{'embedding': result.embedding1}]
            else:
                logger.error("[DeepFace] Couldn't extract embedding from verify result")
                return None
                
        except Exception as e2:
            logger.error(f"[DeepFace] Error in alternative approach: {e2}")
            return None

    if not embedding_objs or len(embedding_objs) == 0:
        logger.warning("[DeepFace] No embedding returned.")
        return None

    # embedding_objs is a list of dict(s). Typically something like:
    # [{'embedding': [...], 'facial_area': {...}, 'dominant_emotion':...}, ...]
    vec = embedding_objs[0]['embedding']  # should be a list or numpy array
    emb = np.array(vec, dtype=np.float32)
    return emb
    