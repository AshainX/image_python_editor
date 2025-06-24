import cv2
import numpy as np
import onnxruntime as ort   #pip install onnxruntime
from PIL import Image

def load_u2net_model(model_path="u2net.onnx"):
    return ort.InferenceSession(model_path)

def remove_background(image_bgr, session):
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img, (320, 320))
    img_input = img_input.astype(np.float32) / 255.0
    img_input = img_input.transpose((2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: img_input})[0][0][0]
    pred = cv2.resize(pred, (img.shape[1], img.shape[0]))
    mask = (pred > 0.5).astype(np.uint8) * 255
    b, g, r = cv2.split(image_bgr)
    rgba = cv2.merge((b, g, r, mask))
    return rgba

def convert_rgba_to_pil_image(rgba_image):
    return Image.fromarray(cv2.cvtColor(rgba_image, cv2.COLOR_BGRA2RGBA))
