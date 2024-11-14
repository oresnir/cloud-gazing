import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Load ResNet50 model
model = ResNet50(weights='imagenet')

def classify_cloud_shape(shape_image):
    """Classify shape using a pre-trained ResNet50 model."""
    resized_shape = cv2.resize(shape_image, (224, 224))
    image_array = img_to_array(resized_shape)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    label, score = decoded_predictions[0][1], decoded_predictions[0][2]
    return label, score

def simplify_contour(contour, epsilon_factor=0.01):
    """Simplify contour using the Douglas-Peucker algorithm."""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def find_contours(image):
    """Find clound contours and filter small ones."""
    # Convert to grayscale and apply GaussianBlur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize image to separate clouds
    _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to connect fragments
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Detect edges and find contours
    edges = cv2.Canny(morphed, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours
    image_height, image_width = image.shape[:2]
    min_contour_area = image_height * image_width * 0.001
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    return filtered_contours
