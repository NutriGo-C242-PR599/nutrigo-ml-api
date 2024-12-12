import os
import cv2
import numpy as np
import pytesseract
import tensorflow as tf
import re
import time
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

print("All libraries imported successfully!")


def ocr_on_roi(image, xmin, ymin, xmax, ymax):
    """
    Extracts the Region of Interest (ROI) from the image and performs OCR on it.

    Args:
        image_path (str): Path to the image.
        x (int): The x-coordinate of the top-left corner of the ROI.
        y (int): The y-coordinate of the top-left corner of the ROI.
        width (int): The width of the ROI.
        height (int): The height of the ROI.

    Returns:
        str: The OCR text result from the ROI.
    """
    # Load the image
    #image = cv2.imread(image)

    # Extract the Region of Interest (ROI)
    roi = image[ymin:ymax, xmin:xmax]

    # Convert the ROI to grayscale for better OCR results
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Perform OCR on the ROI
    ocr_result = pytesseract.image_to_string(roi)
    cv2.imwrite('roi.jpg', gray_roi)
    return ocr_result



# Load the TensorFlow Lite model once
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to perform detection
def tflite_detect_image(interpreter, image, lblpath, min_conf=0.5, savepath='static', txt_only=False):
    # Load the label map into memory
    image_raw = image.copy()
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Load image and resize to expected shape [1xHxWx3]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating-point model (non-quantized model)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence scores

    detections = []
    rois = []  # List to store ROIs

    # Loop over all detections and filter based on the confidence threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            # Get bounding box coordinates
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # Get the label for the detected class
            object_name = labels[int(classes[i])]
            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

            # Extract ROI (Region of Interest)
            roi = image[ymin:ymax, xmin:xmax]
            rois.append((object_name, roi))  # Save ROI with its label

    if not txt_only:
        # Draw rectangles and labels on the image
        for detection in detections:
            cv2.rectangle(image, (detection[2], detection[3]), (detection[4], detection[5]), (10, 255, 0), 2)
            label = '%s: %.2f%%' % (detection[0], detection[1] * 100)
            cv2.putText(image, label, (detection[2], detection[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            ts = time.time()
            filename = f'output_image_{int(ts)}.jpg'
            output_image_path = os.path.join(savepath, filename)
            cv2.imwrite(output_image_path, image)

        # You can return the processed image, or save it to a file
        # cv2.imwrite(os.path.join(savepath, 'output_image.jpg'), image)

    return detections, rois, filename,image_raw # Return detections and ROIs


def find_line_with_word(text, word):
    """
    Finds and returns the entire line containing the specified word.

    Parameters:
        text (str): The input text to search.
        word (str): The word to search for.

    Returns:
        list: A list of lines containing the word.
    """
    # Compile a regex pattern for the word with word boundaries
    pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)

    # Split the text into lines
    lines = text.splitlines()

    # Find and return lines containing the word
    matched_lines = [line for line in lines if pattern.search(line)]
    return matched_lines

# Example usage:

# Load the model once
# interpreter = load_model('model/detect.tflite')
# image = cv2.imread('test_image.jpg')
# # Now, use it for detection
# detections, rois = tflite_detect_image(interpreter, image, 'model\labels.txt')

# print("Detections:", detections)
# print("ROIs:", rois)
# words = ["gula", "protein", "karbohidrat", "energi"]

# for idx, detection in enumerate(detections):
#     _, _, xmin, ymin, xmax, ymax = detection
#     width = xmax - xmin
#     height = ymax - ymin
#     print(f"ROI {idx + 1}: x={xmin}, y={ymin}, width={width}, height={height}")
#     x = idx+1
#     y = xmin
#     w = width
#     h = height

#     ocr_result = ocr_on_roi(image_path='test_image.jpg', x=x, y=y, width=w, height=h)
    
#     for word in words:
#         lines = find_line_with_word(ocr_result, word)
#         print(f"\nLines containing '{word}':")
#         for line in lines:
#             print(line)
    
    # print(f"OCR Result for ROI {idx + 1}: {ocr_result}")

