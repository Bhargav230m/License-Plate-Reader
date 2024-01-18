import cv2 as cv
import easyocr
import json

# Preprocess the cropped image for better ocr of license numbers
def preprocess(cropped_image):
    """
    Preprocessor for the reducing noise from the image
    Then eroding it to brighten up the thesholded crop
    Then we return the img
    """
    
    grayscale_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)  # Converts the image to grayscale
    _, thresholded_image = cv.threshold(grayscale_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Getting the threshold of the image
    img = cv.bitwise_not(thresholded_image)
    img = cv.erode(img, (9, 9), 8)  # Eroding to get a better and ligthen up image
    
    return img

# Read the license plate from the cropped image
def read_license_plate(cropped_image):
    """
    Here we initialize the reader that reads the license plates from the cropped image
    Then we get the preprocessed image from the preprocessor
    Then we read the license plate and return the results
    """
    
    reader = easyocr.Reader(['en'])  # Initializes the reader
    img = preprocess(cropped_image)  # Preprocess the crop
    result = reader.readtext(img)  # We read the license number off
    
    return result

# Extract the results and coordinates
def extract_results(model_results, frame):
    """
    Here we loop through the results
    We convert the results to a list then to json
    We get the box results
    Then we extract the xyxy coordinates from the box
    We read the license plate and print the results
    """
    
    for r in model_results:  # Looping through the results
        if r:  # If result then execute the inside code
            data_string = r.tojson()  # Convert the raw res to list
            data_json = json.loads(data_string)  # Convert the list to json
            data = data_json[0]["box"]  # Reading the box results
            cropped_image = frame[int(data["y1"]):int(data["y2"]), int(data["x1"]):int(data["x2"])]  # Cropping the image with the box coordinates
            result = read_license_plate(cropped_image)  # Use the read license plate function to read the license plates
            return result  # Returning the result
    return None  # None is returned
