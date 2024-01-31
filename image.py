import cv2
import easyocr
import streamlit as st
import numpy as np

reader = easyocr.Reader(['en'])

def extract_text(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to improve OCR results
    resized_image = cv2.resize(gray_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Apply adaptive thresholding to the resized image
    _, threshold_image = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply blur to the thresholded image
    blurred_image = cv2.GaussianBlur(threshold_image, (5, 5), 0)

    # Apply edge detection to the blurred image
    edge_detected_image = cv2.Canny(blurred_image, 50, 150)

    try:
        # Perform OCR using EasyOCR on the thresholded image
        result = reader.readtext(blurred_image)
    except Exception as e:
        st.error(f"An error occurred during text extraction: {str(e)}")
        result = []

    return result, gray_image, blurred_image, edge_detected_image

def main():
    st.title("Text Extraction and Image Processing with EasyOCR and Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image using OpenCV
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)

        # Display the original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Buttons to display different images
        if st.button("Show Grayscale Image"):
            _, gray_image, _, _ = extract_text(image)
            st.image(gray_image, caption="Grayscale Image", use_column_width=True)

        if st.button("Show Blurred Image"):
            _, _, blurred_image, _ = extract_text(image)
            st.image(blurred_image, caption="Blurred Image", use_column_width=True)

        if st.button("Show Edge Detection"):
            _, _, _, edge_detected_image = extract_text(image)
            st.image(edge_detected_image, caption="Edge Detection", use_column_width=True)

        # Button to extract text
        if st.button("Extract Text"):
            result, _, _, _ = extract_text(image)

            # Display the extracted text
            st.header("Extracted Text:")
            for detection in result:
                st.text(detection[1])

if __name__ == "__main__":
    main()
