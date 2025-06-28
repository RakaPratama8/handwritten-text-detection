import streamlit as st

# st.set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="Handwritten Text Detection",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="auto",
)


def main():
    import cv2
    import numpy as np
    from PIL import Image
    from transformers import (
        AutoProcessor,
        AutoModelForVision2Seq,
    )  # Adjust import if needed
    import torch

    # Load model and processor once
    @st.cache_resource(show_spinner="Loading OCR model...")
    def load_model():
        processor = AutoProcessor.from_pretrained(
            "raka-pratama/Llama-3.2-Vision-handwritten-ocr"
        )
        model = AutoModelForVision2Seq.from_pretrained(
            "raka-pratama/Llama-3.2-Vision-handwritten-ocr"
        )
        return processor, model

    processor, model = load_model()

    # --- Placeholder Model Function ---
    # This is where you will integrate your actual machine learning model.
    # For now, it just draws a sample bounding box and returns dummy text.
    def detect_handwritten_text(image: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Use the Llama-3.2-Vision-handwritten-ocr model to detect handwritten text.
        """
        # Convert OpenCV image (BGR) to PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Preprocess image for the model
        inputs = processor(images=pil_image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = model.generate(**inputs)
            detected_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Optionally, draw a box or highlight (if your model returns bounding boxes, add here)
        processed_image = image.copy()
        # If bounding boxes are available, draw them here

        return processed_image, detected_text.strip()

    # --- Streamlit App Layout ---

    st.title("✍️ Handwritten Text Detection from Webcam")

    st.markdown(
        """
        This application uses your webcam to capture an image and then (simulates) detecting
        handwritten text within it.

        **How to use:**
        1.  Click the **"Take Photo"** button below.
        2.  Allow the browser to access your webcam.
        3.  Position your handwritten text in the frame and click the capture button.
        4.  The app will display the captured image and the detection results.
        """
    )

    # --- Webcam and Image Processing ---
    img_file_buffer = st.camera_input("Take Photo")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV, we have to convert it to a NumPy array
        bytes_data = img_file_buffer.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if cv_image is None:
            st.error(
                "Failed to decode image. Please try taking the photo again."
            )  # Potential error: decoding failed
            return

        # Convert BGR image from OpenCV to RGB for display
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.error(
                f"Error converting image color: {e}"
            )  # Potential error: conversion failed
            return

        st.subheader("Original Captured Image")
        st.image(
            rgb_image,
            caption="Image captured from webcam.",
            use_container_width=True,  # use_column_width is deprecated
        )

        with st.spinner("Detecting text..."):
            # Call the placeholder model function
            processed_image, detected_text = detect_handwritten_text(cv_image)

            try:
                processed_rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                st.error(
                    f"Error converting processed image color: {e}"
                )  # Potential error: conversion failed
                return

            st.subheader("Processed Image with Detections")
            st.image(
                processed_rgb_image,
                caption="Image with simulated text detection.",
                use_container_width=True,  # use_column_width is deprecated
            )

            st.subheader("Detected Text")
            st.code(detected_text, language=None)

    else:
        st.info("Waiting for a photo to be taken...")

    # --- Footer ---
    st.markdown(
        """
        ---
        *Built with [Streamlit](https://streamlit.io/)*
        """
    )


if __name__ == "__main__":
    main()
