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

    # --- Placeholder Model Function ---
    # This is where you will integrate your actual machine learning model.
    # For now, it just draws a sample bounding box and returns dummy text.
    def detect_handwritten_text(image: np.ndarray) -> tuple[np.ndarray, str]:
        """
        A placeholder function to simulate handwritten text detection.

        Args:
            image: A NumPy array representing the input image from the webcam.

        Returns:
            A tuple containing:
            - The image with a bounding box drawn on it (as a NumPy array).
            - A dummy string for the detected text.
        """
        # Create a copy of the image to draw on
        processed_image = image.copy()
        detected_text = "Placeholder: Hello World!"

        # --- MODEL INTEGRATION POINT ---
        # 1. Pre-process the 'image' (e.g., resize, normalize, grayscale) as required by your model.
        # 2. Pass the pre-processed image to your model for inference.
        #    predictions = your_model.predict(processed_image)
        # 3. Process the model's output to get bounding boxes and recognized text.
        #    For example, let's assume your model outputs a list of tuples,
        #    where each tuple is (box, text).
        #
        # Example of what you might do with model output:
        # for (box, text) in predictions:
        #     (x, y, w, h) = box
        #     cv2.rectangle(processed_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        #     cv2.putText(processed_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        #     detected_text += text + " "
        # ---------------------------------

        # For this boilerplate, we'll just draw a static, sample box.
        # This helps visualize where the output would be.
        h, w, _ = processed_image.shape
        start_point = (int(w * 0.2), int(h * 0.3))  # Top-left corner
        end_point = (int(w * 0.8), int(h * 0.7))  # Bottom-right corner
        color_bgr = (36, 255, 12)  # A bright green color
        thickness = 2

        cv2.rectangle(processed_image, start_point, end_point, color_bgr, thickness)
        cv2.putText(
            processed_image,
            "Your detected text would appear here",
            (start_point[0], start_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_bgr,
            2,
        )

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
            rgb_image, caption="Image captured from webcam.", use_container_width=True  # use_column_width is deprecated
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
