import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('unet_model.keras')


def segment_image(image_path, threshold):
    # Read the original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize and normalize the image for prediction
    image_resized = cv2.resize(image_gray, (256, 256))
    image_resized = image_resized / 255.0
    image_resized = np.expand_dims(image_resized, axis=-1)
    image_resized = np.expand_dims(image_resized, axis=0)

    # Predict the crack areas
    prediction = model.predict(image_resized)
    predicted_image = np.squeeze(prediction, axis=0)
    predicted_image = np.squeeze(predicted_image, axis=-1)
    predicted_image = (predicted_image > threshold).astype(np.uint8)

    # Resize the predicted mask back to original image size
    predicted_image = cv2.resize(predicted_image, (image.shape[1], image.shape[0]))

    # Create a copy of the original image to overlay the cracks
    overlay_image = image_rgb.copy()

    # Highlight the crack areas in red
    overlay_image[predicted_image == 1] = [255, 0, 0]  # Color the crack areas red

    # Combine the original image with the overlay image
    result_image = image_rgb.copy()
    result_image[predicted_image == 1] = overlay_image[predicted_image == 1]

    return result_image


def segment_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))), True)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_resized = cv2.resize(gray, (256, 256))
            image_resized = image_resized / 255.0
            image_resized = np.expand_dims(image_resized, axis=-1)
            image_resized = np.expand_dims(image_resized, axis=0)
            prediction = model.predict(image_resized)
            predicted_image = np.squeeze(prediction, axis=0)
            predicted_image = np.squeeze(predicted_image, axis=-1)
            predicted_image = (predicted_image > 0.5).astype(np.uint8)
            predicted_image = cv2.resize(predicted_image, (frame.shape[1], frame.shape[0]))
            overlay_frame = frame.copy()
            overlay_frame[predicted_image == 1] = [0, 0, 255]  # Color the crack areas red
            result_frame = frame.copy()
            result_frame[predicted_image == 1] = overlay_frame[predicted_image == 1]
            out.write(result_frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def update_threshold(val, image_path):
    threshold = val / 100
    result_image = segment_image(image_path, threshold)
    cv2.imshow('Segmented Crack', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))


def choose_file(file_type):
    if file_type == 'image':
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            cv2.namedWindow('Segmented Crack')
            cv2.createTrackbar('Threshold', 'Segmented Crack', 50, 100, lambda val: update_threshold(val, file_path))
            update_threshold(50, file_path)  # Initialize with default threshold 0.5
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif file_type == 'video':
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            segment_video(file_path)
            messagebox.showinfo("Info", "Video segmentation complete. Output saved as 'output.avi'")


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    while True:
        file_type = None
        while file_type not in ['image', 'video']:
            file_type = messagebox.askquestion("Choose File Type", "Do you want to segment an image or a video?",
                                               type='yesnocancel')
            if file_type == 'yes':
                file_type = 'image'
            elif file_type == 'no':
                file_type = 'video'
            else:
                return

        choose_file(file_type)

        if messagebox.askyesno("Continue", "Do you want to process another file?") == False:
            break


if __name__ == "__main__":
    main()
