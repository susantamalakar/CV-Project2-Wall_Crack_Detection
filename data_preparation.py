# data_preparation.py
import os
import cv2
import numpy as np

def create_mask(image):
    # Dummy function to create a binary mask for crack segmentation
    # Replace this with actual mask generation logic
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return mask

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            mask = create_mask(image)
            mask_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")
            cv2.imwrite(mask_path, mask)

def main():
    base_dir = "wall_crack_dataset"
    sets = ["train", "val", "test"]

    for set_name in sets:
        for category in ["crack", "no_crack"]:
            input_dir = os.path.join(base_dir, set_name, category)
            output_dir = os.path.join(base_dir, "masks", set_name, category)
            process_images(input_dir, output_dir)

if __name__ == "__main__":
    main()
