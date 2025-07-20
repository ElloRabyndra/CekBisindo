from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import torch  # Added torch to check CUDA

# Main function for detection
def main():
    # Make sure to activate the virtual environment (bisindo_env) before running this script.

    # Path to your best model
    model_path = 'runs/detect/bisindo_detection_v1/weights/best.pt'

    # Ensure the model exists
    if not os.path.exists(model_path):
        print(f"Error: Best model not found at '{model_path}'")
        print("Make sure you have trained the model and the path to 'best.pt' is correct.")
        return  # Use return instead of exit()

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Display whether GPU is used or not
    if torch.cuda.is_available():
        print("CUDA (GPU) is available and will be used for detection.")
    else:
        print("CUDA (GPU) is not available. Detection will use CPU (may be slow).")

    image_to_detect = 'sample/sample_3.png'

    # Ensure the image/folder to detect exists
    if not os.path.exists(image_to_detect):
        print(f"Error: Image or folder '{image_to_detect}' not found.")
        print("Please replace 'image_to_detect' with a valid path.")
        return

    print(f"\nRunning detection on: {image_to_detect}")

    # Perform detection. The model will automatically handle image or folder.
    # save=True will save the result image to the 'runs/detect/predict/' folder
    # conf: confidence threshold (minimum confidence to display detections)
    # iou: NMS IoU threshold (to reduce overlapping bounding boxes)
    results = model(image_to_detect, save=True, conf=0.25, iou=0.7, workers=8)  # Added workers

    # Display detection results (optional, for a single image)
    # If you're detecting multiple images, this will likely only show the last one.
    # To see all results, check the 'runs/detect/predict/' folder

    for r in results:
        # r.plot() returns an image with bounding boxes and labels drawn
        im_bgr = r.plot()
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

        plt.figure(figsize=(10, 8))
        plt.imshow(im_rgb)
        plt.title("BISINDO Detection Result")
        plt.axis('off')
        plt.show()

        # Print detection results in text format
        print(f"\nDetections for image: {r.path}")
        if r.boxes:  # If there are detections
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()  # Bounding box coordinates [x1, y1, x2, y2]
                label = model.names[class_id]
                print(f"  - Label: {label}, Confidence: {confidence:.2f}, Box: [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
        else:
            print("  No objects detected.")

    print("\nDetection completed!")
    print("Detected images saved in folder: runs/detect/predictX/ (depending on the latest output folder)")
    
if __name__ == '__main__':
    main()

# Note: This script assumes you have a data.yaml file in the same directory as the script.