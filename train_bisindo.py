# Activate the virtual environment first
# .\bisindo_env\Scripts\activate

from ultralytics import YOLO
import os
import torch  # Import torch to check CUDA availability

# Main function to train the model
def main():
    # 1. Load pre-trained YOLOv8 model
    print("Loading pre-trained YOLOv8 model...")
    model = YOLO('yolov8s.pt')

    # Check if GPU is available
    if torch.cuda.is_available():
        print("CUDA (GPU) is available and will be used for training.")
    else:
        print("CUDA (GPU) is not available. Training will use CPU (may be slower).")

    # 2. Specify path to data.yaml file
    data_yaml_path = 'bisindo-sign-language/data.yaml'

    # Ensure the data.yaml file exists
    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml file not found at '{data_yaml_path}'")
        print("Make sure your folder structure is correct or adjust the data_yaml_path.")
        return
    else:
        print(f"Using data.yaml from: {data_yaml_path}")

    # 3. Train the model
    print("\nStarting model training...")
    results = model.train(
        data=data_yaml_path,
        epochs=200,  # Increase epochs to 200 (previously 50)
        imgsz=640,
        batch=-1,
        name='bisindo_detection_v2',
        workers=8,
        mosaic=1.0  # **IMPORTANT CHANGE**: Adding Mosaic augmentation
        # hsv_h=0.015, # Hue augmentation
        # hsv_s=0.7,   # Saturation augmentation
        # hsv_v=0.4,   # Value (brightness) augmentation
        # degrees=15.0 # Random rotation (if more than what Roboflow applies is desired)
    )

    print("\nTraining completed!")
    print(f"Training results saved in: {model.trainer.save_dir}")
    print("The best model is usually found at: " + os.path.join(model.trainer.save_dir, 'weights', 'best.pt'))

if __name__ == '__main__':
    main()

# Note: This script assumes you have a data.yaml file in the same directory as the script.