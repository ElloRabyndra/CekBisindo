from ultralytics import YOLO
import os
import torch 

# Main function for model evaluation
def main():

    # Path to the best model
    model_path = 'runs/detect/bisindo_detection_v1/weights/best.pt'

    # Ensure the model exists
    if not os.path.exists(model_path):
        print(f"Error: Best model not found at '{model_path}'")
        print("Make sure you have trained the model and the path to 'best.pt' is correct.")
        return  # Use return instead of exit()

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Check if GPU is available
    if torch.cuda.is_available():
        print("CUDA (GPU) is available and will be used for evaluation.")
    else:
        print("CUDA (GPU) is not available. Evaluation will use CPU (may be slow).")

    # Path to the data.yaml file
    data_yaml_path = 'bisindo-sign-language/data.yaml'

    print(f"\nEvaluating model on validation dataset (val)...")
    # Evaluation on validation dataset
    metrics_val = model.val(data=data_yaml_path, split='val', workers=8)  # Add workers

    print("\nEvaluation Results on Validation Set:")
    print(f"mAP50-95 (mean Average Precision @ IoU 0.50-0.95): {metrics_val.box.map}")
    print(f"mAP50 (mean Average Precision @ IoU 0.50): {metrics_val.box.map50}")
    print(f"Precision: {metrics_val.box.mp}")
    print(f"Recall: {metrics_val.box.mr}")

    # Evaluation on test dataset
    print(f"\nEvaluating model on test dataset (test)...")
    metrics_test = model.val(data=data_yaml_path, split='test', workers=8)  # Add workers

    print("\nEvaluation Results on Test Set:")
    print(f"mAP50-95 (mean Average Precision @ IoU 0.50-0.95): {metrics_test.box.map}")
    print(f"mAP50 (mean Average Precision @ IoU 0.50): {metrics_test.box.map50}")
    print(f"Precision: {metrics_test.box.mp}")
    print(f"Recall: {metrics_test.box.mr}")

if __name__ == '__main__':
    main()

# Note: This script assumes you have a data.yaml file in the same directory as the script.