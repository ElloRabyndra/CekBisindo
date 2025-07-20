# CekBisindo: BISINDO Sign Language Detection

## Project Overview 

CekBisindo is a dedicated sign language detection project specifically designed for **BISINDO (Bahasa Isyarat Indonesia)**. This project aims to accurately identify individual static hand signs representing letters of the alphabet, providing a foundational step towards more comprehensive sign language recognition systems. Leveraging a pre-labeled image dataset, CekBisindo employs state-of-the-art object detection techniques to recognize hand gestures, making it a valuable tool for learning, communication, or assistive technologies related to BISINDO.

## Tech Stack 🛠️

This project is built using modern deep learning and computer vision libraries:

* **Python**: The primary programming language used for development.
* **PyTorch**: An open-source machine learning framework used for building and training the deep learning model.
* **Ultralytics YOLOv8**: The cutting-edge object detection model used for sign recognition, chosen for its efficiency and accuracy in real-time applications.
* **OpenCV (cv2)**: Utilized for image processing tasks, including loading, manipulation, and displaying detection results.
* **Numpy**: Essential for numerical operations and array manipulation.
* **Matplotlib**: Used for visualizing training metrics and detection results.

## Dataset 📂

The core of this project relies on a comprehensive dataset of BISINDO hand signs.

* [cite_start]**Source**: The dataset was acquired from Roboflow Universe[cite: 1, 2, 3], a platform for computer vision datasets.
* [cite_start]**Name**: BISINDO Sign Language - v1[cite: 2].
* [cite_start]**Content**: It consists of 3240 static images [cite: 2][cite_start], meticulously labeled with bounding boxes for **27 distinct classes**: all letters from 'A' to 'Z' and an additional 'NOTHING' class[cite: 1].
* [cite_start]**Format**: The annotations are provided in **YOLOv8 format**[cite: 3], which is directly compatible with the Ultralytics YOLO framework.
* [cite_start]**Pre-processing & Augmentation**: The dataset underwent auto-orientation and was resized to 640x640 pixels[cite: 3]. [cite_start]Crucially, it was augmented to create three versions of each source image, applying transformations such as horizontal/vertical flips, random cropping, and random rotations[cite: 3].
* [cite_start]**License**: Public Domain[cite: 1].
* **Download Link**: You can download the dataset directly from Roboflow Universe here: [https://universe.roboflow.com/rr-pguxk/bisindo-sign-language/dataset/1](https://universe.roboflow.com/rr-pguxk/bisindo-sign-language/dataset/1)

**Note:** The dataset files themselves (`train/images`, `train/labels`, etc.) are not included in this repository due to their size. Only the `data.yaml` configuration file is included, which points to the dataset structure. You will need to download and extract the dataset into the project's root directory, ensuring the structure matches the `data.yaml` configuration.

## Getting Started 🚀

Follow these steps to set up the project, train your model, and run detections.

### 1. Project Setup

1.  **Clone the Repository (or Download)**:
    ```bash
    git clone [https://github.com/USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME # Replace with your repository name, e.g., CekBisindo
    ```
2.  **Download and Place the Dataset**:
    * Download the "YOLOv8 (pytorch)" format dataset from the Roboflow link provided above.
    * Extract the downloaded `.zip` file. You should get a folder named `bisindo-sign-language`.
    * Place this `bisindo-sign-language` folder directly into the root of your project directory (e.g., `CekBisindo/`). The `data.yaml` file should be inside this folder.
        ```
        CekBisindo/
        ├── bisindo-sign-language/
        │   ├── train/
        │   ├── valid/
        │   ├── test/
        │   └── data.yaml
        ├── train_bisindo.py
        ├── evaluate_model.py
        ├── detect_bisindo.py
        └── .gitignore
        ```

### 2. Environment Setup

It's highly recommended to use a Python virtual environment to manage dependencies.

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv bisindo_env
    ```
2.  **Activate the Virtual Environment**:
    * **Windows**:
        ```bash
        .\bisindo_env\Scripts\activate
        ```
    * **macOS/Linux**:
        ```bash
        source bisindo_env/bin/activate
        ```
    You should see `(bisindo_env)` at the beginning of your command prompt, indicating the environment is active.

3.  **Install Required Libraries**:
    Install all necessary Python packages within your active virtual environment.

    ```bash
    # Install PyTorch (choose the appropriate version based on your GPU/CPU)
    # For NVIDIA GPU (e.g., CUDA 11.8):
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

    # For CPU ONLY:
    # pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

    # Install Ultralytics YOLOv8
    pip install ultralytics

    # Install other utilities
    pip install opencv-python numpy matplotlib
    ```

### 3. Train the Model 🧠

Train the YOLOv8 model using your BISINDO dataset.

1.  **Open `train_bisindo.py`**: Review the `epochs` and `name` parameters. The default `epochs=200` and `name='bisindo_detection_v2'` are set for improved generalization. You can adjust them as needed.
2.  **Run the Training Script**:
    Make sure your `bisindo_env` is active.
    ```bash
    python train_bisindo.py
    ```
    This process can take significant time depending on your hardware (especially if you're using a CPU instead of a GPU). Training results, including the trained model weights (`best.pt`), will be saved in the `runs/detect/bisindo_detection_v2/weights/` directory.

### 4. Evaluate the Model 📊

After training, evaluate your model's performance on the validation and test datasets.

1.  **Open `evaluate_model.py`**: Ensure the `model_path` points to your newly trained model (e.g., `'runs/detect/bisindo_detection_v2/weights/best.pt'`).
2.  **Run the Evaluation Script**:
    Make sure your `bisindo_env` is active.
    ```bash
    python evaluate_model.py
    ```
    This script will output metrics like mAP (mean Average Precision), Precision, and Recall on both the validation and test sets, giving you insights into your model's accuracy.

### 5. Run Inference (Detection) 💡

Use your trained model to detect BISINDO signs on new images.

1.  **Open `detect_bisindo.py`**:
    * **Crucially, modify the `image_to_detect` variable** to point to the image or folder of images you want to test.
        ```python
        image_to_detect = 'path/to/your/new/image.jpg' # Example: 'bisindo-sign-language/test/images/X_jpg.rf.763c3d5268c741031c26c117e3f8b054.jpg'
        # Or uncomment the line below to detect on an entire folder
        # folder_to_detect = 'path/to/your/image_folder/'
        # image_to_detect = folder_to_detect
        ```
    * Ensure the `model_path` points to your trained model (`'runs/detect/bisindo_detection_v2/weights/best.pt'`).
2.  **Run the Detection Script**:
    Make sure your `bisindo_env` is active.
    ```bash
    python detect_bisindo.py
    ```
    The script will display detected signs with bounding boxes and save the output images in a `runs/detect/predictX/` folder.

---

Feel free to contribute, open issues, or suggest improvements!