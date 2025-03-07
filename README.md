# Image Classification with CNN

Build a Convolutional Neural Network (CNN) model to classify images from a given dataset into predefined categories/classes.

[Task Descriptions and Project Instructions](https://github.com/ironhack-labs/project-1-deep-learning-image-classification-with-cnn)


## 📊 Project Results
In this project, we classified images from the animals 10 data set.
- Pre-processed data 
- Built a sequential CNN model 
- Optimized the model
- Prediction accuracy of: 81.63% 

---

## 📂 Repository Folders and Files

Here is a short description of the folder and files available on the repository.


### Documents
- Presentation

### Notebooks  
- split_validation_set: split the data set to one set for training and testing (90%) and a second one to make predictions (10%)
- model_1.ypynb : The starting point model
- model_optimized_ypnb: The optimized model
- test_images: folder of unseen images to test on the model

---

## 📦 Required Modules and Dependencies

The following modules are required for running this project:

### 🔹 **System & File Management**
- `os` – Handles file paths and directory operations.  
- `zipfile` – Extracts compressed datasets.  
- `time` – Measures execution time of processes.  
- `google.colab.drive` – Mounts Google Drive for dataset storage.  
- `gdown` – Downloads datasets from Google Drive.

### 🔹 **Data Preprocessing & Image Handling**
- `tensorflow.keras.preprocessing.image` – Loads, processes, and augments images.  
- `ImageDataGenerator` – Applies **data augmentation** to improve model generalization.  
- `opencv` (`cv2`) – (Optional) Handles advanced image processing.  
- `numpy` – Handles numerical operations for image arrays.  

### 🔹 **Deep Learning: TensorFlow & Keras**
- `tensorflow` – Provides the framework for CNN training and inference.  
- `tensorflow.keras.models.Sequential` – Defines a sequential CNN model.  
- `tensorflow.keras.layers.Conv2D` – Extracts features from images using convolutional layers.  
- `tensorflow.keras.layers.MaxPooling2D` – Reduces spatial dimensions while retaining key features.  
- `tensorflow.keras.layers.Flatten` – Converts feature maps into a **1D vector** for dense layers.  
- `tensorflow.keras.layers.Dense` – Fully connected layers for classification.  
- `tensorflow.keras.layers.Dropout` – Regularization technique to prevent **overfitting**.  
- `tensorflow.keras.layers.BatchNormalization` – Normalizes activations for **stable training**.  
- `tensorflow.keras.regularizers.l2` – Applies **L2 Regularization** to prevent **overfitting**.  

### 🔹 **Model Optimization & Callbacks**
- `tensorflow.keras.callbacks.ReduceLROnPlateau` – Adjusts the **learning rate** dynamically when training slows down.  
- `tensorflow.keras.callbacks.EarlyStopping` – Stops training when validation performance **stagnates**.  
- `tensorflow.keras.optimizers.Adam` – Adaptive **optimization algorithm** for faster convergence.  
- `tensorflow.keras.backend as K` – Provides **low-level functions** for manipulating tensors.  

### 🔹 **Model Evaluation & Metrics**
- `sklearn.metrics.confusion_matrix` – Computes the confusion matrix for model predictions.  
- `sklearn.metrics.classification_report` – Provides a summary of **precision, recall, and F1-score**.  
- `sklearn.metrics.precision_score` – Measures how many of the predicted positives are actually correct.  
- `sklearn.metrics.recall_score` – Evaluates how well the model identifies **true positives**.  
- `sklearn.metrics.f1_score` – Balances **precision and recall** into a single metric.  

### 🔹 **Data Visualization**
- `matplotlib.pyplot` – Plots **loss curves, accuracy trends, and sample images**.  
- `seaborn` – Creates **confusion matrix heatmaps** for visualizing model predictions.  

---

## ⚙️ Installation
Before running the project, ensure that all required dependencies are installed.

### **🔹 Install Using `requirements.txt`**
To install all necessary libraries at once, run:

```bash
pip install -r requirements.txt

### Additional Folders
**test_images**: folder of unseen images to test on the model

---
