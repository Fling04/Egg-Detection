# Eggshell Crack Detection Using Digital Image Processing Techniques

Automated eggshell crack detection using Digital Image Processing (DIP) techniques and deep feature extraction with ResNet50.

---

## Overview

This project investigates how Digital Image Processing (DIP) techniques affect the performance of automated eggshell crack detection.

Rather than focusing on developing a new deep learning model, the objective was to analyze how different preprocessing methods influence feature extraction and anomaly detection. Images were processed using several computer vision techniques before being passed through a pretrained ResNet50 network used as a fixed feature extractor. An Isolation Forest model then classified eggs as either **normal** or **cracked**.

The project demonstrates that carefully designed preprocessing pipelines can significantly improve machine learning performance by emphasizing meaningful image features while suppressing irrelevant information.

---

## Objectives

* Evaluate the impact of image preprocessing on crack detection.
* Compare multiple Digital Image Processing techniques.
* Measure how preprocessing affects anomaly detection performance.
* Determine which preprocessing pipeline produces the most reliable results.

---

## Dataset

The final dataset contains **1,979 egg images** collected from multiple public datasets.

* **Training**

  * 1,511 normal eggs

* **Testing**

  * 377 normal eggs
  * 91 cracked eggs

Because publicly available cracked egg images are limited, the dataset contains significant variation in lighting, background clutter, shell texture, and crack size, making the task more representative of real industrial environments.

---

## Image Processing Pipeline

Every image first underwent baseline preprocessing:

* Grayscale conversion
* Gaussian Blur (3×3)
* Image resizing (256×256)

Several additional preprocessing techniques were then evaluated independently.

### Spatial Cropping

Removes unnecessary background information so the model focuses primarily on the egg.

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

Improves local contrast while minimizing excessive noise amplification.

### Morphological Processing

Reduces shell texture variation by smoothing small speckles that may otherwise resemble cracks.

### Edge Detection

Combines Sobel and Laplacian operators to emphasize crack boundaries and shell contours.

### Combined Pipeline

The final pipeline combines:

* Spatial Cropping
* CLAHE
* Edge Detection

to maximize crack visibility while reducing background noise.

---

## Machine Learning Pipeline

```text
Raw Image
      ↓
Grayscale + Gaussian Blur
      ↓
Selected Preprocessing Method
      ↓
ResNet50 Feature Extraction
      ↓
Isolation Forest
      ↓
Normal Egg / Cracked Egg
```

ResNet50 was used strictly as a pretrained feature extractor with frozen weights. Rather than retraining the network, extracted feature vectors were classified using an Isolation Forest trained only on normal eggs.

This approach isolates the impact of preprocessing from the learning capacity of the neural network.

---

## Technologies

* Python
* OpenCV
* NumPy
* TensorFlow / Keras
* ResNet50
* Scikit-Learn
* Isolation Forest
* Matplotlib

---

## Experimental Results

| Method                |   Accuracy | Crack Precision | Crack Recall |   F1 Score |
| --------------------- | ---------: | --------------: | -----------: | ---------: |
| Crop                  |     75.82% |          52.24% |       38.46% |     44.30% |
| CLAHE                 |     76.65% |          55.56% |       32.97% |     41.38% |
| Edge Detection        |     73.63% |          43.59% |       18.68% |     26.15% |
| Morphology (3×3)      |     80.22% |          62.67% |       51.65% |     56.63% |
| Morphology (5×5)      |     81.87% |      **68.66%** |       50.55% |     58.23% |
| Combined (No Crop)    |     78.57% |          59.42% |       45.05% |     51.25% |
| **Combined Pipeline** | **83.24%** |          67.86% |   **62.64%** | **65.14%** |

---

## Key Findings

* Background removal through spatial cropping significantly improved classification performance.
* CLAHE enhanced local contrast, making crack features more distinguishable under varying illumination.
* Morphological operations reduced false positives caused by natural shell texture.
* Edge detection alone removed too much contextual information, reducing recall.
* Combining preprocessing techniques produced the strongest overall performance across all evaluation metrics.

The experiments demonstrate that preprocessing is not merely a preparation step—it directly influences the quality of the learned feature representation and overall anomaly detection performance.

---

## Future Improvements

Potential directions for future work include:

* Larger and more balanced datasets
* Semantic segmentation for precise crack localization
* Modern architectures such as EfficientNet or Vision Transformers
* Real-time deployment for industrial conveyor belt inspection
* Evaluation using dedicated anomaly detection frameworks

---

## Repository Structure

```text
.
├── notebooks/
├── src/
├── images/
├── report/
│   └── Final_Report.pdf
└── README.md
```

---

## Full Technical Report

The complete project report, including methodology, implementation details, and experimental analysis, is available in **report/Final_Report.pdf**.
