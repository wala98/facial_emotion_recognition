# Facial Emotion Recognition System
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

This computer vision project focuses on **Facial Emotion Recognition (FER)** using both transfer learning and custom convolutional neural network (CNN). The system processes live webcam input and predicts facial emotions in real-time .

---

## 🌟 Motivation

Facial emotions are crucial in human communication. This project aims to build an intelligent system that can recognize emotions like happiness, anger, or sadness from facial expressions — enabling applications in:

- 📚 **Live class monitoring**
- 🧠 **Mental health assistance**
- 🧾 **Customer sentiment analysis**
- 🕹 **Human-computer interaction in games**




## Features

- Real-time emotion detection using webcam
- Fine-tuned  models with PyTorch
- Face detection using Haar Cascades (OpenCV)
- Supports 7 emotion classes (e.g., happy, sad, angry...)



---

## 🗃 Dataset

**FER2013** dataset from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

- 35,887 grayscale 48x48 facial images
- 7 labeled emotion categories
- Preprocessed (aligned and centered faces)

---

## Model Architectures

This project explores **five models**:

### 1. Custom CNN
### 2. EfficientNet-B0 *(Transfer Learning)*
### 3. ResNet18 *(Transfer Learning)*
### 4. GoogLeNet/Inception *(Transfer Learning)*
### 5. MobileNetV2 *(Transfer Learning)*

---
## 🧑‍💻 How to Use

1. **Download the following files** into the same directory:
   - `haarcascade/` folder (contains Haar Cascade XML for face detection)
   - `best_model_full_efficientnet_b0.pth` (your trained EfficientNet-B0 model)
   - `efficientnet_b0_history.pkl` (training history for analysis/plotting)
   - `face_emotion_recognition.py` (main script for webcam-based emotion recognition)

2. **Open** `face_emotion_recognition.py` in your preferred code editor (e.g., VSCode, PyCharm).

3. **Run the script** to activate your webcam and start detecting facial emotions in real time:

   ```bash
   python face_emotion_recognition.py

---
## 📸 
![Image](https://github.com/wala98/facial_emotion_recognition/blob/b5abaeb3a8d217534a61ff1f933cc7d63c2b96ae/output%20exemple/Screenshot%202025-05-09%20135243.png)
![Image](https://github.com/wala98/facial_emotion_recognition/blob/b5abaeb3a8d217534a61ff1f933cc7d63c2b96ae/output%20exemple/Screenshot%202025-05-09%20135259.png)
![Image](https://github.com/wala98/facial_emotion_recognition/blob/b5abaeb3a8d217534a61ff1f933cc7d63c2b96ae/output%20exemple/Screenshot%202025-05-09%20135831.png)
