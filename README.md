# 👕 Fashion-MNIST Classification

A clean and well-regularized **image classification project** on the **Fashion-MNIST** dataset using **Artificial Neural Networks (ANN)** built with **TensorFlow / Keras**.

This repository focuses on **understanding learning behavior, regularization, and evaluation**, rather than jumping straight to CNNs.

---

## 📌 Project Highlights
- Uses a **pure ANN (Dense Network)** — no CNN layers in the current implementation
- Emphasis on **training stability**, **overfitting control**, and **clean ML workflow**
- Proper use of **Batch Normalization**, **Dropout**, and **Early Stopping**
- Achieves **~89–90% accuracy** on Fashion-MNIST with an MLP

---

## 🧠 Model Architecture (ANN)

Input (28×28 grayscale image)  
→ Flatten  
→ Dense (384) + GELU  
→ BatchNorm + Dropout  
→ Dense (256) + GELU  
→ BatchNorm + Dropout  
→ Dense (128) + GELU  
→ Dropout  
→ Dense (10) + Softmax  

### 🔍 Design Choices
- **GELU activation** for smoother gradients
- **Batch Normalization before Dropout** for stable convergence
- **AdamW optimizer** to reduce weight overfitting
- **EarlyStopping** to prevent training past optimal epoch

---

## 📊 Dataset
**Fashion-MNIST**
- 60,000 training images
- 10,000 test images
- 28×28 grayscale images
- 10 clothing categories

### Preprocessing
- Pixel normalization (/255)
- One-hot encoding of labels
- Visualization of samples before training

---

## 🚀 Training Configuration

Optimizer: AdamW  
Loss: Categorical Crossentropy  
Metric: Accuracy  
Batch Size: 64  
Epochs: Up to 30 (Early Stopping enabled)  
Validation: 20% split  

Training curves show **minimal overfitting** and **healthy generalization gap**.

---

## 📈 Results
- **Test Accuracy:** ~89–90%
- Smooth convergence
- No validation collapse
- Model stops at optimal epoch via EarlyStopping

This performance is **strong for a pure ANN** on image data.

---

## ▶️ How to Run

pip install tensorflow numpy matplotlib scikit-learn  
jupyter lab  

Open `main.ipynb` and run all cells sequentially.

> Trained model files are intentionally **not committed** to keep the repository lightweight and reproducible.

---

## 📚 Key Learnings
- Dense networks can still perform competitively on image tasks
- Proper **regularization > deeper networks**
- EarlyStopping is often more valuable than extra epochs
- Clean Git practices matter in ML projects

---

## 🔮 Future Work

### 🔹 Model Comparisons
- LogisticRegression (max_iter=1000)
- RandomForestClassifier (n_estimators=100)
- ANN vs CNN performance comparison

### 🔹 Deep Learning Enhancements
- CNN-based architecture
- Custom training loop with tf.GradientTape
- Visualization of hidden layer activations
- Confusion matrix & error analysis

---

## 👤 Author
**Sutikshan Upman**  
Aspiring AI Engineer | Exploring Neural Networks & ML Foundations
