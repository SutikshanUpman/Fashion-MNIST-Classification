# 👕 Fashion-MNIST Image Classification (ANN & CNN)

A **comparative deep learning project** on the **Fashion-MNIST** dataset exploring how **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** perform on low-resolution image data using **TensorFlow / Keras**.

This project emphasizes **learning behavior, regularization, and generalization**, rather than chasing accuracy numbers blindly.

---

## 📌 Project Highlights
- Implements both **ANN (MLP)** and **CNN** models on the same dataset
- Clean and reproducible **machine learning workflow**
- Strong focus on **overfitting control** and **validation analysis**
- Uses modern best practices:
  - **AdamW optimizer**
  - **Batch Normalization**
  - **Dropout**
  - **EarlyStopping** and **ReduceLROnPlateau**
- Demonstrates a realistic case where **CNN does not significantly outperform ANN** due to dataset limitations

---

## 📊 Dataset
**Fashion-MNIST**
- 60,000 training images
- 10,000 test images
- 28×28 grayscale images
- 10 clothing categories

### Preprocessing
- Pixel normalization (`/255`)
- Shape handling:
  - ANN → flattened `(784)`
  - CNN → `(28, 28, 1)`
- Sample visualization before training

---

## 🧠 Models Implemented

### 🔹 Artificial Neural Network (ANN)
**Purpose:** Strong baseline & regularization study

**Architecture**
```
Input (28×28)
→ Flatten
→ Dense (384) + GELU
→ BatchNorm + Dropout
→ Dense (256) + GELU
→ BatchNorm + Dropout
→ Dense (128) + GELU
→ Dropout
→ Dense (10) + Softmax
```

**Key Design Choices**
- **GELU activation** for smoother gradients
- **BatchNorm before Dropout** for stable convergence
- **AdamW optimizer** to reduce weight overfitting
- **EarlyStopping** to prevent over-training

**Performance**
- Test Accuracy: **~89–90%**
- Smooth convergence
- Minimal overfitting

---

### 🔹 Convolutional Neural Network (CNN)
**Purpose:** Evaluate spatial learning advantage on Fashion-MNIST

**Architecture**
```
Input (28×28×1)
→ Conv2D (32) + ReLU
→ BatchNorm
→ MaxPooling
→ Conv2D (64) + ReLU
→ BatchNorm
→ MaxPooling
→ Flatten
→ Dense (128) + ReLU
→ Dropout
→ Dense (10) + Softmax
```

**Observations**
- Higher **training accuracy** than ANN
- Validation accuracy plateaus early
- **Test accuracy similar to ANN (~91–92%)**

This behavior highlights **dataset saturation** and mild overfitting rather than model failure.

---

## 🚀 Training Configuration
- Optimizer: **AdamW**
- Loss: **Sparse Categorical Crossentropy**
- Metric: **Accuracy**
- Batch Size: `64`
- Epochs: up to `30`
- Validation Split: `20%`
- Callbacks:
  - EarlyStopping (restore best weights)
  - ReduceLROnPlateau

---

## 📈 Results Summary
| Model | Test Accuracy |
|-----|--------------|
| ANN | ~89–90% |
| CNN | ~91–92% |

> CNN improves representation capacity but does not significantly improve generalization on Fashion-MNIST due to low image complexity.

---

## 🧠 Key Learnings
- CNNs are **not automatically superior** on all image datasets
- Dataset complexity limits achievable gains
- Regularization often matters more than depth
- Validation behavior is more important than training accuracy
- Strong baselines are critical before adding complexity

---

## ▶️ How to Run
```bash
pip install tensorflow numpy matplotlib scikit-learn
jupyter lab
```

Open `main.ipynb` and run all cells sequentially.

> Trained models are not committed to keep the repository lightweight and reproducible.

---

## 🔮 Future Work
- Data augmentation to improve CNN generalization
- ANN vs CNN comparison plots
- Confusion matrix & error analysis
- Evaluation on higher-complexity datasets (e.g., CIFAR-10)
- Transfer learning with lightweight CNNs

---

## 👤 Author
**Sutikshan Upman**  
Aspiring AI / ML Engineer  
Focused on fundamentals, model behavior, and real-world learning
