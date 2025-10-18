
# Fashion MNIST Classification using Artificial Neural Network (ANN)

## 📘 Project Overview
This project demonstrates image classification on the **Fashion MNIST** dataset using a **pure Artificial Neural Network (ANN)** — *no Convolutional Neural Networks (CNNs)* or *Recurrent Neural Networks (RNNs)* are used.*

The model is implemented using **TensorFlow** and **Keras**, proving that even a simple feedforward ANN can achieve strong performance on image data.

---

## 🧠 Model Architecture
The ANN follows a **fully connected (dense) layer architecture**:

1. **Input Layer:** `Flatten(input_shape=(28, 28))` – reshapes 2D image data into 1D vectors.  
2. **Hidden Layers:**
   - Dense layer with **512 neurons**, activation = **GELU**
   - **Batch Normalization** for stable training
   - **Dropout (0.3)** to reduce overfitting
   - Dense layer with **256 neurons**, activation = **GELU**
3. **Output Layer:** Dense(10, activation='softmax') for multi-class classification.

> ⚠️ **Highlight:** This project uses *only an ANN*. There are **no CNN or RNN layers** — making it a demonstration of how dense layers alone can handle image recognition tasks.

---

## 📊 Dataset
**Fashion MNIST Dataset** (from TensorFlow Datasets):  
- 60,000 training images  
- 10,000 testing images  
- Each image is 28×28 grayscale, representing clothing categories (e.g., shirts, trousers, shoes).

**Preprocessing:**
- Normalized pixel values by dividing by 255.
- Visualized 9 sample images with labels before training.

---

## 🚀 Training Details
| Parameter | Value |
|------------|--------|
| Optimizer | RMSprop |
| Loss Function | Sparse Categorical Crossentropy |
| Metric | Accuracy |
| Epochs | 50 |
| Batch Size | 128 |
| Validation Split | 20% |

Plots of training vs validation accuracy were generated to visualize learning progress.

---

## 📈 Results
The ANN achieved solid accuracy on both training and test datasets, confirming that feedforward dense layers can perform well for simpler image classification problems without CNNs or RNNs.

---

## 🧰 Dependencies
Install the required libraries using:

```bash
pip install tensorflow numpy pandas matplotlib
```

---

## ▶️ Usage
1. Clone or download this repository.  
2. Open **`main.ipynb`** in Jupyter Notebook.  
3. Run all cells sequentially to train and evaluate the ANN model.

---

## 📚 Key Learnings
- Showcases how **ANNs** (without CNNs or RNNs) can classify image data.  
- Demonstrates the effects of **Batch Normalization** and **Dropout** on model generalization.  
- Provides an end-to-end example of model training, evaluation, and visualization using Keras.

---

## 👤 Author
**Sutikshan Upman**  
AI/ML Enthusiast | Exploring Neural Networks with TensorFlow

---

## 🪪 License
This project is released under the **MIT License**.
