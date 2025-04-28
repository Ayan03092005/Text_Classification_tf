# 📚 Project Title:  
**Movie Review Sentiment Classification using TensorFlow Hub and IMDB Dataset**

---

# 📝 Project Description:

In this project, we build a simple yet powerful **text classification model** to predict the **sentiment** (positive or negative) of **movie reviews**.  
We use the popular **IMDB movie review dataset** from **TensorFlow Datasets** and **pre-trained word embeddings** (GNews Swivel) from **TensorFlow Hub** to speed up learning and improve performance.

The project involves:

- Loading and preparing the IMDB review data.
- Using **GNews Swivel embeddings** (20-dimensional vectors) to represent the input text.
- Building a **Sequential model** with:
  - Pre-trained embedding layer (`hub.KerasLayer`)
  - A hidden `Dense` layer with ReLU activation
  - A final `Dense` layer with Sigmoid activation for binary classification.
- Training the model for **25 epochs** to achieve good accuracy.
- Evaluating the model on unseen test data.
- Predicting the sentiment of new movie reviews.

This project demonstrates how **transfer learning** with **pre-trained embeddings** can quickly and effectively solve a basic Natural Language Processing (NLP) task without needing a huge custom model.

---

# 🎯 Key Points:

| Component        | Details |
|------------------|---------|
| Dataset | IMDB Reviews (from TensorFlow Datasets) |
| Embedding | GNews Swivel 20-dimension embeddings (from TensorFlow Hub) |
| Model Type | Sequential Neural Network |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Epochs | 25 |
| Metrics | Accuracy |
| Goal | Predict whether a movie review is positive or negative |

---

# 📈 Results:
- After 25 epochs, the model typically achieves **~85–87% validation accuracy**.
- Predictions on new examples show good performance in classifying positive vs negative reviews.

---

# 🚀 Tools Used:
- **Python 3**
- **TensorFlow 2.x**
- **TensorFlow Datasets (tfds)**
- **TensorFlow Hub (hub)**
- **Numpy**

---

# 🧠 Why this Approach?
- **Pre-trained embeddings** reduce the amount of training data needed and improve generalization.
- **Simple model architecture** makes it easy to understand and fast to train.
- **Transfer Learning** is a powerful method for achieving high performance with low effort in NLP tasks.
