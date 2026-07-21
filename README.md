# 📰 Fake News Detection using Deep Learning (LSTM)

## 📌 Project Overview

Fake news has become a major challenge in today's digital world. This project aims to automatically classify news articles as **Fake** or **Genuine** using a Deep Learning approach.

The application uses **Natural Language Processing (NLP)** techniques along with an **LSTM (Long Short-Term Memory)** neural network to analyze news text and predict its authenticity. A user-friendly **Tkinter GUI** allows users to upload datasets, train the model, visualize performance, and test custom news articles.

---

## 🚀 Features

- Upload Fake News Dataset (CSV)
- Text preprocessing using NLP
- TF-IDF Feature Extraction
- LSTM Deep Learning Model
- Save and Load Trained Model
- Predict Fake or Genuine News
- Accuracy & Loss Graph Visualization
- GUI developed using Tkinter

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Matplotlib
- Tkinter

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── Main.py
├── news.csv
├── testNews.txt
├── testNews1.txt
├── model.json
├── model_weights.weights.h5
├── history.pckl
├── README.md
└── requirements.txt
```

---

## 📊 Workflow

```
Dataset
   │
   ▼
Text Preprocessing
   │
   ▼
Cleaning
(Removing Stopwords,
Punctuation,
Lemmatization)
   │
   ▼
TF-IDF Vectorization
   │
   ▼
Train/Test Split
   │
   ▼
LSTM Deep Learning Model
   │
   ▼
Prediction
(Fake / Genuine)
```

---

## ⚙️ Preprocessing Steps

- Convert text into lowercase
- Remove punctuation
- Remove stopwords
- Tokenization
- Lemmatization
- TF-IDF Vectorization
- Normalize feature vectors

---

## 🧠 Deep Learning Model

The model consists of:

- LSTM Layer (128 Units)
- Dropout Layer (0.2)
- LSTM Layer (128 Units)
- Dropout Layer (0.2)
- Dense Layer (32 Units, ReLU)
- Dropout Layer (0.2)
- Output Layer (Softmax)

Loss Function:
```
Sparse Categorical Crossentropy
```

Optimizer:
```
Adam
```

Evaluation Metric:
```
Accuracy
```

---

## 💻 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
python Main.py
```

---

## 📷 Application Modules

### Upload Dataset

Load the fake news dataset in CSV format.

### Preprocess Dataset

- Clean text
- Remove stopwords
- Generate TF-IDF vectors

### Run LSTM

Train the LSTM model or load the saved model.

### Accuracy & Loss Graph

Displays the model training performance.

### Test News Detection

Upload a text file or CSV containing news and classify it as:

- Genuine News
- Fake News

---

## 📈 Model Output

The model predicts:

```
Given news predicted as GENUINE
```

or

```
Given news predicted as FAKE
```

---

## 📌 Future Enhancements

- BERT-based Fake News Detection
- Transformer Models
- Real-time News Verification
- Flask/Django Web Deployment
- Mobile Application
- API Integration
- Live News Scraping

---

## 📚 Libraries Used

- tensorflow
- keras
- nltk
- sklearn
- pandas
- numpy
- matplotlib
- tkinter
- pickle

---

## 🎯 Learning Outcomes

Through this project I gained practical experience in:

- Deep Learning
- Natural Language Processing
- Text Classification
- LSTM Networks
- Machine Learning Pipelines
- Feature Engineering
- GUI Development using Tkinter
- Model Saving & Loading

---

## 👩‍💻 Author

**Bobbala Ramannagari Sai Grishma**

B.Tech Computer Science Engineering

Python | Machine Learning | Deep Learning | Generative AI | Full Stack Development


---
