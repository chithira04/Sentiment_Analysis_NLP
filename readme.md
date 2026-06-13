# IMDB Movie Review Sentiment Analysis using Deep Learning

## Project Overview

This project implements a Sentiment Analysis system for IMDB movie reviews using Natural Language Processing (NLP) and Deep Learning techniques. The objective is to classify movie reviews as Positive or Negative based on their textual content.

The project includes data preprocessing, exploratory data analysis, feature engineering, model development using LSTM and GRU networks, model evaluation, and deployment using Flask.

---

## Dataset

Dataset: IMDB Movie Reviews Dataset

* Total Reviews: 50,000
* Positive Reviews: 25,000
* Negative Reviews: 25,000

Dataset Source:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## Project Objectives

* Perform text preprocessing and cleaning.
* Explore the dataset using EDA techniques.
* Implement tokenization and sequence padding.
* Compare traditional machine learning and deep learning approaches.
* Build and evaluate LSTM and GRU models.
* Deploy the best model using Flask.
* Provide real-time sentiment predictions with confidence scores.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
* TensorFlow / Keras
* Flask

---

## Data Preprocessing

The following preprocessing techniques were applied:

* HTML tag removal
* Lowercase conversion
* Special character removal
* Stopword removal
* Tokenization
* Sequence padding

---

## Exploratory Data Analysis

The following analyses were performed:

* Class distribution visualization
* Review length distribution
* Word frequency analysis
* Word Cloud generation

---

## Models Implemented

### 1. Logistic Regression with TF-IDF

A baseline machine learning model using TF-IDF vectorization.

### 2. Bidirectional LSTM

Architecture:

* Embedding Layer
* Bidirectional LSTM Layer
* Dropout Layer
* Dense Layer
* Sigmoid Output Layer

### 3. Bidirectional GRU

Architecture:

* Embedding Layer
* Bidirectional GRU Layer
* Dropout Layer
* Dense Layer
* Sigmoid Output Layer

---

## Training Techniques

* Adam Optimizer
* Binary Crossentropy Loss
* Early Stopping
* ReduceLROnPlateau
* Validation Split

---

## Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

---

## Results

| Model                        | Accuracy |
| ---------------------------- | -------- |
| TF-IDF + Logistic Regression | ~89%     |
| Bidirectional LSTM           | ~91%     |
| Bidirectional GRU            | ~92%     |

The Bidirectional GRU model achieved the best overall performance.

---

## Flask Web Application

The trained model was deployed using Flask to provide real-time sentiment predictions.

Features:

* User-friendly web interface
* Review text input
* Sentiment prediction
* Confidence score display

---

## Project Structure

```text
Sentiment_Analysis_Project/
│
├── IMDB Dataset.csv
│
├── sentimentAnalysis.ipynb
│
├── sentiment_model.h5
├── tokenizer.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
```
## 📸 Screenshots

screenshot for both pages are in screenshot directory

---


## Installation

Clone the repository:

```bash
git clone <repository_url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 👩‍💻 Author

- CHITHIRA CV    

---

## 📄 License

This project is for educational purposes.
