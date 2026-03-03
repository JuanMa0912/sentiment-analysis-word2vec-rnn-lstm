# 📊 Sentiment Analysis with Word2Vec, RNN and LSTM

Comparative study of machine learning and deep learning models for binary sentiment classification on the IMDb 50K Movie Reviews dataset.

---

## 📌 Project Overview

This project implements and compares three approaches for sentiment analysis:

- **Logistic Regression + Word2Vec (average embeddings)**
- **Simple RNN**
- **LSTM**

The objective is to evaluate whether modeling full token sequences (RNN/LSTM) improves performance over a simpler averaged-embedding representation.

---

## 📂 Dataset

- **Dataset:** IMDb 50K Movie Reviews  
- **Task:** Binary classification (Positive / Negative)  
- **Balanced dataset:** 25K positive, 25K negative reviews  

---

## 🧠 Methodology

### 1️⃣ Text Preprocessing
- Lowercasing
- Tokenization
- Padding & truncation
- `MAX_LEN` = 95th percentile of sequence lengths

### 2️⃣ Word Embeddings
- Word2Vec (100 dimensions)
- Embedding layer initialized with pretrained weights
- `trainable=False` (frozen for fair comparison)

### 3️⃣ Models

#### 🔹 Logistic Regression
- Input: Average Word2Vec embeddings
- Fast and deterministic training

#### 🔹 Simple RNN
- 64 recurrent units
- Dropout = 0.3
- EarlyStopping (patience=3)

#### 🔹 LSTM
- 64 LSTM units
- Dropout = 0.3
- EarlyStopping (patience=3)
- Same hyperparameters as RNN (controlled comparison)

---

## ⚙️ Experimental Design Decisions

- Frozen embedding to isolate classifier performance.
- Padding = `'post'`, Truncating = `'pre'`.
- Moderate architecture size to avoid overfitting.
- Same architecture and hyperparameters for RNN and LSTM (control of variables).

---

## 📊 Results

| Model                  | Accuracy | Observations |
|------------------------|----------|--------------|
| Logistic Regression    | 0.8679   | Best overall performance |
| RNN                    | 0.7381   | Less stable training |
| LSTM                   | 0.7499   | More stable than RNN |

---

## 🔎 Key Findings

- Logistic Regression with averaged embeddings outperformed recurrent networks.
- LSTM showed more stable convergence than RNN.
- Increased model complexity did not translate into better performance.
- For balanced binary sentiment classification, averaged embeddings captured sufficient semantic information.

---

## 🖥️ Requirements

```bash
python >= 3.9
tensorflow
keras
gensim
scikit-learn
pandas
numpy
matplotlib
