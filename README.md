# CoNLL-2003 Named Entity Recognition (NER) Comparison

> ### 🌟 Special Thanks & Sincere Appreciation
> **A big, heartfelt thank you to Eng. Karim Essam** for his exceptional guidance, mentorship, and invaluable support throughout this project. His practical insights and encouragement made tackling these experiments and building this project an inspiring learning experience.

---

## 📌 Overview

This project is a hands-on exploration and comparative benchmark of **Named Entity Recognition (NER)** using the canonical **CoNLL-2003** English dataset. 

The goal was to trace how entity recognition techniques evolved: starting from traditional machine learning with sparse **n-gram features** and word embeddings (**Word2Vec**, **GloVe**), up to recurrent deep learning models (**Bidirectional LSTMs** with concatenated representations and learned embeddings).

To make the results interactive, the project includes a custom **Streamlit Web Application** featuring live sentence analysis, interactive token visualization, and an evaluation leaderboard.

---

## 🚀 Key Results & Leaderboard

All models were evaluated on the CoNLL-2003 validation set using strict entity-level span F1 (evaluating full entity chunks rather than naive token-level accuracy):

| Rank | Model Architecture | Precision | Recall | Span F1-Score |
| :---: | :--- | :---: | :---: | :---: |
| 🥇 | **Bi-LSTM + Combined Features** (Word2Vec + GloVe + Context) | **85.90%** | 78.36% | **81.96%** |
| 🥈 | **N-gram + Logistic Regression** (Rich handcrafted context) | 80.12% | **82.04%** | **81.07%** |
| 🥉 | **Bi-LSTM + Trainable Embedding Layer** | 73.56% | 73.70% | **73.63%** |
| 4 | **GloVe (100d) + Logistic Regression** | 66.88% | 59.09% | **62.74%** |
| 5 | **Word2Vec (100d) + Logistic Regression** | 50.30% | 33.94% | **40.53%** |

### 💡 Key Takeaways
1. **Bi-LSTM with concatenated pre-trained features** achieved the highest overall F1 score (**81.96%**) and top precision (**85.90%**), proving the value of combining dense semantic representations with sequential recurrence.
2. **N-gram with handcrafted features** proved to be an exceptionally strong baseline (**81.07% F1**), demonstrating that local context features (capitalization, prefixes, suffixes, surrounding words) carry significant signal for NER.
3. **Stand-alone embeddings (Word2Vec/GloVe) with a simple linear classifier** struggled with polysemy and boundary detection without sequential recurrent context.

---

## 🧠 Model Architectures

1. **N-gram + Logistic Regression**:
   - Extracts character prefixes (first 3 chars), suffixes (last 3 chars), capitalization flags (`is_title`, `is_upper`, `is_digit`), and sliding word bigrams (`previous+word`, `word+next`).
   - High-dimensional sparse feature representation with an L2-regularized classifier.

2. **Word2Vec + Logistic Regression**:
   - Pre-trained Skip-gram / CBOW vectors (50-dimensional).
   - Window context averaging across preceding, target, and succeeding tokens.

3. **GloVe + Logistic Regression**:
   - Global log-bilinear word representation vectors capturing co-occurrence statistics across Wikipedia.

4. **Bi-LSTM + Combined Pretrained Features**:
   - Two-layer Bidirectional LSTM taking concatenated Word2Vec, GloVe, and context representations.
   - Learns forward and backward sequential dependencies simultaneously.

5. **Bi-LSTM + Learnable Embedding**:
   - An end-to-end differentiable lookup table trained directly with the recurrent network from scratch.

---

## 🏷️ CoNLL-2003 Entity Classes

The dataset uses BIO (Beginning, Inside, Outside) tagging for four named entity categories:

* 👤 **PER (Person):** Real-world human names (e.g., *Bill Gates*, *Marie Curie*).
* 🏢 **ORG (Organization):** Companies, institutions, government bodies (e.g., *Microsoft*, *United Nations*).
* 📍 **LOC (Location):** Countries, cities, geographic landmarks (e.g., *United States*, *Geneva*, *California*).
* 🏷️ **MISC (Miscellaneous):** Nationalities, events, works of art, languages (e.g., *British*, *World Cup*).

---

## 💻 Streamlit Web Application

The interactive web demo provides an intuitive interface to test models and explore findings:

* **Locked Static Sidebar:** Clean desktop navigation that remains steady without intrusive open/close toggles.
* **Modern Google Material Symbols:** High-definition icons replacing text symbols across all tabs.
* **Live Text Analysis:** Type or paste any sentence, or click one of the quick sample chips (*Tech Founders*, *Apple in CA*, *UN in Geneva*).
* **Token Highlighting:** Color-coded token pills showing predicted entity tags and extracted entities table.
* **Benchmark Leaderboard:** View charts comparing Precision, Recall, and F1 across models, with an option to upload custom evaluation CSVs.

---

## 🛠️ Installation & Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/MohameddTamerr/conll2003-ner-comparison.git
cd conll2003-ner-comparison
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit demo
```bash
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
conll2003-ner-comparison/
├── .streamlit/
│   └── config.toml           # Enforced light theme & UI configuration
├── assets/
│   └── ner_illustration.svg  # Custom vector hero artwork
├── data/
│   ├── train.txt             # CoNLL-2003 training split
│   ├── valid.txt             # CoNLL-2003 validation split
│   └── test.txt              # CoNLL-2003 test split
├── models/                   # Serialized model weights & vectorizers
│   ├── ngram_model.joblib
│   ├── word2vec.model
│   ├── glove_vectors.kv
│   ├── feature_lstm.pth
│   └── embedding_lstm.pth
├── notebook/
│   ├── CoNLL2003_NER_Model_Comparison.ipynb # Training & experimentation
│   └── ner_validation_results.csv           # Benchmark scores
├── app.py                    # Streamlit web application
├── inference.py              # Preprocessing & inference pipeline
├── requirements.txt          # Python package dependencies
├── .gitignore
└── README.md
```

---

## 👤 Author

Developed by **Mohamed Tamer**  
Special acknowledgment and gratitude to **Eng. Karim Essam**.
