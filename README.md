# CoNLL-2003 Named Entity Recognition Benchmark

This project compares five named entity recognition (NER) models on the English CoNLL-2003 dataset. It includes a training and evaluation notebook, saved models, and a Streamlit app for testing predictions on custom text.

[Open the Streamlit demo](https://conll2003-ner-comparison.streamlit.app/)

## Validation Results

The models are ranked by entity-level F1 on the validation split. The notebook uses `seqeval` to score predictions based on matching entity boundaries and types. The [validation results CSV](notebook/ner_validation_results.csv) contains the full-precision scores.

| Rank | Model | Precision | Recall | F1 |
| ---: | --- | ---: | ---: | ---: |
| 1 | BiLSTM with combined features | **85.90%** | 78.36% | **81.96%** |
| 2 | N-gram features with logistic regression | 80.12% | **82.04%** | 81.07% |
| 3 | BiLSTM with trainable embeddings | 73.56% | 73.70% | 73.63% |
| 4 | GloVe with logistic regression | 66.88% | 59.09% | 62.74% |
| 5 | Word2Vec with logistic regression | 50.30% | 33.94% | 40.53% |

The combined-feature BiLSTM achieved the highest precision and F1. The n-gram baseline had the highest recall and finished within 0.89 percentage points of the top F1 score.

## Models

| Model | Input representation |
| --- | --- |
| N-gram features with logistic regression | Sparse features for the current token and its neighbors, including bigrams, three-character prefixes and suffixes, and case and digit indicators. |
| Word2Vec with logistic regression | Concatenated 50-dimensional Word2Vec vectors for the previous, current, and next tokens. Word2Vec is trained on the project’s training split. |
| GloVe with logistic regression | Concatenated 50-dimensional pre-trained GloVe vectors for the same three-token window. |
| BiLSTM with combined features | Fixed Word2Vec and GloVe vectors for each token, concatenated with the mean Word2Vec vector over the three-token window. |
| BiLSTM with trainable embeddings | A 100-dimensional embedding layer trained jointly with the bidirectional LSTM. |

The [training notebook](notebook/CoNLL2003_NER_Model_Comparison.ipynb) covers preprocessing, feature extraction, training, validation, and held-out test evaluation of the model selected on validation F1.

## Dataset and Labels

The repository includes the CoNLL-2003 training, validation, and test splits. The English dataset labels four entity types.

| Label | Entity type | Example |
| --- | --- | --- |
| `PER` | Person | Marie Curie |
| `ORG` | Organization | United Nations |
| `LOC` | Location | San Francisco |
| `MISC` | Miscellaneous | British |

In the BIO tagging scheme, `B-` marks the first token of an entity, `I-` marks subsequent tokens in the same entity, and `O` marks a token outside any entity. For example, in “She visited San Francisco,” the tags for `She`, `visited`, `San`, and `Francisco` are `O`, `O`, `B-LOC`, and `I-LOC`, respectively.

## Run the Demo Locally

Install Python 3.11, then clone the repository.

```bash
git clone https://github.com/MohameddTamerr/conll2003-ner-comparison.git
cd conll2003-ner-comparison
```

Create a virtual environment using the commands for your platform.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the dependencies and start the app.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` to run a model on custom text, inspect token labels and extracted entities, or compare evaluation results. The benchmark view also accepts a custom CSV with `Model`, `Precision`, `Recall`, and `F1` columns.

## Repository Guide

| Path | Contents |
| --- | --- |
| [`app.py`](app.py) | Streamlit demo and benchmark views |
| [`inference.py`](inference.py) | Model loading, feature extraction, and prediction |
| [`data/`](data/) | CoNLL-2003 dataset splits |
| [`models/`](models/) | Saved classifiers, embeddings, and neural model weights |
| [`notebook/`](notebook/) | Training notebook and validation results |
| [`requirements.txt`](requirements.txt) | Dependencies for running the app |

## Author and Acknowledgments

Developed by Mohamed Tamer. Thanks to Eng. Karim Essam for his guidance and mentorship throughout the project.
