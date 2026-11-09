"""Inference matching the preprocessing and model architectures in the assignment notebook."""
from pathlib import Path
import joblib
import numpy as np

MODEL_NAMES = [
    'N-gram + Logistic Regression',
    'Word2Vec + Classifier',
    'GloVe + Classifier',
    'LSTM + Combined Features',
    'LSTM + Embedding Layer',
]
MODEL_FILES = {
    MODEL_NAMES[0]: ('ngram_model.joblib', 'ngram_vectorizer.joblib'),
    MODEL_NAMES[1]: ('word2vec_classifier.joblib', 'word2vec.model'),
    MODEL_NAMES[2]: ('glove_classifier.joblib', 'glove_vectors.kv'),
    MODEL_NAMES[3]: ('feature_lstm.pth', 'metadata.joblib'),
    MODEL_NAMES[4]: ('embedding_lstm.pth', 'metadata.joblib'),
}


def tokenize(text):
    # Demo tokenizer for free-form user text. The training corpus is already tokenized.
    import re
    return re.findall(r"\w+(?:['’\-]\w+)*|[^\w\s]", text, flags=re.UNICODE)


def word_features(words, index):
    word = words[index]
    previous = words[index - 1] if index > 0 else '<START>'
    following = words[index + 1] if index + 1 < len(words) else '<END>'
    return {
        'word': word.lower(), 'previous': previous.lower(), 'next': following.lower(),
        'previous+word': previous.lower() + '|' + word.lower(),
        'word+next': word.lower() + '|' + following.lower(),
        'prefix': word[:3].lower(), 'suffix': word[-3:].lower(),
        'is_title': word.istitle(), 'is_upper': word.isupper(), 'is_digit': word.isdigit(),
    }


def embedding_rows(words, embedding):
    rows = []
    for i in range(len(words)):
        context = [words[j] if 0 <= j < len(words) else '' for j in (i-1, i, i+1)]
        rows.append(np.concatenate([
            np.asarray(embedding[w.lower()], dtype=np.float32)
            if w.lower() in embedding else np.zeros(50, dtype=np.float32)
            for w in context
        ]))
    return np.asarray(rows, dtype=np.float32)


def make_lstm(pretrained, metadata, state):
    import torch
    from torch import nn
    vocab = metadata['word_to_id']
    tags = metadata['tag_names']

    class LSTMTagger(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_pretrained = pretrained
            if pretrained:
                self.w2v = nn.Embedding(len(vocab), 50, padding_idx=0)
                self.glove = nn.Embedding(len(vocab), 50, padding_idx=0)
                input_size = 150
            else:
                self.embedding = nn.Embedding(len(vocab), 100, padding_idx=0)
                input_size = 100
            self.lstm = nn.LSTM(input_size, 64, batch_first=True, bidirectional=True)
            self.output = nn.Linear(128, len(tags))

        def forward(self, words):
            if self.use_pretrained:
                w2v = self.w2v(words)
                glove_vec = self.glove(words)
                left = torch.cat([torch.zeros_like(w2v[:, :1]), w2v[:, :-1]], dim=1)
                right = torch.cat([w2v[:, 1:], torch.zeros_like(w2v[:, :1])], dim=1)
                context = (left + w2v + right) / 3
                x = torch.cat([w2v, glove_vec, context], dim=-1)
            else:
                x = self.embedding(words)
            features, _ = self.lstm(x)
            return self.output(features)

    model = LSTMTagger()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_model(name, models_dir):
    import torch
    models_dir = Path(models_dir)
    required = MODEL_FILES[name]
    missing = [p for p in required if not (models_dir / p).is_file()]
    if missing:
        raise FileNotFoundError('Missing model files: ' + ', '.join(missing))

    if name == MODEL_NAMES[0]:
        return joblib.load(models_dir / required[0]), joblib.load(models_dir / required[1])
    if name == MODEL_NAMES[1]:
        from gensim.models import Word2Vec
        return joblib.load(models_dir / required[0]), Word2Vec.load(str(models_dir / required[1])).wv
    if name == MODEL_NAMES[2]:
        from gensim.models import KeyedVectors
        return joblib.load(models_dir / required[0]), KeyedVectors.load(str(models_dir / required[1]), mmap='r')

    metadata = joblib.load(models_dir / 'metadata.joblib')
    # weights_only avoids unpickling arbitrary Python objects in the .pth file.
    state = torch.load(models_dir / required[0], map_location='cpu', weights_only=True)
    model = make_lstm(name == MODEL_NAMES[3], metadata, state)
    return model, metadata


def predict(text, name, models_dir, loaded=None):
    words = tokenize(text)
    if not words:
        return [], []
    model, feature_data = loaded if loaded is not None else load_model(name, models_dir)
    if name == MODEL_NAMES[0]:
        features = [word_features(words, i) for i in range(len(words))]
        tags = model.predict(feature_data.transform(features)).tolist()
    elif name in MODEL_NAMES[1:3]:
        tags = model.predict(embedding_rows(words, feature_data)).tolist()
    else:
        import torch
        ids = [feature_data['word_to_id'].get(w.lower(), 1) for w in words]
        with torch.no_grad():
            logits = model(torch.tensor([ids], dtype=torch.long))
        label_ids = logits.argmax(dim=-1)[0].tolist()
        tags = [feature_data['id_to_tag'][i] for i in label_ids]
    return words, tags


def extract_entities(words, tags):
    # Reconstruct BIO spans; lone I- tags begin a new span for robust UI display.
    spans = []
    start, kind = None, None
    def flush(end):
        if start is not None:
            spans.append({'Entity': ' '.join(words[start:end]), 'Type': kind, 'Start': start, 'End': end})
    for index, tag in enumerate(tags):
        if tag == 'O' or '-' not in tag:
            flush(index)
            start, kind = None, None
            continue
        prefix, label = tag.split('-', 1)
        if prefix == 'B' or kind != label or start is None:
            flush(index)
            start, kind = index, label
    flush(len(words))
    return spans
