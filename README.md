# Sentiment Analysis on IMDb Reviews

Binary sentiment classification (positive / negative) of IMDb movie reviews using
recurrent neural networks built with **PyTorch** and **TorchText**.

This repository implements and compares three recurrent architectures for the same
task: a plain RNN, a bidirectional RNN, and a bidirectional LSTM.

## Overview

Each model takes a tokenized movie review, embeds the tokens, runs them through a
recurrent network, and predicts a single sentiment score (positive vs. negative).
The accompanying notebooks walk through data loading, vocabulary building with
pre-trained word vectors, training, and evaluation.

## Models / Approach

All three models share the same overall pipeline (embedding → recurrent layer →
dropout → linear output) and the same `forward` interface
`forward(text, text_lengths)`, using packed padded sequences so variable-length
reviews are handled efficiently.

| File | Class | Recurrent layer | Output head |
| ---- | ----- | --------------- | ----------- |
| [`rnn_model.py`](rnn_model.py) | `RNN` | `nn.RNN` | `Linear(hidden_dim, output_dim)` using the last hidden state |
| [`birnn_model.py`](birnn_model.py) | `RNN` | `nn.RNN` (bidirectional) | `Linear(hidden_dim * 2, output_dim)` over concatenated forward/backward hidden states |
| [`bilstm_model.py`](bilstm_model.py) | `LSTM` | `nn.LSTM` (bidirectional) | `Linear(hidden_dim * 2, output_dim)` over concatenated forward/backward hidden states |

Common components, set up in the model `__init__`:

- **Embedding layer** — `nn.Embedding` with a padding index, initialized from
  pre-trained word vectors in the notebooks.
- **Recurrent layer** — `nn.RNN` / `nn.LSTM`, configurable number of layers,
  bidirectionality, and dropout.
- **Dropout** — applied to the embeddings and to the final hidden state.
- **Fully connected layer** — maps the recurrent hidden representation to the
  output dimension.

Sequences are packed with `nn.utils.rnn.pack_padded_sequence` before being fed to
the recurrent layer, which is why the models require `text_lengths` alongside the
text tensor.

## Dataset

- **IMDb** movie review dataset, loaded through TorchText.
- 25,000 training examples and 25,000 test examples.
- Tokenization is performed with the **spaCy** tokenizer.
- A vocabulary of 25,000 most-frequent tokens is built (plus `<unk>` and `<pad>`,
  giving 25,002 entries) and initialized with pre-trained **`wiki.simple`**
  (300-dimensional) word vectors.

## Setup

```bash
# Clone the repository
git clone https://github.com/JayeshSuryavanshi/Sentiment-Analysis-IMDb-Reviews.git
cd Sentiment-Analysis-IMDb-Reviews

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English model used for tokenization
python -m spacy download en_core_web_sm
```

> Note: the notebooks pin `torch==1.7.1` and `torchtext==0.8.1`, which use the
> legacy `torchtext.data` API (`Field`, `BucketIterator`, etc.). Newer TorchText
> releases moved these to `torchtext.legacy` and later removed them, so the pinned
> versions are recommended for reproducing the results.

## How to Run / Train

The end-to-end workflow (data download, vocabulary, training, and evaluation)
lives in the Jupyter notebooks. Open the notebook for the architecture you want
and run the cells top to bottom:

```bash
jupyter notebook Sentiment_Analysis_RNN.ipynb
# or Sentiment_Analysis_biRNN.ipynb / Sentiment_Analysis_biLSTM.ipynb
```

The notebooks were developed in Google Colab (with GPU/CUDA) and will download the
IMDb dataset and the `wiki.simple` vectors automatically on first run.

The `*.py` files contain just the model definitions and can be imported into your
own training script, for example:

```python
from rnn_model import RNN          # plain RNN
# from birnn_model import RNN      # bidirectional RNN
# from bilstm_model import LSTM    # bidirectional LSTM

model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim,
            n_layers, bidirectional, dropout, pad_idx)
```

## Results

Reported from the committed notebook outputs:

| Model | Notebook | Test Loss | Test Accuracy |
| ----- | -------- | --------- | ------------- |
| Plain RNN | `Sentiment_Analysis_RNN.ipynb` | 0.616 | 66.97% |
| Bidirectional RNN | `Sentiment_Analysis_biRNN.ipynb` | — | — |
| Bidirectional LSTM | `Sentiment_Analysis_biLSTM.ipynb` | — | — |

The bidirectional RNN and bidirectional LSTM notebooks do not contain saved
evaluation outputs, so their metrics are not reported here. Re-run those notebooks
to obtain their test results.

## Tech Stack

- Python 3
- PyTorch (`torch`)
- TorchText (`torchtext`)
- spaCy (tokenization)
- Jupyter / Google Colab
- Pre-trained `wiki.simple` word vectors

## Repository Structure

```
Sentiment-Analysis-IMDb-Reviews/
├── rnn_model.py                      # Plain RNN model definition
├── birnn_model.py                    # Bidirectional RNN model definition
├── bilstm_model.py                   # Bidirectional LSTM model definition
├── Sentiment_Analysis_RNN.ipynb      # End-to-end RNN training/eval notebook
├── Sentiment_Analysis_biRNN.ipynb    # End-to-end bidirectional RNN notebook
├── Sentiment_Analysis_biLSTM.ipynb   # End-to-end bidirectional LSTM notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## Acknowledgements

The model architectures, training loop, and notebook walkthroughs are based on
Ben Trevett's excellent
[pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)
tutorial series. This repository adapts that material to the IMDb dataset and
compares the RNN, bidirectional RNN, and bidirectional LSTM variants.
