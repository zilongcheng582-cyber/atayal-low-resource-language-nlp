# Atayal NLP Summer Project

A low-resource NLP pipeline for the Zeyuan dialect of Atayal,
based on aligned text and audio data from Taiwan's K-12 curriculum.

## Project Goals
- Corpus statistical analysis
- Tokenization and preprocessing pipeline
- Character-level LSTM language model (PyTorch)
- (Optional) Audio feature extraction (MFCC)

## Results
- Corpus: 688 sentences, 1,851 unique words, 61 unique characters
- LSTM trained for 10 epochs, final loss: 0.3163
- Model successfully generates Atayal-like text

## Repository Structure
```
notebooks/
├── 01_corpus_analysis.ipynb
├── 02_tokenization.ipynb
└── 03_lstm_language_model.ipynb
data/
└── raw/  ← 12 chapters of Atayal text
```

## Data Source
Aligned text and audio from Taiwan's 十二年國教 Atayal (Zeyuan dialect) curriculum.

## Setup
Clone the repo and open any notebook in Google Colab.
Data is automatically downloaded from this repository.

## Tech Stack
- Python 3
- PyTorch
- Google Colab

