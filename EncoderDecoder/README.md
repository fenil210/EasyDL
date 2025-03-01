# Encoder-Decoder Network (English-German Translation)

A complete sequence-to-sequence implementation using the Multi30k dataset.

## Features
- Bahdanau attention mechanism
- BLEU score evaluation
- Gradient clipping
- Teacher forcing
- Proper padding handling
- Real-world translation task

## Usage
1. Install requirements:
```
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

```
2. Run training:
python main.py
```