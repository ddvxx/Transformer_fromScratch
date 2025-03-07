# Transformer from Scratch

This repository contains the implementation of a Transformer model from scratch, following the original architecture explained in the paper "Attention Is All You Need".

## 📌 Transformer Architecture

The Transformer is a model based on attention mechanisms and has revolutionized natural language processing (NLP). Its architecture consists of the following key modules:

### 🔹 Embeddings
Converts words into fixed-dimension vectors. A positional encoding is also applied to retain the order of words in the sequence.

### 🔹 Multi-Head Attention Mechanism
Allows the model to focus on different parts of the input sequence simultaneously. It is computed as:

\[ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V \]

Where:
- \( Q \) (queries), \( K \) (keys), and \( V \) (values) are derived from the input transformed by learnable weights.

### 🔹 Layer Normalization and Feed-Forward
Each Transformer block contains a fully connected network with nonlinear activation functions and normalization layers.

### 🔹 Complete Architecture
The Transformer consists of:
- **Encoder:** Processes the input sequence and transforms it into a contextual representation.
- **Decoder:** Generates the output sequentially, using the encoder's representation and additional attention mechanisms.

## 🚀 Repository Contents

- `model.py`: Implementation of the Transformer model from scratch.
- `train.py`: Script for training the model on a dataset.
- `inference.py`: Code for performing inference with a trained model.
- `config.py`: It contains the config for the model to train.

## 📚 Usage
### Train the model
You have to change the config.py file to tweak the hyperparameters. The model available in the repository is trained with the HuggingFace [opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-es?views%5B%5D=en_es), english to spanish dataset.
```bash
python train.py
```

### Generate text with the trained model
```bash
python inference.py
```

## RESULTS
In the epoch 21, the model is capable of the next translations:
![image](https://github.com/user-attachments/assets/615bd154-c891-456f-a155-4010e219273c)

