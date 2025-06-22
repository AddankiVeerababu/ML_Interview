# ML_Interview
## **Question 1: Suppose that we design a deep architecture to represent a sequence by stacking self-attention layers with positional encoding. What could be issues? (paragraph format)**


When we build a deep model using many self-attention layers, we add positional encoding to help the model understand the order of the input sequence. However, there are several issues that can arise. One main problem is that as the input passes through multiple layers, the model may gradually lose track of position information, especially if we use fixed encodings like sine and cosine. These encodings do not adapt during training, so their impact may weaken in deeper layers, making it harder for the model to preserve the order of tokens.

Another issue is that attention layers, particularly in deeper parts of the model, can begin to focus on the same regions of the sequence. This reduces the diversity of information being learned across layers and limits the model’s ability to capture complex or hierarchical patterns. Additionally, if the positional encoding remains static and does not evolve with the learning process, the model may struggle to understand nuanced relationships in longer or more structured sequences.

To address these limitations, learnable positional encodings are often used. These allow the model to update how it understands position during training, leading to better flexibility and performance. However, learnable encodings introduce their own challenges. If not regularized properly, they can cause overfitting to specific sequence lengths or token positions seen during training, which reduces generalization. Since they are typically defined for a fixed maximum length, they may also struggle to handle variable-length or longer sequences effectively unless special care is taken during design.

Additionally, stacking many layers increases computational cost and memory usage, which can become inefficient when processing long sequences. In very deep networks, there is also a risk of gradient saturation through the positional channels, making it harder for the model to reinforce position-awareness across layers. These challenges highlight the importance of carefully designing how positional encoding is integrated into deep transformer architectures to ensure both order-awareness and training stability.


## **Question 2: Can you design a learnable positional encoding method using pytorch? (Create dummy dataset)**

# Learnable Positional Encoding in PyTorch

This project implements a learnable positional encoding using PyTorch. It includes a dummy dataset of token sequences, applies token embeddings using `nn.Embedding`, and adds learnable positional embeddings to those sequences.

---

## ✅ What the Code Does

- Creates a dataset of random token sequences
- Embeds the token IDs into vectors
- Adds learnable positional encodings to the embedded tokens
- Prints sample outputs to verify the method

---

## 🧱 Code Structure

### `DummySequenceDataset`
A PyTorch dataset that returns random sequences of token IDs.

**Arguments:**
- `vocab_size`: Number of unique tokens
- `seq_len`: Length of each token sequence
- `num_samples`: Total number of sequences

Each sequence is a 1D tensor of shape `[seq_len]`.

---

### `LearnablePositionalEncoding`
A PyTorch module that creates a learnable tensor of shape `[1, max_len, d_model]`, where:
- `max_len` is the maximum sequence length
- `d_model` is the embedding size

It adds this tensor to the input token embeddings to inject positional information.

---

### `run_demo()`
This function:
- Creates a small dummy dataset
- Initializes token embeddings and positional encoding
- Passes one batch through both layers
- Prints:
  - Token IDs
  - Shape of token embeddings
  - Output after adding positional encoding

---

## ▶️ How to Run

Make sure PyTorch is installed:
```bash
pip install torch

