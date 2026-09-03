def build_model(vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 512):
    """Build lazily so importing preprocessing does not require PyTorch."""
    import torch.nn as nn

    class Captioner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.output = nn.Linear(hidden_dim, vocab_size)

        def forward(self, image_embedding, captions):
            tokens = self.embedding(captions)
            sequence = nn.functional.pad(tokens, (0, 0, 1, 0))
            sequence[:, 0] = image_embedding
            hidden, _ = self.lstm(sequence)
            return self.output(hidden)

    return Captioner()

