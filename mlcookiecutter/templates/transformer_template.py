import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, num_classes):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, hidden_dim)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, hidden_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Average pooling over sequence
        x = self.fc(x)
        return x
