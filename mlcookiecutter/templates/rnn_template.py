import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # LSTM expects inputs in the shape (batch, seq, input_size)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(
            x.device
        )

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Get the last output of the sequence
        out = self.fc(out)
        return out
