import torch.nn as nn
import torch

# ========== Basic BiLSTM Model ==========
class BiLSTM(nn.Module):
    def __init__(self, input_size, num_class):
        super().__init__()
        # Bi-directional LSTM layer
        self.lstm = nn.LSTM(
            input_size=201,     # Input feature dimension (e.g., MFCC + others)
            hidden_size=256,
            bidirectional=True,
            batch_first=True
        )
        self.tanh = nn.Tanh()                     # Non-linearity after LSTM
        self.dropout = nn.Dropout(p=0.3)          # Dropout for regularization
        self.fc1 = nn.Linear(512, 256)            # 512 = 256 * 2 (bidirectional)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_class)      # Final classifier

    def forward(self, x):  # x: [B, T, F]
        y, _ = self.lstm(x)            # y: [B, T, 512]
        x = y[:, -1, :]                # Use the last timestep output
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


# ========== Stacked BiLSTM Model ==========
class StackedLSTM(nn.Module):
    def __init__(self, input_size, num_class):
        super().__init__()
        # Two-layer Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=201,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        y, _ = self.lstm(x)            # y: [B, T, 512]
        x = y[:, -1, :]                # Take last timestep
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu1(x)
        return self.fc2(x)


# ========== Temporal Additive Attention Module ==========
class TemporalAdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Attention parameters: learnable linear layers
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

        # Initialization
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.zeros_(self.v.bias)

    def forward(self, lstm_outputs):  # lstm_outputs: [B, T, H]
        lstm_outputs = self.norm(lstm_outputs)
        W_out = self.W(lstm_outputs)                  # [B, T, H]
        scores = self.v(torch.tanh(W_out))            # [B, T, 1]

        # Prevent numerical issues with NaNs or extreme values
        scores = torch.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=-10.0)
        scores = scores.clamp(min=-10.0, max=10.0)

        weights = torch.softmax(scores, dim=1)        # [B, T, 1]
        weights = torch.nan_to_num(weights, nan=1.0 / weights.shape[1])  # Fallback for NaN cases

        # Compute weighted sum of LSTM outputs across time
        context = torch.sum(weights * lstm_outputs, dim=1)  # [B, H]
        return context, weights.squeeze(-1)  # [B, H], [B, T]


# ========== BiLSTM with Additive Attention ==========
class LSTMAdditiveAttention(nn.Module):
    def __init__(self, input_size, num_class):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=201,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        self.attn = TemporalAdditiveAttention(hidden_dim=512)  # 256 * 2

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):  # x: [B, T, F]
        lstm_out, _ = self.lstm(x)                     # [B, T, 512]
        context, attn_weights = self.attn(lstm_out)    # [B, 512], [B, T]
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, attn_weights


# ========== Stacked BiLSTM with Additive Attention ==========
class StackedLSTMAdditiveAttention(nn.Module):
    def __init__(self, input_size, num_class):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=201,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.attn = TemporalAdditiveAttention(hidden_dim=512)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):  # x: [B, T, F]
        lstm_out, _ = self.lstm(x)                     # [B, T, 512]
        context, attn_weights = self.attn(lstm_out)    # [B, 512], [B, T]
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x, attn_weights
