"""
Models for position prediction from inductance time series.

CNN:
  PositionCNN:    feedforward — window of inductance → (x, y, z)
  PositionCNN_AR: autoregressive — window of (inductance + prev positions) → (x, y, z)

LSTM:
  PositionLSTM:    feedforward or autoregressive — configurable input_size

CNN-LSTM:
  PositionCNN_LSTM: Conv1d feature extraction → LSTM → FC head

Transformer:
  PositionTransformer: feedforward or autoregressive — configurable input_size
"""

import math
import torch
import torch.nn as nn


class PositionCNN(nn.Module):
    """
    Conv1d blocks → GlobalAvgPool → FC head → (x, y, z)

    No MaxPool between conv layers (sequence is short, 15-40 steps).
    GAP makes the model agnostic to window size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: list = None,
        kernel_sizes: list = None,
        fc_dims: list = None,
        dropout: float = 0.2,
        output_dim: int = 3,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        if fc_dims is None:
            fc_dims = [128, 64]

        # Conv blocks
        layers = []
        ch_in = in_channels
        for ch_out, ks in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # FC head
        fc_layers = []
        dim_in = conv_channels[-1]
        for dim_out in fc_dims:
            fc_layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            dim_in = dim_out
        fc_layers.append(nn.Linear(dim_in, output_dim))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, in_channels)
        Returns:
            (batch, output_dim)
        """
        x = x.permute(0, 2, 1)  # -> (batch, in_channels, window_size)
        x = self.conv(x)         # -> (batch, conv_channels[-1], window_size)
        x = self.gap(x)          # -> (batch, conv_channels[-1], 1)
        x = x.squeeze(-1)        # -> (batch, conv_channels[-1])
        return self.head(x)      # -> (batch, output_dim)


class PositionCNN_AR(nn.Module):
    """
    Autoregressive 1D CNN for position prediction.

    Input: (batch, window_size, 1+output_dim) — [inductance, prev_positions...]
    Output: (batch, output_dim) — predicted position

    During training with teacher forcing, previous positions are ground truth.
    During inference, previous positions are the model's own predictions.
    """

    def __init__(
        self,
        conv_channels: list = None,
        kernel_sizes: list = None,
        fc_dims: list = None,
        dropout: float = 0.2,
        output_dim: int = 3,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        if fc_dims is None:
            fc_dims = [128, 64]

        in_channels = 1 + output_dim  # inductance + position dims

        # Conv blocks
        layers = []
        ch_in = in_channels
        for ch_out, ks in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool1d(1)

        fc_layers = []
        dim_in = conv_channels[-1]
        for dim_out in fc_dims:
            fc_layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            dim_in = dim_out
        fc_layers.append(nn.Linear(dim_in, output_dim))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, 4) — [inductance, prev_x, prev_y, prev_z]
        Returns:
            (batch, 3)
        """
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        return self.head(x)


class PositionLSTM(nn.Module):
    """
    LSTM → FC head → (x, y, z)

    Uses the last hidden state of the LSTM as the sequence representation.
    Works for both feedforward (input_size=1) and AR (input_size=4).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        fc_dims: list = None,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_dim: int = 3,
    ):
        super().__init__()
        if fc_dims is None:
            fc_dims = [128, 64]

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * (2 if bidirectional else 1)

        fc_layers = []
        dim_in = lstm_out_size
        for dim_out in fc_dims:
            fc_layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            dim_in = dim_out
        fc_layers.append(nn.Linear(dim_in, output_dim))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, input_size)
        Returns:
            (batch, output_dim)
        """
        # output: (batch, seq_len, hidden_size * num_directions)
        output, _ = self.lstm(x)
        # Take the last timestep
        last = output[:, -1, :]
        return self.head(last)


class PositionCNN_LSTM(nn.Module):
    """
    Conv1d feature extraction → LSTM → FC head → (x, y, z)

    The CNN extracts local features at each timestep (preserving sequence length),
    then the LSTM models temporal dependencies across the enriched sequence.
    Uses the last LSTM hidden state → FC head for output.

    Works for both feedforward (input_size=1) and AR (input_size=1+output_dim).
    """

    def __init__(
        self,
        input_size: int = 1,
        conv_channels: list = None,
        kernel_sizes: list = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        fc_dims: list = None,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_dim: int = 3,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        if fc_dims is None:
            fc_dims = [128, 64]

        # Conv1d blocks (same padding preserves sequence length)
        layers = []
        ch_in = input_size
        for ch_out, ks in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)

        # LSTM on top of CNN features
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * (2 if bidirectional else 1)

        # FC head
        fc_layers = []
        dim_in = lstm_out_size
        for dim_out in fc_dims:
            fc_layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            dim_in = dim_out
        fc_layers.append(nn.Linear(dim_in, output_dim))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, input_size)
        Returns:
            (batch, output_dim)
        """
        # CNN feature extraction
        x = x.permute(0, 2, 1)  # -> (batch, input_size, window_size)
        x = self.conv(x)         # -> (batch, conv_channels[-1], window_size)
        x = x.permute(0, 2, 1)  # -> (batch, window_size, conv_channels[-1])

        # LSTM sequence modeling
        output, _ = self.lstm(x)  # -> (batch, window_size, hidden_size)
        last = output[:, -1, :]   # last timestep

        return self.head(last)


class PositionTransformer(nn.Module):
    """
    Transformer encoder → FC head → (x, y, z)

    Projects input features to d_model, adds sinusoidal positional encoding,
    runs through TransformerEncoder layers, takes the last token → FC head.
    Works for both feedforward (input_size=1) and AR (input_size=1+output_dim).
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        fc_dims: list = None,
        dropout: float = 0.1,
        max_len: int = 500,
        output_dim: int = 3,
    ):
        super().__init__()
        if fc_dims is None:
            fc_dims = [128, 64]

        self.d_model = d_model

        # Project input features to d_model
        self.input_proj = nn.Linear(input_size, d_model)

        # Sinusoidal positional encoding (fixed, not learned)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # FC head
        fc_layers = []
        dim_in = d_model
        for dim_out in fc_dims:
            fc_layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            dim_in = dim_out
        fc_layers.append(nn.Linear(dim_in, output_dim))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, input_size)
        Returns:
            (batch, output_dim)
        """
        seq_len = x.size(1)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]
        x = self.transformer(x)  # (batch, seq_len, d_model)
        last = x[:, -1, :]  # last token
        return self.head(last)


class CausalPositionTransformer(nn.Module):
    """
    MLP Encoder → Causal Transformer → MLP Decoder → (x, y)

    Uses a causal (upper-triangular) attention mask so each token can only
    attend to itself and earlier tokens. Suitable for AR prediction where
    future inductance values should not be visible.
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        encoder_dims: list = None,
        decoder_dims: list = None,
        dropout: float = 0.1,
        max_len: int = 500,
        output_dim: int = 2,
    ):
        super().__init__()
        if encoder_dims is None:
            encoder_dims = [64]
        if decoder_dims is None:
            decoder_dims = [128, 64]

        self.d_model = d_model

        # MLP encoder: input features → d_model
        enc_layers = []
        dim_in = input_size
        for dim_out in encoder_dims:
            enc_layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            dim_in = dim_out
        enc_layers.append(nn.Linear(dim_in, d_model))
        self.encoder = nn.Sequential(*enc_layers)

        # Sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        # Causal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP decoder: d_model → output_dim
        dec_layers = []
        dim_in = d_model
        for dim_out in decoder_dims:
            dec_layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            dim_in = dim_out
        dec_layers.append(nn.Linear(dim_in, output_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, input_size)
        Returns:
            (batch, output_dim)
        """
        seq_len = x.size(1)
        x = self.encoder(x)  # (batch, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]

        # Causal mask: each position can only attend to <= its index
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)

        last = x[:, -1, :]  # last token
        return self.decoder(last)
