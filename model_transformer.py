import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    This is crucial because the self-attention mechanism itself does not
    process the order of the sequence. [1]
    
    Modified to work with batch_first=True transformers.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Creating positional encoding for batch_first format [1, seq_len, d_model]
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] with batch_first=True
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """A standard Transformer model for language modeling."""
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        d_model = config['embedding_dim']
        nhead = config['num_heads']
        d_hid = config['ff_dim']
        nlayers = config['num_blocks']
        dropout = config['dropout']

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # A standard Transformer Encoder Layer includes self-attention and a feed-forward network.
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
        # The final linear layer to project back to vocabulary size for logits.
        self.decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len], for causal masking.
        """
        # Causal mask ensures that attention is only paid to previous tokens.
        # Note: For TransformerEncoder with batch_first=True, we don't need to permute
        # the input, but we still need to be careful with the mask shape
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Generate mask if not provided - make sure to use the correct sequence length
        # after accounting for batch_first=True
        if src_mask is None:
            seq_len = src.size(1)  # sequence length is the second dimension with batch_first=True
            # Create a causal mask appropriate for self-attention
            device = src.device
            src_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        
        output = self.transformer_encoder(src, src_mask)
        # No need to permute since batch_first=True
        
        logits = self.decoder(output)
        return logits