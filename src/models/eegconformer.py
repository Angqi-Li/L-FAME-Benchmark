# models/eegconformer.py
"""
Defines the Conformer EEG classification model that combines shallow CNN and Transformer encoder blocks.
"""
import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange
from models.utils.transformer_blocks import TransformerEncoder


class Square(nn.Module):
    def forward(self, x):
        return x ** 2

class SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))
# fixed positional encoder      
class PositionalEncoding(nn.Module):
    """
    Implements fixed (non-learnable) sinusoidal positional encoding.
    """
    # [MODIFIED] Increased default length to 1000 to safely accommodate variable sequence lengths
    def __init__(self, embedding, length=1000, device='cuda'):
        """
        :param embedding: embedding dimension (d_model)
        :param length: maximum sequence length
        :param device: device to store the encoding
        """
        super().__init__()

        # Create positional encoding matrix of shape (length, embedding)
        encoding = torch.zeros(length, embedding, device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / embedding))

        encoding[:, 0::2] = torch.sin(position * div_term)  # even indices
        encoding[:, 1::2] = torch.cos(position * div_term)  # odd indices

        self.encoding = encoding.unsqueeze(0)  # shape: (1, length, embedding)
        self.encoding.requires_grad = False

    def forward(self, x):
        """
        :param x: input tensor of shape (batch_size, seq_len, embedding)
        :return: tensor with positional encoding added
        """
        seq_len = x.size(1)
        # [MODIFIED] Added .to(x.device) to prevent device mismatch errors during multi-GPU training
        return x + self.encoding[:, :seq_len, :].to(x.device)
    

class Conformer(nn.Module):
    def __init__(self,
                number_channel=22, 
                nb_classes=4, 
                dropout_rate=0.5,
                emb_size=40,
                num_heads=8,
                depth=6,
                forward_expansion=4,
                forward_drop_p=0.5,
                attn_drop_p=0.5,
                time_kernel=25): # [MODIFIED] Added time_kernel for fair baseline search
        """
        Parameters:
            number_channel: Number of EEG channels
            nb_classes: Number of output classes
            dropout_rate: Dropout rate
            emb_size: Transformer embedding dimension
            num_heads: Number of attention heads
            depth: Number of Transformer blocks
            time_kernel: Temporal receptive field size
        """
        super().__init__()

        # Shallow CNN block to extract basic EEG spatial-temporal features
        self.shallownet = nn.Sequential(
            # [MODIFIED] Replaced hardcoded 40 with dynamic emb_size, and 25 with time_kernel
            nn.Conv2d(1, emb_size, (1, time_kernel), (1, 1)),                
            nn.Conv2d(emb_size, emb_size, (number_channel, 1), (1, 1)),   
            nn.BatchNorm2d(emb_size),
            Square(),
            nn.AvgPool2d((1, 75), (1, 15)),
            SafeLog(),
        )

        # Projection into token sequence for Transformer input
        self.projection = nn.Sequential(
            # [MODIFIED] Replaced hardcoded 40 with dynamic emb_size
            nn.Conv2d(emb_size, emb_size, (1, 1), stride=(1, 1)),         
            Rearrange('b e (h) (w) -> b (h w) e'),            
        )

        # Positional Encoding to add sequential information
        # [MODIFIED] Sequence length dynamically handled by the class now, passed large buffer
        self.pos_encoding = PositionalEncoding(embedding=emb_size, length=1000)  


        # Transformer Encoder to capture temporal dependencies between segments
        self.trans = TransformerEncoder(
                    num_heads=num_heads,
                    depth=depth, 
                    emb_size=emb_size
                )

        # Final classification head
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            # [MODIFIED] Replaced nn.Linear(2440) with LazyLinear to prevent shape crashes
            nn.LazyLinear(out_features=nb_classes)                      
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Conformer model.
        Input shape: [batch, 1, channels, time]
        Output shape: [batch, nb_classes]
        """
        x = self.shallownet(x)     # -> [B, emb_size, 1, time_reduced]
        x = self.projection(x)     # -> [B, seq_len, emb_size]
        
        # [MODIFIED] Uncommented this line. Transformers without positional encoding are blind to sequence order.
        x = self.pos_encoding(x)   # -> [B, seq_len, emb_size] 
        
        x = self.trans(x)          # -> [B, seq_len, emb_size]
        x = self.flatten(x)        # -> [B, seq_len * emb_size]
        x = self.fc(x)             # -> [B, nb_classes]
        return x