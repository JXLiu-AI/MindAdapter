import torch
import torch.nn as nn

class NonLinearAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=4096, dropout=0.5):
        super().__init__()
        # A simple MLP with residual connection logic in the forward pass
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, input_dim) 
        )
        
        # Initialize the last layer to zeros to ensure the training starts 
        # with the identity transformation (preserving the linear alignment)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

class RefinedAligner(nn.Module):
    def __init__(self, frozen_btm, adapter):
        super().__init__()
        self.btm = frozen_btm
        self.adapter = adapter
        
        # Freeze the linear BTM
        for param in self.btm.parameters():
            param.requires_grad = False
        self.btm.eval()

    def forward(self, x):
        # 1. Linear Base Alignment (Frozen)
        with torch.no_grad():
            coarse_aligned = self.btm(x)
        
        # 2. Non-linear Residual Refinement (Trainable)
        # We apply the refinement in the target space
        fine_tuning = self.adapter(coarse_aligned)
        
        return coarse_aligned + fine_tuning

class BTM(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(BTM, self).__init__()
        self.input_proj = nn.Linear(dim1, dim2, bias=False)
        self.output_proj = nn.Linear(dim2, dim3, bias=False)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.output_proj(x)
        return x

