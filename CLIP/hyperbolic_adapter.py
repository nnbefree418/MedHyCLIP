"""
Hyperbolic Adapter Modules for Hyper-MVFA

This module implements hyperbolic geometric layers on the Poincaré ball model.
Key components:
- MobiusLinear: Linear transformation in hyperbolic space
- HyperbolicAdapter: Residual adapter with hyperbolic operations
- Utility functions for hyperbolic projections and distances

Reference: Hyperbolic Neural Networks (HNN++)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt


class MobiusLinear(nn.Module):
    """
    Linear layer in hyperbolic space (Poincaré ball).
    
    Performs: logmap0 -> linear transform -> expmap0
    This keeps computations stable by working in the tangent space.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        c: Curvature of the Poincaré ball (default: 0.1)
        bias: Whether to use bias term (default: True)
    """
    
    def __init__(self, in_features, out_features, c=0.1, bias=True):
        super(MobiusLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.ball = geoopt.PoincareBall(c=c)
        
        # Weight and bias are stored in Euclidean space
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with Xavier uniform for stability."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """
        Forward pass through Möbius linear layer.
        
        Args:
            x: Input tensor in hyperbolic space, shape [N, in_features]
        
        Returns:
            Output tensor in hyperbolic space, shape [N, out_features]
        """
        # Map to tangent space at origin
        x_tan = self.ball.logmap0(x)  # [N, in_features]
        
        # Apply linear transformation in tangent space
        # F.linear computes x_tan @ weight.T (+ bias if present)
        if self.bias is not None:
            res = F.linear(x_tan, self.weight, self.bias)
        else:
            res = F.linear(x_tan, self.weight)
        
        # Map back to hyperbolic space
        res_hyp = self.ball.expmap0(res)  # [N, out_features]
        
        return res_hyp


class HyperbolicAdapter(nn.Module):
    """
    Hyperbolic adapter for CLIP features (without internal residual).
    
    Architecture:
        Input (hyperbolic) -> MobiusLinear -> HyperReLU -> MobiusLinear -> Output
    
    Note: Residual connection is handled externally in CLIP_Inplanted,
          similar to the original ClipAdapter design.
    
    Returns two values (like ClipAdapter):
        1. Bottleneck features (after layer1 + activation)
        2. Output features (after layer2, no residual)
    
    Args:
        input_dim: Input feature dimension (default: 1024 for CLIP ViT-L)
        bottle_dim: Bottleneck dimension (default: 768)
        c: Curvature of Poincaré ball (default: 0.1)
    """
    
    def __init__(self, input_dim=1024, bottle_dim=768, c=0.1):
        super(HyperbolicAdapter, self).__init__()
        self.input_dim = input_dim
        self.bottle_dim = bottle_dim
        self.c = c
        self.ball = geoopt.PoincareBall(c=c)
        
        # Two-layer MLP in hyperbolic space
        self.layer1 = MobiusLinear(input_dim, bottle_dim, c=c)
        self.layer2 = MobiusLinear(bottle_dim, input_dim, c=c)
    
    def forward(self, x):
        """
        Forward pass through hyperbolic adapter.
        
        Args:
            x: Input tensor in hyperbolic space
               Shape: [L+1, B, D] (transformer format)
        
        Returns:
            Tuple of (bottleneck_features, output_features):
                - bottleneck_features: Shape [L+1, B, bottle_dim]
                - output_features: Shape [L+1, B, input_dim]
        """
        # First Möbius linear layer
        h = self.layer1(x)  # [L+1, B, bottle_dim]
        
        # Hyperbolic ReLU: logmap -> ReLU -> expmap
        h_tan = self.ball.logmap0(h)
        h_tan = F.relu(h_tan)
        h_bottleneck = self.ball.expmap0(h_tan)
        
        # Second Möbius linear layer
        h_out = self.layer2(h_bottleneck)  # [L+1, B, input_dim]
        
        # Return both bottleneck features and output features
        # (similar to ClipAdapter's return signature)
        return h_bottleneck, h_out


def project_to_hyperbolic(x, ball):
    """
    Project Euclidean features to hyperbolic space (Poincaré ball).
    
    Args:
        x: Euclidean tensor, shape [..., D]
        ball: geoopt.PoincareBall object
    
    Returns:
        Hyperbolic tensor with same shape
    """
    return ball.expmap0(x)


def compute_hyperbolic_distance_batch(x, y, ball):
    """
    Compute pairwise hyperbolic distances in a vectorized manner.
    
    This function computes distances between all pairs of points in x and y
    without using Python for loops.
    
    Args:
        x: Tensor of hyperbolic points, shape [N, D]
        y: Tensor of hyperbolic points, shape [M, D]
        ball: geoopt.PoincareBall object
    
    Returns:
        Distance matrix, shape [N, M]
        dist[i, j] = hyperbolic_distance(x[i], y[j])
    """
    N, D = x.shape
    M, _ = y.shape
    
    # Expand dimensions for broadcasting
    # x: [N, D] -> [N, 1, D] -> [N, M, D]
    # y: [M, D] -> [1, M, D] -> [N, M, D]
    x_expand = x.unsqueeze(1).expand(N, M, D)
    y_expand = y.unsqueeze(0).expand(N, M, D)
    
    # Flatten to [N*M, D] for batch distance computation
    x_flat = x_expand.reshape(N * M, D)
    y_flat = y_expand.reshape(N * M, D)
    
    # Compute distances using geoopt's dist function (vectorized)
    distances = ball.dist(x_flat, y_flat)  # [N*M]
    
    # Reshape back to [N, M]
    distances = distances.reshape(N, M)
    
    return distances

