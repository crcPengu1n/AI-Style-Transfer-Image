"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from ps4_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')

# 5 points
def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    content_current_flat = content_current.view(content_current.shape[1], -1)
    content_original_flat = content_original.view(content_original.shape[1], -1)

    # Compute the squared differences between the current and original feature maps
    loss = (content_current_flat - content_original_flat) ** 2

    # Sum over all elements in the feature map and scale by the content weight
    content_loss_value = content_weight * loss.sum()

    return content_loss_value

# 9 points
def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    features = features.view(features.size(0), features.size(1), -1)

    # Compute the Gram matrix: (N, C, C)
    gram = torch.bmm(features, features.transpose(1, 2))

    if normalize:
        # Get the number of elements that contributed to each position in the Gram matrix
        divisor = features.size(1) * features.size(2)  # C * (H*W)
        gram /= divisor

    return gram

# 9 points
def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    loss = 0.0

    # Loop over each layer index, target Gram matrix, and weight
    for i, layer in enumerate(style_layers):
        # Compute the Gram matrix for the current features at this layer
        current_gram = gram_matrix(feats[layer])

        # Compute the loss for this layer
        layer_loss = style_weights[i] * torch.sum((current_gram - style_targets[i]) ** 2)

        # Add this layer's loss to the total style loss
        loss += layer_loss

    return loss

# 8 points
def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    vertical_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    # Compute the difference between adjacent pixel values in the width dimension
    horizontal_diff = img[:, :, :, 1:] - img[:, :, :, :-1]

    # Square the differences and sum over all dimensions except the batch dimension
    loss = (vertical_diff**2).sum() + (horizontal_diff**2).sum()

    # Multiply by the TV weight
    loss = tv_weight * loss

    return loss

# 10 points
def guided_gram_matrix(features, masks, normalize=True):
  """
  Inputs:
    - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
      a batch of N images.
    - masks: PyTorch Tensor of shape (N, R, H, W)
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, R, C, C) giving the
      (optionally normalized) guided Gram matrices for the N input images.
  """
  guided_gram = None
  # Replace "Pass" statement with your code
  # Get dimensions
  N, R, C, H, W = features.size()

  # Expand masks to match feature shape (N, R, H, W) -> (N, R, 1, H, W)
  masks = masks.unsqueeze(2)

  # Apply masks to features element-wise
  guided_features = features * masks

  # Initialize tensor to store Gram matrices
  gram = torch.zeros((N, R, C, C), device=features.device, dtype=features.dtype)

  # Loop through each guidance channel to compute its Gram matrix
  for i in range(R):
      # Flatten the spatial dimensions (H, W) of the guided features
      flat_features = guided_features[:, i].view(N, C, H * W)

      # Compute the Gram matrix for the current masked features (batch operation)
      gram[:, i] = torch.bmm(flat_features, flat_features.transpose(1, 2))

      if normalize:
          # Normalize the Gram matrix by the number of elements contributing to each sum
          gram[:, i] /= (C * H * W)

  return gram

# 9 points
def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    loss = 0.0

    # Loop over each layer index, target Gram matrix, and weight
    for i, layer in enumerate(style_layers):
        # Get the feature maps for the current layer
        current_features = feats[layer]
        
        # Get the content masks for the current layer
        current_masks = content_masks[layer]
        
        # Compute the guided Gram matrix for the current features
        current_gram = guided_gram_matrix(current_features, current_masks)

        # Compute the loss for this layer
        for r in range(current_gram.size(1)):  # Loop through each region
            layer_loss = style_weights[i] * torch.sum((current_gram[:, r] - style_targets[i][:, r]) ** 2)

            # Add this region's loss to the total style loss
            loss += layer_loss

    return loss
