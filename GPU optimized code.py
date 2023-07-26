# -*- coding: utf-8 -*-
"""sample for Omar GBU

Collaboratory automatically generates them.

The original file is located at
    https://colab.research.google.com/drive/11qTuivZhjGPb4qvcxlKzSb6flJcmtdjh
"""

import torch
from torch import nn

device = "cuda"

# layout:
# 1. Read all embeddings and put them in one tensor
#
# for doing the cosine sim for a matrix upload the embeddings here as tensors 
# I_good = torch.tensor([]) # 2D array (m,n) m = total number of images, n = emb for each img
# J_good = torch.tensor([]) # 2D array (m,n) m = total number of images, n = emb for each img
# # 2 to device

# 3 Do the cos sim
# # GIVEN EXAMPLE 
I = torch.rand((86,128)).to(device)
J = torch.rand((64,128)).to(device)

compare = I @ J.T

len_i1 = torch.norm(I,dim=1, keepdim = True)
len_J = torch.norm(J, dim=1, keepdim=True)
# len_J.shape

total = compare / (len_i1 @ len_J.T)

total.shape
