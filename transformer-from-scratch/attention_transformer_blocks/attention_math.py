from streamlit import context
import torch
import math

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:     # Dim is the dimension to apply softmax over, ex. torch.Size([dim0, dim1, dim2])
    """
    Numerically stable softmax.
    x: input tensor
    dim: dimenstion to apply softmax over
    returns: tensor of same shape as x
    """
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]       # max values across the dimension, extracts values only with [0] and retains dims
    num = torch.exp(x - max_vals)                           # element-wise exp - max vals
    den = torch.sum(num, dim=dim, keepdim=True)             # sum of all e^(x-max(x)), keeps dim of max same as input (x)
    output = num/den

    return output

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k, v,: FloatTensor (B, T, C)

    returns:
        out: FloatTensor (B, T, C)
        att: FloatTensor (B, T, T)  # attention weights
    """
    similarity_score = q @ k.transpose(-2, -1)          # Matrix multiplication of Query x Key to get similarity score
    scaled_similarity_score = similarity_score/math.sqrt(q.shape[-1])     # Shape = (Batch, Query, Key)
    att = softmax(scaled_similarity_score, dim=-1)      # converts scores into attention weights by normalizing keys for each query
    out = att @ v                                       # creates a Context Vector for each query token

    return (out, att)

B, T, C = 2, 4, 8
q = torch.randn(B, T, C)
k = torch.randn(B, T, C)
v = torch.randn(B, T, C)

out, att = scaled_dot_product_attention(q, k, v)

print("out.shape:", out.shape)      # expect (2, 4, 8)
print("att.shape:", att.shape)      # expect (2, 4, 4)

    # Each query's attention weights over keys should sum to 1
print("row sums:", att[0].sum(dim=-1))  # expect ~ tensor([1., 1., 1., 1.])