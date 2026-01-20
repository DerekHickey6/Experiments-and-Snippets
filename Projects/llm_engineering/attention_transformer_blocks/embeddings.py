import torch
import torch.nn as nn

def make_demo_batch() -> torch.Tensor:
    """
    Returns:
        idx: LongTensor of shape (B, T)
    """
    output = []
    for i in range(2): # B - Batch size
        temp_list = []
        for j in range(4):  # T - Sequence length
            if i % 2 == 0:
                temp_list.append(1 + j)
            else: temp_list.append(4 - j)
        output.append(temp_list)

    return torch.tensor(output, dtype=torch.long)

def embed_tokens(idx: torch.Tensor, vocab_size: int, embed_dim: int, block_size: int) -> torch.Tensor:
    """
    Args:
        idx (torch.Tensor): (B, T)

    Returns x:
        FloatTensor (B, T, C)
    """
    token_embedding = nn.Embedding(vocab_size, embed_dim)   # Token embedding Table, each row == 1 token ID
    pos_embedding = nn.Embedding(block_size, embed_dim)     # Positional embedding table

    seq_len = idx.shape[1]      # extract seq_len from idx tensor
    positions = torch.tensor(range(seq_len), dtype=torch.long)

    # Look up embeddings + replaces integer values with learned vector, adding dimension
    tok_emb = token_embedding(idx)
    pos_emb = pos_embedding(positions)

    print(tok_emb.shape)
    print(pos_emb.shape)
    x = tok_emb + pos_emb   # creates a position-aware token representation 

    return x



# tensor([[1, 2, 3, 4],
#         [4, 3, 2, 1]])
idx = make_demo_batch()
x = embed_tokens(idx, 10, 8, 8)
print(x.shape)

