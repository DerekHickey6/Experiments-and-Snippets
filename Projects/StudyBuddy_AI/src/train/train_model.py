import os
import re
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ai_engine import SimpleTokenizer, TinyGPT

# Loads final dataset
def load_corpus(path):
    """Load the full training text from final_dataset.txt"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Turn text -> (input, target) sequence
class TextDataset(Dataset):
    def __init__(self, text, seq_len=20, min_freq=3):
        """
        text: raw string (entire corpus)
        seq_len: number of tokens per training example
        min_freq: minimum word frequency to keep in vocab
        """
        self.seq_len = seq_len
        MAX_SAMPLES = 300000
        
        # 1. Build tokenizer vocab on the full text
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.build_vocab([text], min_freq=min_freq)
        
        # 2. Tokenize full text into word tokens
        tokens = re.findall(r"\w+", text.lower())
        
        # 3. Convert tokens to ids
        ids = [self.tokenizer.word2idx.get(t, 1) for t in tokens]
        
        self.inputs = []
        self.targets = []
        
        # 4. Build sliding windows for language model
        # called "Next-token predictions"
        for i in range(min(len(ids) - seq_len, MAX_SAMPLES)):
            x = ids[i : i + seq_len]
            y = ids[i + 1 : i + seq_len + 1]
            
            self.inputs.append(x)
            self.targets.append(y)
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y
    
    
# === Training Loop ===
def train_model(corpus_path= "data/processed/final_dataset.txt",
                seq_len=20,
                batch_size=32,
                epochs=3,
                lr=1e-3,
                min_freq=3):
    """Main training loop for TinyTransformer"""
    # 1. Load text
    print(f"Loading corpus from: {corpus_path}")
    text = load_corpus(corpus_path)
    
    # 2. Build dataset from string to tensors
    dataset = TextDataset(text, seq_len=seq_len, min_freq=min_freq)
    vocab_size = len(dataset.tokenizer.word2idx)
    
    print(f"Vocab Size: {vocab_size}")
    print(f"Training samples: {len(dataset)}")
    
    # 3. initialize Dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Create a model and putting it on the target device
    model = TinyGPT(vocab_size=vocab_size, embed_dim=32, n_heads=2, hidden_dim=64, max_seq_len=seq_len)
    device = torch.device("cpu")
    model.to(device)
    
    # 5. create loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 6. Training loop
    model.train()
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(loader):
            # put data to target device
            x = x.to(device)
            y = y.to(device)
            
            # Zero previous gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x)      # (batch, seq, vocab)
            
            # Flatten logits + target for loss calculation
            batch_size_now, seq_len_now, vocab_now = logits.shape
            logits_flat = logits.reshape(-1, vocab_now)    # (batch*seq, vocab)
            y_flat = y.view(-1)
            
            # Compute the loss
            loss = loss_fn(logits_flat, y_flat)
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Show progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg = total_loss / (batch_idx + 1)
                elapsed = time.time() - epoch_start_time
                elapsed_min = int(elapsed / 60)
                elapsed_sec = int(elapsed % 60)
                print(f"Epoch {epoch+1} | Step {batch_idx+1} | Avg Loss: {avg:.4f} | Elapsed Time: {elapsed_min:.0f} min, {elapsed_sec} sec")
        
        epoch_loss = total_loss / max(1, (batch_idx+1))
        total_time = time.time() - total_start_time
        total_time_min = int(total_time / 60)
        total_time_sec = int(total_time % 60)
        print(f"Epoch {epoch+1} finished | Avg Loss: {epoch_loss:.4f} | Total Time: {total_time_min:.0f} min, {total_time_sec} sec")
        
    # 7. Save Model + tokenizer
    os.makedirs("models", exist_ok=True)
    
    # saves model
    model_path = "models/tinyGPT.pt"
    torch.save(model.state_dict(), model_path)
    
    # saves vocabulary
    vocab_path = "models/tokenizer_vocab.pt"
    torch.save(
        {
            "word2idx": dataset.tokenizer.word2idx,
            "idx2word": dataset.tokenizer.idx2word
        },
        vocab_path
    )
    
    print(f"Saved model -> {model_path}")
    print(f"Saved vocabulary -> {vocab_path}")
    
if __name__ == "__main__":
    train_model()
                