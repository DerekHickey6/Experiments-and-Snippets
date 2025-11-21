import torch
from torch import nn
import torch.nn.functional as F     # functional versions of operations like: activations, loss functions, etc.
import re       # regular expression module - for splitting text into tokens like words

class SimpleTokenizer:
    def __init__(self):
        
        self.word2idx = {"<PAD>":0, "<UNK>":1}  # word to id mapping, <PAD> for sequencing to same length for batches
        self.idx2word = {0:"<PAD>", 1:"<UNK>"}  # id to work mapping, <UNK> for words that are unknown
    
    # counts how often each word appears
    # adds word that apear at least 'min_freq' times    
    def build_vocab(self, text_list, min_freq=1):
        freq={}                                 # stored as key = word, value = count
        for text in text_list:
            # extracts "word-like" elements (no punctuation) and returns a list to iterate through
            for w in re.findall(r"\w+", text.lower()):
                # Counts how often a word gets repeated
                freq[w] = freq.get(w, 0) + 1
            
        # Create vocab IDs
        for word, count in freq.items():    # loop through word, count pair in freq dict 
            if count >= min_freq:           # filter by frequency
                idx = len(self.word2idx)    # finds next available index
                self.word2idx[word] = idx   # add to word-to-index mapping
                self.idx2word[idx] = word   # add to index-to-word mapping
    
    # === Bridge between text + neural networks ===
    # returns a fixed-length list of integers
    def encode(self, text, max_len=100):
        # Tokenizes String to only word-chars (no - punctuation)
        tokens = re.findall(r"\w+", text.lower())
        # Converts words to id's
        ids = [self.word2idx.get(t, 1) for t in tokens]
        # Sequence padding = Truncates id's + pads with zeros if too shorts
        return ids[:max_len] + [0] * (max_len - len(ids)) 
    
    # Takes list of IDs, looks up corrisponding words and returns spaced out sentence
    def decode(self, ids):
        # returns a string of words (separated by a space)
        return " ".join(self.idx2word.get(i, "<UNK>") for i in ids)
    
# === PyTorch Neural Network Classes ===

class TinyGPT(nn.Module):
    def __init__(self, vocab_size,
                 embed_dim=32,
                 n_heads=2,
                 hidden_dim=64,
                 max_seq_len=100):
        super().__init__()
        
        # Token + positional embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)    # Turns IDs into vectors
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)     # Turns word positions into vectors
        
        # One or more transformer blocks, with causal mask at forward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True  # Keeps format (batch, seq, dim)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2      # can change later
        )
        
        # Final linear head -> logits over vocabulary
        # maps position's hidden vector to a distribution over vocab
        self.fc = nn.Linear(embed_dim, vocab_size)
        
        # Store max length for safety, safeguard to not exceed positional embeddings
        self.max_seq_len = max_seq_len
        
    def _causal_mask(self, seq_len, device):
        # Builds an upper-triangular mask to each position only cares about current/past tokens (not future)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)
        return mask
                
    def forward(self, x):
        """
        x: (batch, seq_len) of token IDs
        returns: logits (batch, seq_len, vocab_size)
        """
        
        batch_size, seq_len = x.shape
        
        # Clamp seq _len to max_seq_len to avoid indexing pos_emb out of range
        if seq_len > self.max_seq_len:
            x = x[:, -self.max_seq_len:]
            seq_len = self.max_seq_len
            
        # Token + Positional embeddings
        tok = self.token_emb(x)     # (batch, seq, emb_dim)
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)     # (1, seq)
        pos = self.pos_emb(positions)                                       # (1, seq, emb_dim)

        h = tok + pos
        
        # Causal mask so token i can only see token <= i
        mask = self._causal_mask(seq_len=seq_len, device=x.device)      # (seq, seq)

        h = self.transformer(h, mask=mask)  # (batch, seq, vocab_size)
        
        logits = self.fc(h)
        return logits

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, # Vocab size for input word ids range and output prediction must be size of vocab_size
                 embed_dim=32, 
                 n_heads=2, 
                 hidden_dim=64): 
        super().__init__()
        
        # initializes a lookup table that maps each word ID to a vector
        # each word has a 64-dimensional vector assigned to it, helps the model learn similar words
        self.embed = nn.Embedding(vocab_size, embed_dim)  ## Word embeddings
        
        # initializes positional embeddings: 1 vector per position in sequence
        self.pos_embed = nn.Embedding(200, embed_dim)
        
        # layer containing: 
        # self-attention - compares 1 word in sentence to every other word, learn word relationships
        # Multi-head attention (n_heads=2) - head 1 learns grammar, head 2 learns semantics 
        # feedforward network (hidden_dim=128) - mixes representations and extracts new feeatures
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=2  # layer 1 - learns word relationship, layer 2 - refines them
        )
        # maps each vector from the transformer into a prediction over the vocabulary
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        # Transforms tensor x into (batch_size, seq_len, embed_size)
        emb = self.embed(x)
        
        # Create a tensor of positions for each token in sequence
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # look up positional embeddings for each position
        pos_embeddings = self.pos_embed(positions)
        
        # Add token embeddings + positional embeddings for final embedding
        # FINAL embedding includes the information about 'which word' + 'where does it go'
        emb = emb + pos_embeddings
        
        # compares each token to all other tokens, learns context & sentence structure - output = (1, seq_len, embed_dim)
        out = self.transformer(emb)
        # Predicts and returns next word
        logits = self.fc(out)
        return logits
    
class LocalChatBot:
    def __init__(self):
        # Initializes word2idx & idx2word dictionarys, encode()/decode() functions
        self.tokenizer = SimpleTokenizer()
        
        # Memory of the model
        self.memory = []
       
        self.model = None
        
    # 
    def generate(self, prompt, max_len=30, k=8, temperature=0.8):
        
        # Build full prompt
        history = " ".join(self.memory)                         # Create sentence of memory contents
        full_prompt = f"{history} User: {prompt} Bot:"          # Given to model to respond to
        
        # Converts input text to list of ints and wraps in a batch dimension
        encoded = torch.tensor([self.tokenizer.encode(full_prompt)], dtype=torch.long)   # outputs shape (1, seq_len)
        
        unk_id = self.tokenizer.word2idx["<UNK>"]
        pad_id = self.tokenizer.word2idx["<PAD>"]
        
        # at each step, predicts a new word and appends it
        for _ in range(max_len):
            # clamp sequence length
            if encoded.shape[1] > self.model.max_seq_len:
                encoded = encoded[:, -self.model.max_seq_len:]
            
            # Forward pass
            logits = self.model(encoded)
            
            # Extract last token's first
            last_logits = logits[0, -1].detach()
            
            # Stops PAD/UNK token
            last_logits[pad_id] = -1e10
            last_logits[unk_id] = -1e10
            
            # repetition penalty
            last_token = encoded[0].tolist()[-10:]
            for prev_id in last_token:
                last_logits[prev_id] *= 0.7 # lower probability of repeating outputs
            
            # Temperature scaling for softer probability distribution
            scaled_logits = last_logits / temperature    
            
            # Convert to probabilities for REAL probability distribution
            probs = F.softmax(scaled_logits, dim=0)
            
            # Top K filtering to restrict samples to best 'k' tokens
            topk_probs, topk_idx = torch.topk(probs, k)
            
            # Normalize top K probabilities
            topk_probs = topk_probs / torch.sum(topk_probs)
            
            # Sample from top-k distribution, chooses a random token, weighted by probability
            next_id = torch.multinomial(topk_probs, 1).item()
            
            # Map back to original vocab index
            next_id = topk_idx[next_id].item()
            
            # STOP condition, if PAD/UNK
            if next_id in (pad_id, unk_id):
                break
            
            # Appends next token + sequence grows by word each iteration
            encoded = torch.cat([encoded, torch.tensor([[next_id]])], dim=1)

            
        # Converts tensor -> Python list
        # Turns multiple token IDs -> human readable sentence
        ids = encoded[0].tolist()
        reply = self.tokenizer.decode(ids)
        
        # Save User message + Bot reply to memory, 
        self.memory.append(f"User: {prompt}")
        self.memory.append(f"Bot: {reply}")
        self.memory = self.memory[-8:]   # saves only 10 most recent messages
        
        return reply

    def load_model(self, model_path="models/tinyGPT.pt", vocab_path="models/tokenizer_vocab.pt"):
        data =torch.load(vocab_path, map_location="cpu")
        self.tokenizer.word2idx = data["word2idx"]
        self.tokenizer.idx2word = data["idx2word"]
        
        vocab_size = len(self.tokenizer.word2idx)
        
        # recreate model with correct vocab size
        self.model = TinyGPT(vocab_size=vocab_size, embed_dim=32, n_heads=2, hidden_dim=64, max_seq_len=20)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        
        print("Loaded trained model + vocab")

