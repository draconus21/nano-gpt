import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

batch_size = 32
block_size = 8  # context length
max_iters = 3000

device = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(1337)

# get data
this_dir = Path(__file__).parent
data_f = this_dir / "../data/input.txt"
with open(data_f, "r") as f:
    t = f.read()


chars = list(set(t))  # the"alphabet"
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}  # mapping from letter -> int
itos = {i: ch for i, ch in enumerate(chars)}  #  mapping from int -> letter

encode = lambda s: [stoi[c] for c in s]  # convert string to list of ints
decode = lambda l: "".join([itos[i] for i in l])  # convert list of ints to string

# pret data for training -> train/test split
data = torch.tensor(encode(t), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]


# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)

    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads off the probabities/scores for the next token from this LUT
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of ints
        logits = self.token_embedding_table(idx)  #  (B, T, C)

        loss = None

        # compute loss if target is available
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get current logits
            logits = self(idx)  # (B, T, C)
            # take only the last logits
            logits = logits[:, -1, :]  # (B, C)
            # get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel()
m.to(device)

xb, yb = get_batch("valid")
logits, loss = m(xb, yb)
print(logits.shape, loss)

start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(start, max_new_tokens=100)[0].tolist()))
