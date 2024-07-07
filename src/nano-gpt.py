import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

batch_size = 32
block_size = 8  # context length
max_iters = 5000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"

splits = ["train", "valid"]

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
    if split not in splits:
        raise ValueError(f"Split must be in {splits}. Got {split}")
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Implements a single head"""

    def __init__(self, head_size=16, n_embed=32, block_size=8):
        super().__init__()
        self.k = nn.Linear(n_embed, head_size, bias=False)
        self.q = nn.Linear(n_embed, head_size, bias=False)
        self.v = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        wei = self.q(x) @ self.k(x).transpose(-2, -1)  # (B, T, head) @ (B, head, T) -> (B, T, T)
        wei = wei * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # perform attention
        v = self.v(x)  # (B, T, head)
        out = wei @ v  # (B, T, T) @ (B, T, head) -> (B, T, head)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention"""

    def __init__(self, n_heads, **head_kwargs):
        super().__init__()
        self.heads = nn.ModuleList([Head(**head_kwargs) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class FeedForward(nn.Module):
    """a simple feed forward network"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):
    def __init__(self, n_embed=32, block_size=8):
        super().__init__()
        # each token reads off the probabities/scores for the next token from this LUT
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(n_heads=4, head_size=n_embed // 4, n_embed=n_embed, block_size=block_size)
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of ints
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  #  (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)  # apply one head of self-attention (B, T, n_head*head)
        x = self.ffwd(x)  # (B, T, n_head*head)
        logits = self.lm_head(x)  # (B, T, vocab_size)

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
            # crop idx to the last block_size tokens
            logits, _ = self(idx[:, -block_size:])  # (B, T, C)
            # take only the last logits
            logits = logits[:, -1, :]  # (B, C)
            # get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for epoch in range(max_iters):

    # get btach of data
    xb, yb = get_batch("train")

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"{epoch}: {losses}")

print("*" * 25, "final", "*" * 25)
losses = estimate_loss()
print(f"{epoch}: {losses}")
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, max_new_tokens=500)[0].tolist()))
print("*" * 25)
