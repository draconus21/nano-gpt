import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

batch_size = 64
block_size = 256  # context length
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4

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

    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.k = nn.Linear(n_embed, head_size, bias=False)
        self.q = nn.Linear(n_embed, head_size, bias=False)
        self.v = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        wei = self.q(x) @ self.k(x).transpose(-2, -1)  # (B, T, head) @ (B, head, T) -> (B, T, T)
        wei = wei * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform attention
        v = self.v(x)  # (B, T, head)
        out = wei @ v  # (B, T, T) @ (B, T, head) -> (B, T, head)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention"""

    def __init__(self, n_head, n_embed, dropout, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size=head_size, n_embed=n_embed, block_size=block_size, dropout=dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple feed forward network"""

    def __init__(self, n_embed, dropout=dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attn) followed by computation (ffwd)"""

    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_head
        assert head_size * n_head == n_embed

        self.sa = MultiHeadAttention(
            n_head=n_head, n_embed=n_embed, dropout=dropout, head_size=head_size, block_size=block_size
        )
        self.ffwd = FeedForward(n_embed)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, n_layer, n_head, n_embed, block_size, dropout):
        super().__init__()
        # each token reads off the probabities/scores for the next token from this LUT
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed=n_embed, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of ints
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  #  (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
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


def infer_to_file(model, epoch, max_new_tokens=500):
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    res = decode(model.generate(start, max_new_tokens=max_new_tokens)[0].tolist())
    result = this_dir / f"../results/result_{epoch}.txt"
    if not Path(result).parent.exists():
        os.makedirs(result.parent)

    with open(result, "w") as f:
        f.write(res)
    return res


model = BigramLanguageModel(n_layer=n_layer, n_head=n_head, n_embed=n_embed, block_size=block_size, dropout=dropout)
model.to(device)


try:
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
            infer_to_file(model, epoch=epoch)
finally:
    print("*" * 25, "final", "*" * 25)
    losses = estimate_loss()
    print(f"{epoch}: {losses}")
    infer_to_file(model, epoch=epoch, max_new_tokens=10000)
