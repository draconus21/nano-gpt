# get data
data_f = "../data/input.txt"
with open(data_f, "r") as f:
    t = f.read()


chars = list(set(t))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
