import torch
import torch.nn as nn
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-learning_rate", type=float, default=1e-3)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-dim_model", type=int, default=512)
parser.add_argument("-num_heads", type=int, default=8)
parser.add_argument("-batch_first", type=int, default=1)
parser.add_argument("-doc_path", type=str, default="assets/source.txt")
parser.add_argument("-memory_length", type=int, default=128)
args = parser.parse_args()


class GPTLayer(nn.Module):
    def __init__(self, dim_model, num_heads, batch_first) -> None:
        super().__init__()
        self.decoder = nn.TransformerDecoderLayer(
            dim_model, num_heads, batch_first=batch_first)

    def forward(self, x, target_mask, key_padding_mask=None):
        x = self.decoder.norm2(
            x+self.decoder._mha_block(x, x, target_mask, key_padding_mask))
        x = self.decoder.norm1(
            x+self.decoder._sa_block(x, target_mask, key_padding_mask))
        x = self.decoder.norm3(x+self.decoder._ff_block(x))
        return x


class GPT(nn.Module):
    def __init__(self, dim_model, n_head, batch_first=True) -> None:
        super().__init__()
        self.layer = GPTLayer(dim_model, n_head, batch_first)

    def forward(self, x, target_mask, key_padding_mask=None):
        return self.layer.forward(x, target_mask, key_padding_mask)


dataset = []
with open(args.doc_path, "r", encoding="utf-8") as f:
    for line in f:
        dataset.extend(list(line))

vocab = list(set(dataset))
dataset_size = len(dataset)
vocab_size = len(vocab)


def tokens_to_indices(tokens):
    indices = []
    for token in tokens:
        indices.append(vocab.index(token))
    return indices


def indices_to_tokens(indices):
    tokens = []
    for i in indices:
        tokens.append(vocab[i])
    return tokens


index_set = tokens_to_indices(dataset)


def vocab_test():
    print(f"tokens: {tokens_to_indices(dataset[:100])}")
    print(f"indices: {indices_to_tokens(tokens_to_indices(dataset[:100]))}")
    print(''.join(indices_to_tokens(tokens_to_indices(dataset[:100]))))
    print(f"num tokens: {len(dataset)} \n vocab:{vocab}")


def sample_batch():
    random_batch = torch.randint(
        0, dataset_size-args.memory_length-1, size=(args.batch_size,)).tolist()
    sources = []
    targets = []
    for start in random_batch:
        source = index_set[start:start+args.memory_length]
        target = index_set[start+1:start+args.memory_length+1]
        sources.append(source)
        targets.append(target)
    return torch.tensor(sources), torch.tensor(targets)


def sample_batch_test():
    source, target = sample_batch()
    for source, target in zip(source, target):
        source_stream = indices_to_tokens(source.tolist())
        target_stream = indices_to_tokens(target.tolist())
        print(
            f"\n=======\nsource:\n{''.join(source_stream)}\n\ntarget:\n{''.join(target_stream)}")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(args.dim_model, args.num_heads)
model.to(DEVICE)
