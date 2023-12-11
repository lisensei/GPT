import torch
from torch.nn import Transformer


def tokens_to_indices(tokens, vocab):
    indices = []
    for token in tokens:
        indices.append(vocab.index(token))
    return indices


def indices_to_tokens(indices, vocab):
    tokens = []
    for i in indices:
        tokens.append(vocab[i])
    return tokens


def sample_batch(dataset, memory_length, batch_size):
    random_batch = torch.randint(
        0, len(dataset)-memory_length-1, size=(batch_size,)).tolist()
    sources = []
    targets = []
    for start in random_batch:
        source = dataset[start:start+memory_length]
        target = dataset[start+1:start+memory_length+1]
        sources.append(source)
        targets.append(target)
    return torch.tensor(sources), torch.tensor(targets)


@torch.no_grad()
def validate(model, dataset, loss_fn, memory_length, batch_size, device):
    valid_loss = 0
    dataset_size = len(dataset)
    batch_length = memory_length*batch_size
    num_batches = dataset_size//(batch_length)
    if num_batches == 0 or dataset_size < batch_length+1:
        raise Exception(f"not enough tokens")
    for b in range(num_batches):
        source = dataset[b*batch_length:(b+1)*batch_length]
        target = dataset[b*batch_length+1:(b+1)*batch_length+1]
        source = torch.tensor(source, device=device).reshape(batch_size, -1)
        target = torch.tensor(target, device=device).reshape(batch_size, -1)
        mask = Transformer.generate_square_subsequent_mask(source.size(
            1), device=device).expand(source.size(0)*model.num_heads, -1, -1)
        out = model(source, mask)
        loss = loss_fn(out.permute(0, 2, 1), target)
        valid_loss += loss.cpu().detach().item()
    return valid_loss/num_batches


def vocab_test(dataset, vocab):
    print(f"tokens: {tokens_to_indices(dataset[:100])}")
    print(
        f"indices: {indices_to_tokens(tokens_to_indices(dataset[:100],vocab),vocab)}")
    print(''.join(indices_to_tokens(
        tokens_to_indices(dataset[:100], vocab), vocab)))
    print(f"num tokens: {len(dataset)} \n vocab:{vocab}")


def sample_batch_test():
    source, target = sample_batch()
    for source, target in zip(source, target):
        source_stream = indices_to_tokens(source.tolist())
        target_stream = indices_to_tokens(target.tolist())
        print(
            f"\n=======\nsource:\n{''.join(source_stream)}\n\ntarget:\n{''.join(target_stream)}")
