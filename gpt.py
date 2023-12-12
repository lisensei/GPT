import torch
import torch.nn as nn
import sys
import logging
import os
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import *
from tqdm import tqdm
parser = ArgumentParser()
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-learning_rate", type=float, default=5e-4)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-dim_model", type=int, default=256)
parser.add_argument("-num_heads", type=int, default=8)
parser.add_argument("-batch_first", type=int, default=1)
parser.add_argument("-memory_length", type=int, default=128)
parser.add_argument("-decoder_backbone", type=int, default=0)
parser.add_argument("-num_layers", type=int, default=4)
parser.add_argument("-doc_path", type=str, default="assets/source.txt")
parser.add_argument("-log_root", type=str, default="runs")
parser.add_argument("-print_frequency", type=int, default=10)
parser.add_argument("-training_percentage", type=float, default=0.95)
parser.add_argument("-manual_seed", type=int, default=424242)
parser.add_argument("-topk", type=int, default=3)
args = parser.parse_args()


class GPTLayer(nn.Module):
    def __init__(self, dim_model, num_heads, batch_first, decoder_backbone) -> None:
        super().__init__()
        '''
        Both torch's transformer encoder and decoer layers can be used as a GPT layer
        Encoder layer can directly be used since it has no cross attention layer and also accepts attention mask.
        Using torch's decoder layer as GPT layer requires modification to the cross attention layer
        '''
        self.decoder_backbone = decoder_backbone
        if decoder_backbone:
            self.layer = nn.TransformerDecoderLayer(
                dim_model, num_heads, batch_first=batch_first)
        else:
            self.layer = nn.TransformerEncoderLayer(
                dim_model, num_heads, batch_first=batch_first)

    def forward(self, x, target_mask, key_padding_mask=None):
        #swap the order of self attention and cross attention when forwarding
        if self.decoder_backbone:
            x = self.layer.norm2(
                x+self.layer._mha_block(x, x, target_mask, key_padding_mask))
            x = self.layer.norm1(
                x+self.layer._sa_block(x, target_mask, key_padding_mask))
            x = self.layer.norm3(x+self.layer._ff_block(x))
        else:
            return self.layer(x, target_mask, key_padding_mask)
        return x


class GPT(nn.Module):
    def __init__(self, dim_model, num_heads, num_layers, num_embedings, memory_length, decoder_backbone=0, batch_first=True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim_model
        self.embeding = nn.Embedding(num_embedings, dim_model)
        self.position_embedding = nn.Embedding(memory_length, dim_model)
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(f"layer {i}", GPTLayer(
                dim_model, num_heads, batch_first, decoder_backbone))
        # self.layer1 = GPTLayer(dim_model, num_heads, batch_first)
        self.fc = nn.Linear(dim_model, num_embedings)

    def forward(self, x, target_mask, key_padding_mask=None):
        token_embeddings = self.embeding(x)
        position_embeddings = self.position_embedding(
            torch.arange(x.size(1), device=x.device)).expand_as(token_embeddings)
        x = token_embeddings+position_embeddings
        for layer in self.layers:
            x = layer.forward(
                x, target_mask, key_padding_mask)
        # out = self.layer1(out, target_mask, key_padding_mask)
        out = self.fc(x)
        return out

    @torch.no_grad()
    def generate(self, target, device, max_length=1000, topk=3):
        self.eval()
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, device=device).reshape(1, -1)
        while target.numel() < max_length:
            if target.numel() > args.memory_length:
                x = target[:, -args.memory_length:]
            else:
                x = target
            target_mask = nn.Transformer.generate_square_subsequent_mask(
                x.size(1)).expand(x.size(0)*args.num_heads, -1, -1).to(device)
            out = self.forward(x, target_mask)
            pred = torch.softmax(out, dim=2).squeeze(0)[-1:, :]
            dist, indices = torch.topk(pred, topk)
            outcome = torch.multinomial(dist, 1).item()
            index = indices[:, outcome:outcome+1]
            target = torch.cat([target, index], dim=1)
        return target


torch.manual_seed(args.manual_seed)
dataset = []
with open(args.doc_path, "r", encoding="utf-8") as f:
    for line in f:
        dataset.extend(list(line))

vocab = list(set(dataset))
dataset_size = len(dataset)
vocab_size = len(vocab)
index_set = tokens_to_indices(dataset, vocab)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(dim_model=args.dim_model, num_heads=args.num_heads,
            num_layers=args.num_layers, num_embedings=vocab_size,
            memory_length=args.memory_length, decoder_backbone=args.decoder_backbone,
            batch_first=args.batch_first)
model.to(DEVICE)


loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)
num_parameters = sum([param.numel() for param in model.parameters()])
num_batches = dataset_size//args.batch_size

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
runt_date = datetime.now().isoformat(timespec="seconds").replace(
    ":", "-" if sys.platform[:3] == "win" else ":")
run_name = f"run-{runt_date}"
writer = SummaryWriter(log_dir=f"{args.log_root}/{run_name}")
log_file_path = f"{args.log_root}/{run_name}/log.txt"
if not os.access(log_file_path, os.F_OK):
    with open(log_file_path, "w") as f:
        pass
file_handler = logging.FileHandler(log_file_path)
logger.addHandler(file_handler)


print_interval = num_batches//args.print_frequency if num_batches//args.print_frequency != 0 else 1
num_training_samples = int(len(index_set)*args.training_percentage)
training_set, validation_set = index_set[:
                                         num_training_samples], index_set[num_training_samples:]
for k, v in args.__dict__.items():
    logger.info(f"{k} : {v}")
logger.info(
    f"number of parameters {num_parameters}\nnumber of batches: {num_batches}")
logger.info(
    f"training set size: {len(training_set)}; validation set size: {len(validation_set)}")
for e in tqdm(range(args.epochs)):
    epoch_loss = 0
    model.train()
    for i in range(num_batches):
        source, target = sample_batch(
            index_set, args.memory_length, args.batch_size)
        source = source.to(DEVICE)
        target = target.to(DEVICE)
        target_mask = nn.Transformer.generate_square_subsequent_mask(
            source.size(1), DEVICE).expand(source.size(0)*args.num_heads, -1, -1)
        out = model(source, target_mask)
        loss = loss_function(out.permute(0, 2, 1), target)
        epoch_loss += loss.cpu().detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % print_interval == 0:
            logger.info(f"epoch:{e} iteration: {i}")
    normalized_training_loss = epoch_loss/num_batches
    normalized_valid_loss = validate(model, validation_set, loss_function,
                                     args.memory_length, args.batch_size, DEVICE)

    writer.add_scalar("training loss", normalized_training_loss, e)
    writer.add_scalar("validation loss", normalized_valid_loss, e)
    indices = model.generate(tokens_to_indices(
        list("I need another story"), vocab), DEVICE)
    logger.info(f"epoch: {e}; generation:\n")
    logger.info(''.join(indices_to_tokens(
        indices.reshape(-1).cpu().tolist(), vocab)))
    torch.save(model, f"{args.log_root}/{run_name}/model.pth")
