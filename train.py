"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from model import GPTConfig, GPT
from glu_dataset import gluDataset
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O


out_dir = './out-glucose-prediction'
eval_interval = 5 # keep frequent because we'll overfit
log_interval = 10 # don't print too too often
always_save_checkpoint = False # if True, always save a checkpoint after each eval
# wandb logging
wandb_log = True # override via command line if you like
wandb_project = 'Glucose Prediction'
wandb_run_name = 'mini-gpt'

# data
dataset = 'glucose_dataset'
batch_size = 64
block_size = 56 # context of up to 256 previous characters

# model
# baby GPT model :)
n_layer = 3
n_head = 4
n_embd = 512
dropout = 0.
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 3e-2 # with baby networks can afford to go a bit higher
max_epochs = 400
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
weight_decay = 1e-4
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
visible_gpu = 0 # which gpu to use
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_gpu)
device = torch.device('cuda:0')
seed_offset = 0
tokens_per_iter = batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

out_dir = os.path.join(out_dir, f'{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}')
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
data_dir = '/remote-home/hongquanliu/Datasets/ZS_DATA/collect_by_id'
train_set = gluDataset(data_dir, mode='train', block_size=block_size)
val_set = gluDataset(data_dir, mode='val', block_size=block_size)
test_set = gluDataset(data_dir, mode='test', block_size=block_size)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)


max_iters = max_epochs * len(train_loader)
lr_decay_iters = max_iters

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
best_test_loss = 1e9

# attempt to derive vocab_size from the dataset
# meta_path = os.path.join(data_dir, 'meta.pkl')
# meta_vocab_size = None
# if os.path.exists(meta_path):
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     meta_vocab_size = meta['vocab_size']
#     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=1, dropout=dropout) # start with model_args from command line
print("Initializing a new model from scratch")
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), 'cuda')

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, mode="offline")

@torch.no_grad()
def estimate_loss(mode=None):
    out = {}
    model.eval()
    losses = []
    mae = []
    if mode == 'test':
        loader = test_loader
        for data in loader:
            for k, v in data.items():
                data[k] = v.to(device)
            logits, loss = model(**data)
            losses.append(loss.item())
            mae.append((torch.sum(torch.abs(logits.squeeze(-1) - data['label']) * data['mask']) / (data['mask'].sum() + 1e-9)).item())
        mae = np.mean(mae)
        mse = np.mean(losses)

        model.train()
        return {
            'mae': mae,
            'mse': mse
        }

    for mode in ['train', 'val']:
        loader = train_loader if mode == 'train' else val_loader
        for data in loader:
            for k, v in data.items():
                data[k] = v.to(device)
            logits, loss = model(**data)
            losses.append(loss.item())
        out[mode] = np.mean(losses)
    model.train()
    return out

# training loop
t0 = time.time()
iter_num = 0
for epoch in range(max_epochs):
    for data in train_loader:
        
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for k, v in data.items():
            data[k] = v.to(device)
        logits, loss = model(**data)
        loss.backward()
        if grad_clip != 0.0:
            clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


         # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
        iter_num += 1

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            test_out = estimate_loss('test')
            mae, mse = test_out['mae'], test_out['mse']
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "test/mse": mse,
                    "test/mae": mae,
                    "lr": lr,
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                        'test_mae': mae,
                        'test_mse': mse
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
