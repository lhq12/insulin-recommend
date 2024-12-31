import os
import time
import math
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from model import GPTConfig, GPT
from glu_dataset import gluDataset
# -----------------------------------------------------------------------------

def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='./out-glucose-prediction')
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--always_save_checkpoint', action='store_true')
    parser.add_argument('--wandb_log', type=str2bool, default=True)
    parser.add_argument('--wandb_run_name', type=str, default='glucose-prediction')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=56)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--bias', type=str2bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=3e-2)
    parser.add_argument('--decay_lr', type=str2bool, default=True)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--lr_decay_iters', type=int, default=600000)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--max_epochs', type=int, default=400)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--visible_gpu', type=str, default='7')

    return parser.parse_args()

args = get_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpu
device = torch.device('cuda:0')
seed_offset = 0
tokens_per_iter = args.batch_size * args.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

out_dir = os.path.join(args.out_dir, f'{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}')
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
data_dir = '/remote-home/hongquanliu/Datasets/ZS_DATA/collect_by_id'
train_set = gluDataset(data_dir, mode='train', block_size=args.block_size)
val_set = gluDataset(data_dir, mode='val', block_size=args.block_size)
test_set = gluDataset(data_dir, mode='test', block_size=args.block_size)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)


max_iters = args.max_epochs * len(train_loader)
lr_decay_iters = max_iters

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
best_test_loss = 1e9

model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                  bias=args.bias, vocab_size=1, dropout=args.dropout) # start with model_args from command line
print("Initializing a new model from scratch")
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op

# optimizer
optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), 'cuda')

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

# logging
if args.wandb_log:
    import wandb
    wandb.init(project='Glucose Prediction', config=args.__dict__, mode="offline")

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
for epoch in range(args.max_epochs):
    for data in train_loader:
        
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for k, v in data.items():
            data[k] = v.to(device)
        logits, loss = model(**data)
        loss.backward()
        if args.grad_clip != 0.0:
            clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


         # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
        iter_num += 1

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss()
            test_out = estimate_loss('test')
            mae, mse = test_out['mae'], test_out['mse']
            if args.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "test/mse": mse,
                    "test/mae": mae,
                    "lr": lr,
                })
            if losses['val'] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': args.__dict__,
                        'test_mae': mae,
                        'test_mse': mse
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
