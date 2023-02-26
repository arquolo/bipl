import argparse
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
from torchvision import transforms as tfs
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

import glow
import glow.metrics as m
import glow.nn as gnn

DEVICE = torch.device('cuda')

glow.lock_seed(42)
torch.backends.cudnn.benchmark = True  # type: ignore
rg = np.random.default_rng()

# ------------------------------ define model ------------------------------


def make_model_default():
    return nn.Sequential(
        nn.Conv2d(3, 6, 5),  # > 28^2
        nn.ReLU(),
        nn.MaxPool2d(2),  # > 14^2
        nn.Conv2d(6, 16, 5),  # > 10^2
        nn.ReLU(),
        nn.MaxPool2d(2),  # > 5^2
        gnn.View(-1),  # > 1:400
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
    )


def make_model_new(init=16):
    def conv(cin, cout=None, groups=1, pad=2, stride=1):
        cout = cout or cin
        ksize = stride + pad * 2
        return nn.Sequential(
            nn.Conv2d(
                cin, cout, ksize, stride, pad, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def conv_down(cin, cout=None):
        cout = cout or cin
        return nn.Sequential(
            conv(cin, cout, pad=1, stride=2),
            gnn.Sum(
                conv(cout, cout * 2),
                conv(cout * 2, pad=2, groups=cout * 2),
                conv(cout * 2, cout)[:-1],
                tail=nn.ReLU(),
                skip=0.1),
        )

    return nn.Sequential(
        conv_down(3, init),  # > 16^2
        conv_down(init, init * 2),  # > 8^2
        conv_down(init * 2, init * 4),  # > 4^2
        conv(init * 4, init * 8, pad=2),
        nn.AdaptiveAvgPool2d(1),  # > 1
        gnn.View(-1),
        nn.Linear(init * 8, 10),
    )


# parse args

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'root', type=pathlib.Path, help='location of cifar10/ folder')
parser.add_argument('--batch-size', type=int, default=4, help='train batch')
parser.add_argument('--epochs', type=int, default=12, help='count of epochs')
parser.add_argument('--steps-per-epoch', type=int, help='steps per epoch')
parser.add_argument('--width', type=int, default=32, help='width of network')
parser.add_argument(
    '--fp16', action='store_true', help='enable mixed precision mode')
parser.add_argument('--plot', action='store_true', help='enable plot')

args = parser.parse_args()

epoch_len = (8000 // args.batch_size
             if args.steps_per_epoch is None else args.steps_per_epoch)
sample_size = args.epochs * epoch_len * args.batch_size

tft = tfs.Compose([
    tfs.RandomCrop(32, padding=4),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
])
ds = CIFAR10(args.root / 'cifar10', transform=tft, download=True)
ds_val = CIFAR10(args.root / 'cifar10', transform=tfs.ToTensor(), train=False)

loader = gnn.make_loader(
    ds,
    args.batch_size,
    sampler=RandomSampler(ds, True, sample_size),
    multiprocessing=False)
val_loader = gnn.make_loader(ds_val, 100, multiprocessing=False)

# net = make_model_default()
net = make_model_new(args.width)
net.to(DEVICE)
opt = torch.optim.AdamW(net.parameters())
print(gnn.param_count(net))

criterion = nn.CrossEntropyLoss()
metrics = [
    m.Lambda(criterion, name='loss'),
    m.Confusion(acc=m.accuracy, kappa=m.kappa),
]
stepper = gnn.Stepper(
    net, opt, criterion, metrics, device=DEVICE, fp16=args.fp16)

history = defaultdict[str, list](list)
with tqdm(total=epoch_len * args.epochs, desc='train') as pbar:
    for i, split in enumerate(glow.ichunked(loader, epoch_len), 1):
        tscalars = stepper.run(split, pbar).scalars
        with tqdm(total=len(val_loader), desc='val', leave=False) as pbar_val:
            vscalars = stepper.run(val_loader, pbar_val, False).scalars

        _tags = sorted({*tscalars, *vscalars})
        scores = {tag: [tscalars[tag], vscalars[tag]] for tag in _tags}

        for tag, values in scores.items():
            history[tag].append(values)
        scalars_fmt = ', '.join(
            f'{t}: {{:.3f}}/{{:.3f}}'.format(*vs) for t, vs in scores.items())
        print(f'[{i:03d}] {scalars_fmt}')

if args.plot:
    fig = plt.figure(f'batch_size={args.batch_size}, fp16={args.fp16}')
    for i, (tag, values_) in enumerate(history.items(), 1):
        ax = fig.add_subplot(1, len(history), i)
        ax.legend(ax.plot(values_), ['train', 'val'])
        ax.set_title(tag)
        ax.set_ylim([
            int(min(x for xs in values_ for x in xs)),
            int(max(x for xs in values_ for x in xs) + 0.999)
        ])
    plt.show()
