from __future__ import annotations

import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm.auto import tqdm

import glow.nn
from glow import metrics as m

metrics: tuple[m.Metric, ...] = (
    m.Lambda(m.accuracy_),
    m.Confusion(
        acc=m.accuracy,
        accb=m.accuracy_balanced,
        iou=m.iou,
        kappa=m.kappa,
    ),
)

c = 8
b = 128
true = torch.randint(c, size=[b])
pred = torch.randn(b, c, requires_grad=True)


class Model(nn.Module):
    def forward(self, x):
        return net.param


net = Model()
net.param = nn.Parameter(data=pred, requires_grad=True)

optim = glow.nn.RAdam(net.parameters())
cm_grad = m.ConfusionGrad()

plt.ion()
_, ax = plt.subplots(ncols=4)
ax[2].plot(true.numpy())
with tqdm(range(32)) as pbar:
    for _ in pbar:
        for _ in range(64):
            net.zero_grad()
            cm = cm_grad(net(None), true)
            loss = -m.accuracy(cm)
            # loss = -m.accuracy_balanced(cm)
            # loss = -m.kappa(cm)
            loss.backward()
            optim.step()
            pbar.set_postfix({'score': -loss.detach_().item()})
            cm.detach_()

        ax[0].imshow(pred.detach().numpy())
        ax[1].imshow(pred.detach().softmax(1).numpy())
        ax[2].cla()
        ax[2].plot(sorted(zip(true.numpy(), pred.detach().argmax(1).numpy())))
        ax[3].imshow(cm.numpy(), vmax=1 / c)

        plt.pause(1e-2)

with torch.no_grad():
    meter = m.compose(*metrics)
    d = meter.send((pred, true))
    print(', '.join(f'{k}: {v:.3f}' for k, v in d.scalars.items()))
    print(', '.join(f'{k}: {v.item():.3f}'
                    for k, v in d.tensors.items() if v.numel() == 1))
