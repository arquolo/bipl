# flake8: noqa

import glow.nn as gnn

optD, optG = gnn.amp_init_opt([netD, netG], [optD, optG])


def step():
    netD.train()
    with optD:
        errD_real = criterion(netD(real), real_label)
        optD.backward(errD_real)

        fake = netG(noise)
        errD_fake = criterion(netD(fake.detach()), fake_label)
        optD.backward(errD_fake)

    netD.eval()
    with optG:
        errG = criterion(netD(fake), real_label)
        optG.backward(errG)
