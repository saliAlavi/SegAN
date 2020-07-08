import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from model import *
from create_dataloader import *
import torch.optim as optim

from tqdm import tqdm


def loss_dice(source, target):
    assert source.size() == target.size()
    assert source.dim() == 4

    cross = source * target
    cross = torch.sum(cross, dim=3)
    cross = torch.sum(cross, dim=2)

    auto_1 = source * source
    auto_1 = torch.sum(auto_1, dim=3)
    auto_1 = torch.sum(auto_1, dim=2)

    auto_2 = source * source
    auto_2 = torch.sum(auto_1, dim=3)
    auto_2 = torch.sum(auto_1, dim=2)

    loss = cross / (auto_1 + auto_2)
    loss /= loss.size(0)

    return loss


if __name__ == "__main__":
    args = None

    dl_BraT_train = None
    dl_BraT_val = None
    d_in = 3
    d_out = 1
    NetS = Segmentor()
    NetC = Critic()

    optimizer_S = optim.Adam(NetS.params(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_C = optim.Adam(NetC.params(), lr=args.lr, betas=(args.beta1, args.beta2))

    if args.gpu:
        device = torch.device('cuda')
        NetS = NetS.to(device)
        NetC = NetC.to(device)
for epoch in range(1, args.epochs + 1):
    NetS.train()
    for j, data in enumerate(dl_BraT_train, 1):
        optimizer_C.zero_grad()
        optimizer_S.zero_grad()

        image, gt = data

        seg_out = NetS(image)
        seg_out = F.softmax(seg_out)
        seg_out = seg_out.detach()
        critic_prd, critic_gt = NetC(image, seg_out, gt)

        loss_C = 1 - torch.mean(torch.abs(critic_prd - critic_gt))

        loss_C.backward()
        optimizer_C.step()

        for p in NetC.params():
            p.data._clamp(-0.01, 0.01)

        seg_out = NetS(image)
        seg_out = F.softmax(seg_out)
        loss_S_target = torch.mean(torch.abs(seg_out, gt))
        critic_prd, critic_gt = NetC(image, seg_out, gt)

        loss_S_dice = loss_dice(critic_prd, critic_gt)
        loss_S = args.alpha * loss_S_dice + loss_S_target

        loss_S.backward()
        optimizer_S.step()
    NetS.eval()
    for j, data in enumerate(dl_BraT_val, 1):
        image, gt = data

        seg_out = NetS(image)
        seg_out = F.softmax(seg_out)
        loss_S_target = torch.mean(torch.abs(seg_out, gt))
