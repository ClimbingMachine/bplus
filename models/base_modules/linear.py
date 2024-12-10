# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail
from math import gcd
from typing import List

import torch
from torch import Tensor, nn


# Conv layer with norm (gn or bn) and relu.
class Conv(nn.Module):
    def __init__(
        self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True
    ):
        super(Conv, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.conv = nn.Conv2d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            padding=(int(kernel_size) - 1) // 2,
            stride=stride,
            bias=False,
        )

        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm = nn.BatchNorm2d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Conv1d(nn.Module):
    def __init__(
        self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True
    ):
        super(Conv1d, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.conv = nn.Conv1d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            padding=(int(kernel_size) - 1) // 2,
            stride=stride,
            bias=False,
        )

        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm="GN", ng=32, act=True):
        super(Linear, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.linear = nn.Linear(n_in, n_out, bias=False)

        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# Post residual layer
class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1, norm="GN", ng=32, act=True):
        super(PostRes, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.conv1 = nn.Conv2d(
            n_in, n_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            n_out, n_out, kernel_size=3, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == "GN":
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.bn1 = nn.BatchNorm2d(n_out)
            self.bn2 = nn.BatchNorm2d(n_out)
        else:
            exit("SyncBN has not been added!")

        if stride != 1 or n_out != n_in:
            if norm == "GN":
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        n_in, n_out, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.GroupNorm(gcd(ng, n_out), n_out),
                )
            elif norm == "BN":
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        n_in, n_out, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(n_out),
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(
        self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True
    ):
        super(Res1d, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == "GN":
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        if stride != 1 or n_out != n_in:
            if norm == "GN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(
                        n_in, n_out, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.GroupNorm(gcd(ng, n_out), n_out),
                )
            elif norm == "BN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(
                        n_in, n_out, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm1d(n_out),
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm="GN", ng=32):
        super(LinearRes, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == "GN":
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        if n_in != n_out:
            if norm == "GN":
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out),
                )
            elif norm == "BN":
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False), nn.BatchNorm1d(n_out)
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out


class Att(nn.Module):
    """
    Attention block to pass context nodes information to target nodes
    This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """

    def __init__(self, n_agt: int, n_ctx: int) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        agts: Tensor,
        agt_idcs: List[Tensor],
        agt_ctrs: List[Tensor],
        ctx: Tensor,
        ctx_idcs: List[Tensor],
        ctx_ctrs: List[Tensor],
        dist_th: float,
    ) -> Tensor:
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist**2).sum(2))
            mask = dist <= dist_th

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts


class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(
        self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor
    ) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts
