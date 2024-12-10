from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from models.base_modules import AttDest, LinearRes


class MidGoalNet(nn.Module):
    """
    Stage 1 mid goal with Linear Residual block
    """

    def __init__(self, config):
        super(MidGoalNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        self.mid_pred = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng),
            nn.Linear(n_actor, 2),
        )

    def forward(
        self,
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
    ) -> List[Tensor]:

        mid_reg = self.mid_pred(actors)
        mid_reg = mid_reg.view(mid_reg.size(0), 1, -1, 2)  # [N, 1, 1, 2]

        out = []

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]  # [n]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)  # [n, 1, 1, 2]
            mid_reg[idcs] = mid_reg[idcs] + ctrs  # [n, 1, 1, 2]
            out.append(mid_reg[idcs])

        return out


class GoalNet(nn.Module):
    """
    Stage 2 motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(GoalNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2),
                )
            )
        self.pred = nn.ModuleList(pred)
        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng),
            nn.Linear(n_actor, 1),
        )

    def forward(
        self,
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
    ) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)  # [N, 6, 2]
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)  # [N, 6, 1, 2]

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]  # [n]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)  # [n, 1, 1,  2]
            reg[idcs] = reg[idcs] + ctrs  # [n, 6, 60, 2]

        dest_ctrs = reg[:, :, -1].detach()  # [N, 6, 2]
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])  # [N, 6]

        cls, sort_idcs = cls.sort(1, descending=True)  # [N, 6] and [N, 6]
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(
            cls.size(0), cls.size(1), -1, 2
        )  # sorted regression

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])

        return out


class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng),
            nn.Linear(n_actor, 1),
        )

    def forward(
        self,
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
    ) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out
