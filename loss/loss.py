import numpy as np
import torch
from torch import Tensor, nn
from utils.utils import gpu

from numpy import ndarray
from typing import Dict, List, Optional, Union

################################## newly added goal loss ##################################
class GoalLoss(nn.Module):
    def __init__(self, config):
        super(GoalLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, goal: Dict[str, List[Tensor]], loc_preds: List[Tensor], has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        cls, goal_reg, mid_goal_reg = goal["cls"], goal["reg"], goal['mid']
        cls       = torch.cat([x for x in cls], 0)
        goal_reg  = torch.cat([x for x in goal_reg], 0)
        loc_preds = torch.cat([x for x in loc_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)
        mid_goal_reg = torch.cat([x for x in mid_goal_reg], 0)
         
        loss_out = dict()
        zero = 0.0 * (cls.sum() + goal_reg.sum())
        loss_out["goal_cls_loss"] = zero.clone()
        loss_out["goal_num_cls"] = 0
        loss_out["goal_reg_loss"] = zero.clone()
        loss_out["goal_num_reg"] = 0
        loss_out["mid_goal_reg_loss"] = zero.clone()
        loss_out["mid_goal_num_reg"] = 0

        num_mods, num_preds, mid_pos = self.config["num_mods"], self.config["num_preds"], self.config["num_preds"]//2 - 1
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        _, last_idcs = last.max(1)                                                   # last_idcs indicates that whether the last frame is available
        mask     = last_idcs == self.config["num_preds"] - 1
        mid_mask = last_idcs >= mid_pos

        mid_goal_reg  = mid_goal_reg[mid_mask]
        mid_preds     = loc_preds[mid_mask]
        
        # mid goal regression loss
        coef = self.config["mid_goal_reg_coef"]                     
        loss_out["mid_goal_reg_loss"] += coef * self.reg_loss(
            mid_goal_reg, mid_preds[:, mid_pos, :]
        )
        loss_out["mid_goal_num_reg"] += mid_goal_reg.size(0)

        cls       = cls[mask]
        goal_reg  = goal_reg[mask]
        loc_preds = loc_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (goal_reg[row_idcs, j, -1] - loc_preds[row_idcs, last_idcs].to(goal_reg.device))
                        ** 2
                    ).sum(1)
                )
            )

        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)
        
        # max-margin loss
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]

        coef = self.config["goal_cls_coef"]
        loss_out["goal_cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["goal_num_cls"] += mask.sum().item()

        # final displacement loss
        goal_reg = goal_reg[row_idcs, min_idcs]                 #[N', 1, 2]

        coef = self.config["goal_reg_coef"]                     
        loss_out["goal_reg_loss"] += coef * self.reg_loss(
            goal_reg, loc_preds[:, -1, :].unsqueeze(1)
        )

        loss_out["goal_num_reg"] += goal_reg.size(0)
        return loss_out

################################## goal loss ends##################################


class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor], loss_out: Dict) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        gt_preds = gt_preds.to(reg.device)
        has_preds = torch.cat([x for x in has_preds], 0)

        #loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        # last_idcs indicates that whether the last frame is available
        max_last, last_idcs = last.max(1)                           
        # max_last indicates whether the ob has at least one valid frame        
        mask = max_last > 1.0

        # valid final cls scores
        cls = cls[mask]
        # valid final prediction positions
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )

        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out

class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)
        self.goal_loss = GoalLoss(config)

    def forward(self, goal: Dict, out: Dict, data: Dict) -> Dict:
        loss_out  = self.goal_loss(goal, gpu(data["loc_preds"]), gpu(data["has_preds"]))
        loss_out  = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]), loss_out)

        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)

        loss_out["loss"] += loss_out["goal_cls_loss"] / (
            loss_out["goal_num_cls"] + 1e-10
        ) + loss_out["goal_reg_loss"] / (loss_out["goal_num_reg"] + 1e-10)

        loss_out["loss"] += loss_out["mid_goal_reg_loss"] / (loss_out["mid_goal_num_reg"] + 1e-10)

        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out, data):
        post_out = dict()
        post_out["preds"]     = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"]  = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["loc_preds"] = [x[0:1].numpy() for x in data["loc_preds"]]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        goal_cls = metrics["goal_cls_loss"] / (metrics["goal_num_cls"] + 1e-10)
        goal_reg = metrics["goal_reg_loss"] / (metrics["goal_num_reg"] + 1e-10)
        loss = cls + reg + goal_cls + goal_reg

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, _ = pred_metrics(preds, gt_preds, has_preds)

        print(
            "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
            % (loss, cls, reg, ade1, fde1, ade, fde)
        )
        print()


def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs
