# Copyright (c) 2024 Horizon Robotics
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, List

import torch
from torch import Tensor, nn

from loss import GaNetLoss, PostProcess
from models.base_modules import A2A, A2M, B2M, M2A, M2M
from models.decoder import GoalNet, MidGoalNet, PredNet
from models.encoder import ActorNet, BoundaryNet, MapNet
from utils import actor_gather, boundary_gather, graph_gather
from utils.helper import (
    Optimizer,
    StepLR,
    collate_fn,
    to_long,
    transfer_device,
)

file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)

config = dict()

config["display_iters"] = 199908 // 2
config["val_iters"] = 199908
config["save_freq"] = 1.0
config["epoch"] = 0
config["horovod"] = False
config["opt"] = "adam"
config["num_epochs"] = 36
config["lr"] = [1e-3, 1e-4]
config["lr_epochs"] = [32]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])
config["device"] = "cuda:0"

if "save_dir" not in config:
    config["save_dir"] = os.path.join(root_path, "results", "banet")

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 1
config["val_batch_size"] = 1
config["workers"] = 0
config["val_workers"] = config["workers"]
config["intersect_dim"] = 3
config["lane_type_dim"] = 4
config["mark_type_dim"] = 14
config["agent_type_dim"] = 14


config["rot_aug"] = False
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6
config["n_actor"] = 128
config["n_map"] = 128
config["actor2map_dist"] = 50.0
config["map2actor_dist"] = 50.0
config["actor2actor_dist"] = 100.0
config["pred_size"] = 60
config["pred_step"] = 1
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 6

config["goal_cls_coef"] = 1
config["goal_reg_coef"] = 0.2
config["mid_goal_reg_coef"] = 0.1

config["cls_coef"] = 2.0
config["reg_coef"] = 1.0
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2
config["device"] = "cpu"


class BANet(nn.Module):
    """
    Lane Graph Network contains following components:
        1. ActorNet: a 1D CNN to process the trajectory input
        2. MapNet: LaneGraphCNN to learn structured map representations
           from vectorized map data
        3. Actor-Map Fusion Cycle: fuse the information between actor nodes
           and lane nodes:
            a. A2M: introduces real-time traffic information to
                lane nodes, such as blockage or usage of the lanes
            b. M2M:  updates lane node features by propagating the
                traffic information over lane graphs
            c. M2A: fuses updated map features with real-time traffic
                information back to actors
            d. A2A: handles the interaction between actors and produces
                the output actor features
        4. PredNet: prediction header for motion forecasting using
           feature from A2A
    """

    def __init__(self, config):
        super(BANet, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        # boundary encoder
        self.lb_net = BoundaryNet(config)
        self.rb_net = BoundaryNet(config)

        self.a2m = A2M(config)
        # self.a2l = A2B(config)
        # self.a2r = A2B(config)

        # boundary to map net
        self.l2m = B2M(config)
        self.r2m = B2M(config)

        self.m2m = M2M(config)
        self.m2a = M2A(config)

        # newly added
        self.l2a = M2A(config)
        self.r2a = M2A(config)
        self.a2a = A2A(config)

        # add goal net
        self.mid_net = MidGoalNet(config)
        self.m_m2a = M2A(config)
        self.m_l2a = M2A(config)
        self.m_r2a = M2A(config)
        self.m_a2a = A2A(config)

        self.goal_net = GoalNet(config)
        self.g_m2a = M2A(config)
        self.g_l2a = M2A(config)
        self.g_r2a = M2A(config)
        self.g_a2a = A2A(config)

        self.pred_net = PredNet(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature
        actors, actor_idcs = actor_gather(
            transfer_device(data["features"], self.config["device"])
        )
        actor_ctrs = transfer_device(data["ctrs"], self.config["device"])
        actors = self.actor_net(actors)

        # construct map features
        graph = graph_gather(
            to_long(transfer_device(data["cl_graph"], self.config["device"]))
        )
        left_graph = boundary_gather(
            to_long(transfer_device(data["left_graph"], self.config["device"]))
        )
        right_graph = boundary_gather(
            to_long(
                transfer_device(data["right_graph"], self.config["device"])
            )
        )

        nodes, node_idcs, node_ctrs = self.map_net(graph)
        left_nodes, left_node_idcs, left_node_ctrs = self.lb_net(left_graph)
        right_nodes, right_node_idcs, right_node_ctrs = self.rb_net(
            right_graph
        )

        # actor-map fusion cycle
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)

        # fuse boundaries
        nodes = self.l2m(
            nodes, graph, left_nodes, left_node_idcs, left_node_ctrs
        )
        nodes = self.r2m(
            nodes, graph, right_nodes, right_node_idcs, right_node_ctrs
        )

        nodes = self.m2m(nodes, graph)
        actors = self.m2a(
            actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs
        )
        actors = self.l2a(
            actors,
            actor_idcs,
            actor_ctrs,
            left_nodes,
            left_node_idcs,
            left_node_ctrs,
        )
        actors = self.r2a(
            actors,
            actor_idcs,
            actor_ctrs,
            right_nodes,
            right_node_idcs,
            right_node_ctrs,
        )
        actors = self.a2a(actors, actor_idcs, actor_ctrs)

        # Mid-Goal Net
        mid_goal = self.mid_net(actors, actor_idcs, actor_ctrs)
        mid_ctrs = [x.squeeze() for x in mid_goal]
        actors = self.m_m2a(
            actors, actor_idcs, mid_ctrs, nodes, node_idcs, node_ctrs
        )
        actors = self.m_l2a(
            actors,
            actor_idcs,
            mid_ctrs,
            left_nodes,
            left_node_idcs,
            left_node_ctrs,
        )
        actors = self.m_r2a(
            actors,
            actor_idcs,
            mid_ctrs,
            right_nodes,
            right_node_idcs,
            right_node_ctrs,
        )
        actors = self.m_a2a(actors, actor_idcs, mid_ctrs)

        # GaNet Procedure
        goal = self.goal_net(actors, actor_idcs, actor_ctrs)
        goal["mid"] = mid_ctrs
        goal_ctrs = [x.squeeze(2)[:, 0, :] for x in goal["reg"]]

        actors = self.g_m2a(
            actors, actor_idcs, goal_ctrs, nodes, node_idcs, node_ctrs
        )
        actors = self.g_l2a(
            actors,
            actor_idcs,
            goal_ctrs,
            left_nodes,
            left_node_idcs,
            left_node_ctrs,
        )
        actors = self.g_r2a(
            actors,
            actor_idcs,
            goal_ctrs,
            right_nodes,
            right_node_idcs,
            right_node_ctrs,
        )

        actors = self.g_a2a(actors, actor_idcs, goal_ctrs)  # 2D X in GANet

        # prediction
        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        rot, orig = transfer_device(
            data["rot"], self.config["device"]
        ), transfer_device(data["orig"], self.config["device"])
        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
        return goal, out


def get_banet():
    net = BANet(config)
    net = net.cuda()

    loss = GaNetLoss(config).cuda()
    post_process = PostProcess(config).cuda()

    params = net.parameters()
    opt = Optimizer(params, config)

    return config, collate_fn, net, loss, post_process, opt
