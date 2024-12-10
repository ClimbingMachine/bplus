from math import gcd
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch import Tensor

from models.base_modules.linear import Att, Linear, MLPLayer


class B2A(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """

    def __init__(self, config):
        super(B2A, self).__init__()
        self.config = config

        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_map, n_map))
        self.att = nn.ModuleList(att)

    def forward(
        self,
        nodes: Tensor,
        node_idcs: List[Tensor],
        node_ctrs: List[Tensor],
        bounds: Tensor,
        bound_idcs: List[Tensor],
        bound_ctrs: List[Tensor],
    ) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                nodes,
                node_idcs,
                node_ctrs,
                bounds,
                bound_idcs,
                bound_ctrs,
                self.config["map2actor_dist"],
            )
        return actors


class A2M(nn.Module):
    """
    Actor to Map Fusion:  fuses real-time traffic information from
    actor nodes to lane nodes
    """

    def __init__(self, config):
        super(A2M, self).__init__()
        self.config = config

        # configuration status
        n_map = config["n_map"]
        intersect_dim = config["intersect_dim"]
        lane_embed = config["lane_type_dim"]

        # parameter embeddings
        self.intersect_embed = nn.Embedding(intersect_dim, n_map)
        self.lane_embed = nn.Embedding(lane_embed, n_map)

        """fuse meta, static, dyn"""
        # self.meta = Linear(n_map * 3, n_map, norm=norm, ng=ng)
        self.meta = MLPLayer(n_map * 3, n_map, n_map)

        att = []
        for i in range(2):
            att.append(Att(n_map, config["n_actor"]))
        self.att = nn.ModuleList(att)

    def forward(
        self,
        feat: Tensor,
        graph: Dict[
            str,
            Union[
                List[Tensor],
                Tensor,
                List[Dict[str, Tensor]],
                Dict[str, Tensor],
            ],
        ],
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
    ) -> Tensor:
        """meta, static and dyn fuse using attention"""
        meta = torch.cat(
            (
                self.intersect_embed(graph["intersect"].long()),
                self.lane_embed(graph["lane_type"].long()),
            ),
            1,
        )

        feat = self.meta(torch.cat((feat, meta), 1))

        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2map_dist"],
            )
        return feat


class M2M(nn.Module):
    """
    The lane to lane block: propagates information over lane
            graphs and updates the features of lane nodes
    """

    def __init__(self, config):
        super(M2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(
                        Linear(n_map, n_map, norm=norm, ng=ng, act=False)
                    )
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: Tensor, graph: Dict) -> Tensor:
        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    if len(graph[k1][k2]["u"]) > 0:
                        temp.index_add_(
                            0,
                            graph[k1][k2]["u"],
                            self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                        )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat


class M2A(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """

    def __init__(self, config):
        super(M2A, self).__init__()
        self.config = config
        n_actor = config["n_actor"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_map))
        self.att = nn.ModuleList(att)

    def forward(
        self,
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
        nodes: Tensor,
        node_idcs: List[Tensor],
        node_ctrs: List[Tensor],
    ) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["map2actor_dist"],
            )
        return actors


class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """

    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config

        n_actor = config["n_actor"]

        att = []
        for _ in range(2):
            att.append(Att(n_actor, n_actor))
        self.att = nn.ModuleList(att)

    def forward(self, actors, actor_idcs, actor_ctrs):
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2actor_dist"],
            )
        return actors


class B2M(nn.Module):
    """
    Actor to Map Fusion:  fuses real-time traffic information from
    actor nodes to lane nodes
    """

    def __init__(self, config):
        super(B2M, self).__init__()
        self.config = config

        # configuration status
        n_map = config["n_map"]
        # mark_embed = config['mark_type_dim']
        norm = "GN"
        ng = 1

        # parameter embeddings
        # self.mark_embed = nn.Embedding(mark_embed, n_map)

        """fuse meta, static, dyn"""
        self.meta = Linear(n_map, n_map, norm=norm, ng=ng)

        att = []
        for i in range(2):
            att.append(Att(n_map, config["n_actor"]))
        self.att = nn.ModuleList(att)

    def forward(
        self,
        feat: Tensor,
        graph: Dict[
            str,
            Union[
                List[Tensor],
                Tensor,
                List[Dict[str, Tensor]],
                Dict[str, Tensor],
            ],
        ],
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
    ) -> Tensor:
        """meta, static and dyn fuse using attention"""

        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2map_dist"],
            )
        return feat


class A2B(nn.Module):
    """
    Actor to Map Fusion:  fuses real-time traffic information from
    actor nodes to lane nodes
    """

    def __init__(self, config):
        super(A2B, self).__init__()
        self.config = config

        # configuration status
        n_map = config["n_map"]
        lane_mark_dim = config["mark_type_dim"]
        norm = "GN"
        ng = 1

        # parameter embeddings
        self.mark_embed = nn.Embedding(lane_mark_dim, n_map)

        """fuse meta, static, dyn"""
        self.meta = Linear(n_map * 2, n_map, norm=norm, ng=ng)

        att = []
        for i in range(2):
            att.append(Att(n_map, config["n_actor"]))
        self.att = nn.ModuleList(att)

    def forward(
        self,
        feat: Tensor,
        graph: Dict[
            str,
            Union[
                List[Tensor],
                Tensor,
                List[Dict[str, Tensor]],
                Dict[str, Tensor],
            ],
        ],
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
    ) -> Tensor:
        """meta, static and dyn fuse using attention"""

        meta = self.mark_embed(graph["mark_type"].long())
        feat = self.meta(torch.cat((feat, meta), 1))

        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2map_dist"],
            )
        return feat
