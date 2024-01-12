import os, torch, math, sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import sparse
from pathlib import Path
from typing import Dict, List, Any
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.interpolate import compute_midpoint_line, interp_arc

from torch.utils.data import Dataset

sys.path.append('..')
os.environ['PYTHONPATH'] = os.pathsep.join(sys.path)
from utils import side_to_directed_lineseg


# global parameters 
lane_types  = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']

agent_types = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 
               'background', 'construction', 'riderless_bicycle', 'unknown']

lane_marks  = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_YELLOW', 'DASHED_WHITE', 
               'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE', 'DOUBLE_SOLID_YELLOW', 
               'DOUBLE_SOLID_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_DASH_WHITE', 'SOLID_BLUE',
               'NONE', 'BLUE', "PEDESTRIAN"]

def get_agent_features(df: pd.DataFrame, 
                       historical_frames = 50, 
                       future_frames = 60, 
                       dim = 2, test = False):
    
    # df = pd.read_parquet(log_data_dir_path, engine='pyarrow')
    total_frames = historical_frames + future_frames
    agent_ids    = list(df['track_id'].unique())                               # list of valid identifiers
    num_agents   = len(agent_ids)

    av_idx       = agent_ids.index('AV')
    focal_index  = agent_ids.index(df['focal_track_id'][0])

    # initialization of vectors
    valid_mask     = torch.zeros(num_agents, total_frames, dtype=bool)
    position       = torch.zeros(num_agents, total_frames, dim, dtype=torch.float)
    velocity       = torch.zeros(num_agents, total_frames, dim, dtype=torch.float)
    heading        = torch.zeros(num_agents, total_frames, dtype=torch.float)
    agent_type     = torch.zeros(num_agents, dtype=torch.uint8)
    agent_category = torch.zeros(num_agents, dtype=torch.uint8)

    for track_id, track_df in df.groupby('track_id'):
        agent_idx = agent_ids.index(track_id)                          # index of the track id 
        agent_steps = track_df['timestep'].values                      # track id valid steps

        valid_mask[agent_idx, agent_steps] = True
        agent_type[agent_idx] = agent_types.index(track_df['object_type'].values[0])
        agent_category[agent_idx] = track_df['object_category'].values[0]
        position[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['position_x'].values,
                                                                            track_df['position_y'].values],
                                                                            axis=-1)).float()
        
        heading[agent_idx, agent_steps] = torch.from_numpy(track_df['heading'].values).float()
        velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values,
                                                                            track_df['velocity_y'].values],
                                                                            axis=-1)).float()
        

        valid_mask[agent_idx, 1: historical_frames] = (valid_mask[agent_idx, :historical_frames - 1] & 
                                                       valid_mask[agent_idx, 1: historical_frames])
        valid_mask[agent_idx, 0] = False
    
    agent_preprocess = {
        'num_nodes': num_agents,
        'av_index': av_idx,
        'valid_mask': valid_mask,
        'focal_index': focal_index,
        'id': agent_ids,
        'type': agent_type,
        'category': agent_category,
        'position': position,
        'heading': heading,
        'velocity': velocity,
    }


    return get_obj_features(agent_preprocess, historical_frames, future_frames, test)



def get_obj_features(data: Dict, 
                     historical_frames: int = 50, 
                     future_frames: int     = 60, 
                     test: bool = False) -> Dict[str, Any]:

    focal_agent_idx = data['focal_index']                                          # focal agent index
    assert data['valid_mask'][data['focal_index'], historical_frames-1] == 1       # assert the focal agent is valid
    theta  = data['heading'][focal_agent_idx][historical_frames-1]                 # theta of the focal agent
    origin = data['position'][focal_agent_idx][historical_frames-1]                # theta of the focal agent

    rotate_mat = torch.Tensor([[torch.cos(theta), -torch.sin(theta)],              # rotation matrix
                               [torch.sin(theta),  torch.cos(theta)]])
    
    predict_mask = data['valid_mask'][:, historical_frames:]                       # has the shape [N, 60]
    observe_mask = data['valid_mask'][:, :historical_frames]                       # has the shape [N, 50]
    feats, ctrs, gt_preds, has_preds, hist, local, types = [], [], [], [], [], [], []

    reduced_agent_ids = []

    for idx in range(data['num_nodes']):
        if not observe_mask[idx, historical_frames - 1]:
            continue
        # current steps
        curr_step = observe_mask[idx]

        # append reduced ids
        reduced_agent_ids.append(data['id'][idx])

        historical_trajectory = data['position'][idx, :historical_frames, :][curr_step]
        historical_velocity   = data['velocity'][idx, :historical_frames, :][curr_step]
        historical_heading    = data['heading'][idx,  :historical_frames][curr_step]
        agent_type            = data['type'][idx]

        # feature vector
        agent_features = torch.zeros((historical_frames, 7), dtype = torch.float32)
        agent_features[curr_step, :2] = torch.matmul(rotate_mat, (historical_trajectory - origin).T).T
        agent_features[curr_step, 2:4] = torch.matmul(rotate_mat, historical_velocity.T).T
        agent_features[curr_step, 4:5] = torch.sin(historical_heading + theta).view(-1, 1)
        agent_features[curr_step, 5:6] = torch.cos(historical_heading + theta).view(-1, 1)
        agent_features[curr_step, 6]   = 1

        hist.append(agent_features[:, :2].clone())
        ctrs.append(agent_features[-1, :2].clone())
        types.append(agent_type)

        # trajectory feature extraction
        feat = agent_features[1:, :2].clone() - agent_features[:-1, :2].clone()
        agent_features[1:, :2] = feat


        # future trajectory and masks
        gt_pred  = torch.zeros((future_frames, 2), dtype = torch.float32)
        loc_pred = torch.zeros((future_frames, 2), dtype = torch.float32)
        has_pred = torch.zeros(future_frames, dtype = bool)

        if not test:
            future_steps           = predict_mask[idx]
            predict_trajectory     = data['position'][idx, historical_frames:, :][future_steps]
            gt_pred[future_steps]  = predict_trajectory
            loc_pred[future_steps] = torch.matmul(rotate_mat, (predict_trajectory - origin).T).T
            has_pred[future_steps] = 1
        

        feats.append(agent_features)
        gt_preds.append(gt_pred)
        has_preds.append(has_pred)
        local.append(loc_pred)

    local = torch.stack(local)
    hist  = torch.stack(hist)
    feats = torch.stack(feats)
    ctrs  = torch.stack(ctrs)
    types = torch.stack(types).long()

    gt_preds = torch.stack(gt_preds)                            # groundtruth prediction instances
    has_preds = torch.stack(has_preds)
    
    reduced_focal_idx = reduced_agent_ids.index(data['id'][focal_agent_idx])
    if not test: assert has_preds[reduced_focal_idx].all()

    return {
        'historical_trajectory': hist,
        'focal_idx': reduced_focal_idx,
        'ctrs': ctrs,
        'agent_features': feats,
        'types': types, 
        'orig': origin,
        'theta': theta,
        'rot': rotate_mat,
        'gt_preds': gt_preds,
        'has_preds': has_preds,
        'loc_preds': local        
    }


def get_graph_features(data: Dict[str, Any], avm: ArgoverseStaticMap):
    
    #
    lane_ids         = list(avm.vector_lane_segments.keys())
    total_lanes      = len(lane_ids)
    polygon_ids      = []

    left_ctrs, right_ctrs, ctrs = [], [], []
    left_feats, right_feats, feats = [], [], []

    left_mark_lst, right_mark_lst, intersect_lst, lane_semantic_lst = [], [], [], []

    for _, lane_id in enumerate(lane_ids):
        curr_lane_seg = avm.vector_lane_segments[lane_id]
        # boundaries and markings 
        right_boundary  = torch.from_numpy(curr_lane_seg.right_lane_marking.polyline[:, :2]).float()
        left_boundary   = torch.from_numpy(curr_lane_seg.left_lane_marking.polyline[:, :2]).float()
        centerline      = torch.from_numpy(avm.get_lane_segment_centerline(lane_id)[:, :2]).float()

        # convert to local view
        right_boundary = torch.matmul(data['rot'], (right_boundary - data['orig']).T).T
        left_boundary  = torch.matmul(data['rot'], (left_boundary - data['orig']).T).T
        centerline = torch.matmul(data['rot'], (centerline - data['orig']).T).T

        # control-based variables
        try:
            right_mark_type = lane_marks.index(curr_lane_seg.right_mark_type)
        except:
            right_mark_type = lane_marks.index('NONE')
        try:
            left_mark_type  = lane_marks.index(curr_lane_seg.left_mark_type)
        except:
            left_mark_type = lane_marks.index('NONE')
        
        # intersection
        intersection = 1 if curr_lane_seg.is_intersection else 0
        # whether the lane is vehicle/bike/bus or pedestrian
        lane_semantic   = lane_types.index(curr_lane_seg.lane_type)

        # localized coordinates as ctrs
        left_ctrs.append((left_boundary[:-1] + left_boundary[1:]) / 2.0)
        right_ctrs.append((right_boundary[:-1] + right_boundary[1:]) / 2.0)
        ctrs.append((centerline[:-1] + centerline[1:]) / 2.0)

        # yaw rates as feats
        left_feats.append(left_boundary[1:] - left_boundary[:-1])
        right_feats.append(right_boundary[1:] - right_boundary[:-1])
        feats.append(centerline[1:] - centerline[:-1])

        # total nodes
        num_centerline_nodes     = len(centerline) - 1
        num_left_boundary_nodes  = len(left_boundary) - 1
        num_right_boundary_nodes = len(right_boundary) - 1

        # left boundary control variables
        left_mark_lst.append(left_mark_type * torch.ones(num_left_boundary_nodes, dtype = torch.float32))
        # right boundary control variables
        right_mark_lst.append(right_mark_type * torch.ones(num_right_boundary_nodes, dtype = torch.float32))   
        
        # add these two controls in the centerline layer
        intersect_lst.append(intersection * torch.ones(num_centerline_nodes, dtype = torch.float32))
        lane_semantic_lst.append(lane_semantic * torch.ones(num_centerline_nodes, dtype = torch.float32))
        
        polygon_ids.append(lane_id)


    # crosswalk features
    for crosswalk in avm.get_scenario_ped_crossings():
        edge1 = torch.from_numpy(crosswalk.edge1.xyz[:, :2]).float()
        edge2 = torch.from_numpy(crosswalk.edge2.xyz[:, :2]).float()
        start_position = (edge1[0] + edge2[0]) / 2
        end_position = (edge1[-1] + edge2[-1]) / 2

        # update the geometry of crosswalks
        if side_to_directed_lineseg((edge1[0] + edge1[-1]) / 2, start_position, end_position) == 'LEFT':
            left_boundary = edge1
            right_boundary = edge2
        else:
            left_boundary = edge2
            right_boundary = edge1

        # for interpolation
        num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() / 2.0) + 1

        centerline = torch.from_numpy(compute_midpoint_line(left_boundary.numpy(), 
                                                            right_boundary.numpy(), 
                                                            num_centerline_points)[0]).float()
                
        left_boundary  = torch.from_numpy(interp_arc(num_centerline_points, left_boundary.numpy())).float()
        right_boundary = torch.from_numpy(interp_arc(num_centerline_points, right_boundary.numpy())).float()

        # convert to local view
        right_boundary = torch.matmul(data['rot'], (right_boundary - data['orig']).T).T
        left_boundary  = torch.matmul(data['rot'], (left_boundary - data['orig']).T).T
        centerline = torch.matmul(data['rot'], (centerline - data['orig']).T).T

        # reversed localed views
        reversed_left_boundary  = torch.flip(right_boundary, dims = [0])
        reversed_right_boundary = torch.flip(left_boundary, dims = [0])
        reversed_centerline = torch.flip(centerline, dims = [0])
        
        # control-based variables
        right_mark_type = lane_marks.index('PEDESTRIAN')
        left_mark_type  = lane_marks.index('PEDESTRIAN')

        # intersection
        intersection    = 2
        # whether the lane is vehicle/bike/bus or pedestrian
        lane_semantic   = lane_types.index('PEDESTRIAN')

        # localized coordinates as ctrs
        left_ctrs.append((left_boundary[:-1] + left_boundary[1:]) / 2.0)
        left_ctrs.append((reversed_left_boundary[:-1] + reversed_left_boundary[1:]) / 2.0)

        right_ctrs.append((right_boundary[:-1] + right_boundary[1:]) / 2.0)
        right_ctrs.append((reversed_right_boundary[:-1] + reversed_right_boundary[1:]) / 2.0)

        ctrs.append((centerline[:-1] + centerline[1:]) / 2.0)
        ctrs.append((reversed_centerline[:-1] + reversed_centerline[1:]) / 2.0)

        # yaw rates as feats
        left_feats.append(left_boundary[1:] - left_boundary[:-1])
        left_feats.append(reversed_left_boundary[1:] - reversed_left_boundary[:-1])

        right_feats.append(right_boundary[1:] - right_boundary[:-1])
        right_feats.append(reversed_right_boundary[1:] - reversed_right_boundary[:-1])

        feats.append(centerline[1:] - centerline[:-1])
        feats.append(reversed_centerline[1:] - reversed_centerline[:-1])

        # total nodes
        num_crosswalk_nodes = len(centerline) - 1

        # left boundary control variables
        left_mark_lst.append(left_mark_type * torch.ones(num_crosswalk_nodes, dtype = torch.float32))
        left_mark_lst.append(left_mark_type * torch.ones(num_crosswalk_nodes, dtype = torch.float32))
        # right boundary control variables
        right_mark_lst.append(right_mark_type * torch.ones(num_crosswalk_nodes, dtype = torch.float32))
        right_mark_lst.append(right_mark_type * torch.ones(num_crosswalk_nodes, dtype = torch.float32))
        # add these two controls in the centerline layer
        intersect_lst.append(intersection * torch.ones(num_crosswalk_nodes, dtype = torch.float32))
        intersect_lst.append(intersection * torch.ones(num_crosswalk_nodes, dtype = torch.float32))
        # lane semantic variables
        lane_semantic_lst.append(lane_semantic * torch.ones(num_crosswalk_nodes, dtype = torch.float32))
        lane_semantic_lst.append(lane_semantic * torch.ones(num_crosswalk_nodes, dtype = torch.float32))

        polygon_ids.append(crosswalk.id)
        polygon_ids.append(crosswalk.id)

    # node idcs features for graph connectivity features it is a number of node counting process
    l_node_idx, r_node_idx, cl_node_idx = [], [], []
    l_count, r_count, cl_count = 0, 0, 0
    
    for ctr in ctrs:
        cl_node_idx.append(range(cl_count, cl_count + len(ctr)))
        cl_count += len(ctr)

    for l_ctr in left_ctrs:
        l_node_idx.append(range(l_count, l_count + len(l_ctr)))
        l_count += len(l_ctr)
        
    for r_ctr in right_ctrs:
        r_node_idx.append(range(r_count, r_count + len(r_ctr)))
        r_count += len(r_ctr)
    
    l_num_nodes = l_count
    r_num_nodes = r_count
    cl_num_nodes = cl_count

    # connectivity in logitudinal way
    l_pre, l_suc = get_logitudinal_adjacent_pairs(polygon_ids, avm, l_node_idx, total_lanes)
    r_pre, r_suc = get_logitudinal_adjacent_pairs(polygon_ids, avm, r_node_idx, total_lanes)
    pre, suc     = get_logitudinal_adjacent_pairs(polygon_ids, avm, cl_node_idx, total_lanes)

    # connectivity indices for graph dilations
    pre_pairs, suc_pairs, left_pairs, right_pairs = get_adjacent_pairs(lane_ids, avm)
    
    
    pre_pairs = torch.tensor(pre_pairs, dtype = torch.long)
    suc_pairs = torch.tensor(suc_pairs, dtype = torch.long)
    left_pairs = torch.tensor(left_pairs, dtype = torch.long)
    right_pairs = torch.tensor(right_pairs, dtype = torch.long)
    
    left_graph, right_graph, cl_graph = dict(), dict(), dict()
    cl_graph['ctrs'] = torch.cat(ctrs, 0)
    cl_graph['feats'] = torch.cat(feats, 0)
    cl_graph['num_nodes'] = cl_num_nodes
    cl_graph['lane_type'] = torch.cat(lane_semantic_lst, 0)
    cl_graph['intersect'] = torch.cat(intersect_lst, 0)

    cl_graph['pre'] = [pre]
    cl_graph['suc'] = [suc]
    # lane idcs list
    cl_lane_idcs = []
    for i, idcs in enumerate(cl_node_idx):
        cl_lane_idcs.append(i * torch.ones(len(idcs), dtype = torch.int64))

    cl_lane_idcs = np.concatenate(cl_lane_idcs, 0)
    cl_graph['lane_idcs']   = cl_lane_idcs
    cl_graph['pre_pairs']   = pre_pairs
    cl_graph['suc_pairs']   = suc_pairs
    cl_graph['left_pairs']  = left_pairs
    cl_graph['right_pairs'] = right_pairs

    ################# left graph features gathering ################# 
    left_graph['ctrs'] = torch.cat(left_ctrs, 0)
    left_graph['feats'] = torch.cat(left_feats, 0)
    left_graph['num_nodes'] = l_num_nodes
    left_graph['mark_type'] = torch.cat(left_mark_lst, 0)
    left_graph['pre'] = [l_pre]
    left_graph['suc'] = [l_suc]

    # left lane idcs list

    l_lane_idcs = []
    for i, idcs in enumerate(l_node_idx):
        l_lane_idcs.append(i * torch.ones(len(idcs), dtype = torch.int64))

    l_lane_idcs = torch.cat(l_lane_idcs, 0)
    left_graph['lane_idcs'] = l_lane_idcs

    left_graph['pre_pairs'] = pre_pairs
    left_graph['suc_pairs'] = suc_pairs
    left_graph['left_pairs'] = left_pairs
    left_graph['right_pairs'] = right_pairs
    
    ################# right graph features gathering ################# 
    right_graph['ctrs']      = torch.cat(right_ctrs, 0)
    right_graph['feats']     = torch.cat(right_feats, 0)
    right_graph['num_nodes'] = r_num_nodes    
    right_graph['mark_type'] = torch.cat(right_mark_lst, 0)
    right_graph['pre'] = [r_pre]
    right_graph['suc'] = [r_suc]
    
    # lane index list
    r_lane_idcs = []
    for i, idcs in enumerate(r_node_idx):
        r_lane_idcs.append(i * torch.ones(len(idcs), dtype = torch.int64))
        
    r_lane_idcs = torch.cat(r_lane_idcs, 0)
    right_graph['lane_idcs'] = r_lane_idcs

    right_graph['pre_pairs'] = pre_pairs
    right_graph['suc_pairs'] = suc_pairs
    right_graph['left_pairs'] = left_pairs
    right_graph['right_pairs'] = right_pairs

    for k1 in ['pre', 'suc']:
        for k2 in ['u', 'v']:
            left_graph[k1][0][k2] = torch.tensor(left_graph[k1][0][k2], dtype = torch.int64)

    for k1 in ['pre', 'suc']:
        for k2 in ['u', 'v']:
            right_graph[k1][0][k2] = torch.tensor(right_graph[k1][0][k2], dtype = torch.int64)

    for k1 in ['pre', 'suc']:
        for k2 in ['u', 'v']:
            cl_graph[k1][0][k2] = torch.tensor(cl_graph[k1][0][k2], dtype = torch.int64)

    data['lane_ids']    = lane_ids
    data['polygon_ids'] = polygon_ids
    data['cl_graph']    = cl_graph   
    data['left_graph']  = left_graph
    data['right_graph'] = right_graph

    for key in ['pre', 'suc']:
        left_graph[key]  += dilated_nbrs(left_graph[key][0], left_graph['num_nodes'], 6)
        right_graph[key] += dilated_nbrs(right_graph[key][0], right_graph['num_nodes'], 6)
        cl_graph[key]    += dilated_nbrs(cl_graph[key][0], cl_graph['num_nodes'], 6)
    
    return data
    
def get_logitudinal_adjacent_pairs(polygon_ids: List[str], 
                                   avm: ArgoverseStaticMap, 
                                   node_idcs: List[int],
                                   lane_limit: int):

    pre, suc = dict(), dict()

    for key in ['u', 'v']:
        pre[key], suc[key] = [], []

    for i, lane_id in enumerate(polygon_ids):
        # connectivity
        if i < lane_limit:
            lane         = avm.vector_lane_segments[lane_id]
            prev_ids     = lane.predecessors
            succ_ids     = lane.successors

            idcs = node_idcs[i]

            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if prev_ids is not None:
                for nbr_id in lane.predecessors:
                    if nbr_id in polygon_ids:
                        j = polygon_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])

            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            
            if succ_ids is not None:
                for nbr_id in lane.successors:
                    if nbr_id in polygon_ids:
                        j = polygon_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])
        else:
            idcs = node_idcs[i]
            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]

    return pre, suc

def get_adjacent_pairs(lane_ids: List[int], avm: ArgoverseStaticMap):

    pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []

    for i, lane_id in enumerate(lane_ids):
        
        lane = avm.vector_lane_segments[lane_id]
        # connectivity
        nbr_ids = lane.predecessors
        if nbr_ids is not None:
            for nbr_id in nbr_ids:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre_pairs.append([i, j])

        nbr_ids = lane.successors
        if nbr_ids is not None:
            for nbr_id in nbr_ids:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc_pairs.append([i, j])

        nbr_id = lane.left_neighbor_id
        if nbr_id is not None:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                left_pairs.append([i, j])

        nbr_id = lane.right_neighbor_id
        if nbr_id is not None:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                right_pairs.append([i, j])
                
    return pre_pairs, suc_pairs, left_pairs, right_pairs 
    

def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = torch.from_numpy(coo.row.astype(np.int64)).long()
        nbr['v'] = torch.from_numpy(coo.col.astype(np.int64)).long()
        nbrs.append(nbr)
    return nbrs

def preprocess(graph, cross_dist, cross_angle=None):
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    if len(graph['pre_pairs']) > 0:
        pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
        
    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    if len(graph['suc_pairs']) > 0:
        suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    out = dict()
    out['left'] = left
    out['right'] = right
    #out['idx'] = graph['idx']
    return out

def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def process_Argo2_dataset(root, split):
    
    processed_path = os.path.join(root, 'processed', split)
    os.makedirs(processed_path, exist_ok = True)
    
    cur_dir = os.path.join(root,  split)
    sub_dir = os.listdir(os.path.join(root,  split))
    
    for _, scenario_id in tqdm(enumerate(sub_dir)):
        # try:
        real_id = "scenario_" + scenario_id + ".parquet"

        log_map_dirpath = os.path.join(cur_dir, scenario_id)            # map absolute directory 
        log_data_dirpath = os.path.join(cur_dir, scenario_id, real_id)  # parquet absolute directory 

        log_map_dirpath = Path(log_map_dirpath)                          # pathlib that corresponds to log map directories 
        avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)          # argoverse map
        df = pd.read_parquet(log_data_dirpath, engine='pyarrow')

        if args.split == 'test':
            data = get_agent_features(df, historical_frames = 50, future_frames = 60, test = True)
        else:
            data = get_agent_features(df, historical_frames = 50, future_frames = 60, test = False)

        data['scenario_id'] = scenario_id
        data = get_graph_features(data, avm)
        out_cl    = preprocess(data['cl_graph'], 6)
        out_left  = preprocess(data['left_graph'], 6)
        out_right = preprocess(data['right_graph'], 6)
        

        data['cl_graph']['left']     = out_cl['left']
        data['cl_graph']['right']    = out_cl['right']
        data['left_graph']['left']   = out_left['left']
        data['left_graph']['right']  = out_left['right']
        data['right_graph']['left']  = out_right['left']
        data['right_graph']['right'] = out_right['right']

        data['scenario_id'] = scenario_id

        torch.save(data, os.path.join(processed_path, scenario_id + '.pt'))  
        # except:
        #     print(scenario_id)
        #     continue
                
class Argo2Dataset(Dataset):
    def __init__(self, root : str, split : str, preprocess: bool = True):
        self.dir = os.path.join(root, "processed", split)
        self.scenario_ids  = os.listdir(self.dir)

        self.preprocess = preprocess

    def __len__(self):
        if not self.preprocess:
            print("Preprocess first, please !")
            return 0
        
        return len(self.scenario_ids)
    
    def __getitem__(self, idx):
        if not self.preprocess:
            print("Preprocess first, please !")
            return None

        scenario_id = self.scenario_ids[idx]
        data = torch.load(os.path.join(self.dir, scenario_id))

        return data            

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type = str, default = '../../')
    parser.add_argument("--split", type = str)
    args = parser.parse_args()
    
    process_Argo2_dataset(root = args.root, split  = args.split)