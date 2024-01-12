import os, torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import sparse
from pathlib import Path
from typing import Dict, List
from av2.map.map_api import ArgoverseStaticMap
from torch.utils.data import Dataset

# global parameters 

lane_mark_dict = dict()
lane_mark_dict['DASH_SOLID_YELLOW'] = 0
lane_mark_dict['DASH_SOLID_WHITE'] = 1
lane_mark_dict['DASHED_YELLOW'] = 2
lane_mark_dict['DASHED_WHITE'] = 3
lane_mark_dict['DOUBLE_DASH_YELLOW'] = 2
lane_mark_dict['DOUBLE_DASH_WHITE'] = 3
lane_mark_dict['DOUBLE_SOLID_YELLOW'] = 4
lane_mark_dict['DOUBLE_SOLID_WHITE'] = 5
lane_mark_dict['SOLID_YELLOW'] = 4
lane_mark_dict['SOLID_WHITE'] = 5
lane_mark_dict['SOLID_DASH_YELLOW'] = 6
lane_mark_dict['SOLID_DASH_WHITE'] = 7
lane_mark_dict['SOLID_BLUE'] = 8
lane_mark_dict['NONE'] = 9
lane_mark_dict['BLUE'] = 10

# LANE TYPE DICTIONARY
lane_type = dict()
lane_type['VEHICLE'] = 0
lane_type['BIKE'] = 1
lane_type['BUS'] = 2

agent_type = dict()
agent_type['vehicle'] = 1
agent_type['pedestrian'] = 2
agent_type['motorcyclist'] = 3
agent_type['cyclist'] = 4
agent_type['bus'] = 5
agent_type['unknown'] = 6


def read_argo2_data(log_data_dir_path: str, is_test = False):
    
    df = pd.read_parquet(log_data_dir_path, engine='pyarrow')
    trajs = df[['position_x', 'position_y']].to_numpy()                     # traj instances
    yaws  = df[['heading']].to_numpy()                                      # traj features (heading)
    velocity =  df[['velocity_x', 'velocity_y']].to_numpy()                     # traj instances
    types = df['object_type'].to_numpy()
    steps = df['timestep'].to_numpy()                                       # timestamps
    objs  = df.groupby(['track_id', 'object_type']).groups
    keys  = list(objs.keys())
    agent_id = [x[0] for x in keys]
    agt_id  = df['focal_track_id'][0]
    agt_idx = agent_id.index(agt_id)

    idcs = objs[keys[agt_idx]]                                   # focal agent index

    agt_traj = trajs[idcs]                                       # focal agent trajectoy
    agt_step = steps[idcs]                                       # focal agent steps
    agt_type = agent_type[types[idcs][0]] if types[idcs][0] in agent_type.keys() else 7  # focal agent type
    agt_yaw  = yaws[idcs]
    agt_vel  = velocity[idcs]

    del keys[agt_idx]
    ctx_trajs, ctx_steps, ctx_type, ctx_yaw = [], [], [], []
    ctx_velocity = []
    for key in keys:
        idcs = objs[key]
        ctx_trajs.append(trajs[idcs])
        ctx_steps.append(steps[idcs])
        ctx_velocity.append(velocity[idcs])
        ctx_yaw.append(yaws[idcs])

        ctx_type.append(agent_type[types[idcs][0]] if types[idcs][0] in agent_type.keys() else 7)

    data = dict()
    data['trajs'] = [agt_traj] + ctx_trajs
    data['steps'] = [agt_step] + ctx_steps
    data['types'] = [agt_type] + ctx_type
    data['velocity'] = [agt_vel] + ctx_velocity
    data['yaw']   = [agt_yaw] + ctx_yaw

    data['track_id'] = agt_id

    #return data
    return get_obj_feats(data, is_test)


def get_obj_feats(data: Dict, is_test = False) -> Dict:
    assert 49 in data['steps'][0]
    idx = np.where(data['steps'][0] == 49)

    theta = np.pi - data['yaw'][0][idx][0, 0]                                  # get last theta and rotate by PI
    origin = data['trajs'][0][idx].reshape(-1)                                 # origin records
    rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],                    # rotation matrix
                           [np.sin(theta),  np.cos(theta)]], np.float32)
    
    feats, ctrs, gt_preds, has_preds, hist, local = [], [], [], [], [], []

    for traj, type, step, vel, yaw in zip(data['trajs'],
                                          data['types'], 
                                          data['steps'], 
                                          data['velocity'], 
                                          data['yaw']):
        if 49 not in step:
            continue
        
        gt_pred  = np.zeros((60, 2), np.float32)
        loc_pred = np.zeros((60, 2), np.float32)                                  # GaNet specific parameter
        has_pred = np.zeros(60, bool)
        # future trajectory encoding
        if not is_test:
            future_mask = np.logical_and(step >= 50, step < 110)
            post_step = step[future_mask] - 50
            post_traj = traj[future_mask]                                  
            gt_pred[post_step]  = post_traj                                       # ground truth prediction
            loc_pred[post_step] = np.matmul(rotate_mat, (post_traj - origin).T).T # ground truth local prediction
            has_pred[post_step] = 1                                               # ground truth indicator

        # historical features preparation
        # curr_feat = np.zeros((50, 3), np.float32)                       # curr feature vector
        # history   = np.zeros((50, 2), np.float32)                       # curr feature vector
        
        obs_mask = step < 50                                 # historical trajectory indicator
        step = step[obs_mask]                                # time step indicator
        traj = traj[obs_mask]                                # historical trajectory 
        vel  = vel[obs_mask] 
        yaw  = yaw[obs_mask]
        
        idcs = step.argsort()
        step = step[idcs]
        traj = traj[idcs]
        vel  = vel[idcs]
        yaw  = yaw[idcs]
        
        for i in range(len(step)):
            if step[i] == 49 - (len(step) - 1) + i:
                break

        traj = traj[i:]
        step = step[i:].astype(int)
        vel  = vel[i:]
        yaw  = yaw[i:]

        feat = np.zeros((50, 7), np.float32)
        feat[step, :2] = np.matmul(rotate_mat, (traj - origin).T).T
        feat[step, 2:4] = np.matmul(rotate_mat, vel.T).T
        feat[step, 4:5] = np.sin(yaw + theta)
        feat[step, 5:6] = np.cos(yaw + theta)
        feat[step, 6] = 1

        hist.append(feat[:, :2].copy())
        ctrs.append(feat[-1, :2].copy())

        feat[1:, :2] -= feat[:-1, :2]
        feat[step[0], :2] = 0
        feats.append(feat)
        gt_preds.append(gt_pred)
        has_preds.append(has_pred)
        local.append(loc_pred)
        
    local = np.asarray(local, np.float32)
    hist  = np.asarray(hist, np.float32)                                   # (local view) historical trajectory instance
    feats = np.asarray(feats, np.float32)                                  # (local view) [heading, velocity_x, velocity_y]
    ctrs = np.asarray(ctrs, np.float32)                                    # last timestep features
    gt_preds = np.asarray(gt_preds, np.float32)                            # groundtruth prediction instances
    has_preds = np.asarray(has_preds, bool)                                # valid 1 or not 0
    
    data['hist']  = hist
    data['features'] = feats
    data['ctrs'] = ctrs
    data['orig'] = origin
    data['theta'] = theta
    data['rot'] = rotate_mat
    data['gt_preds'] = gt_preds
    data['has_preds'] = has_preds
    data['loc_preds'] = local

    return data
    
def get_graph_features(data: Dict, avm: ArgoverseStaticMap):
    lane_ids = list(avm.vector_lane_segments.keys())
    # backups  = []
    l_ctrs, r_ctrs, ctrs, r_feats, l_feats, feats = [], [], [], [], [], []
    l_mark, r_mark = [], []
    intersect_lst, lane_semantic_lst = [], []
    
    # first iteration to get number of nodes
    for i, lane_id in enumerate(lane_ids):

        # current lane_segment
        curr_ln = avm.vector_lane_segments[lane_id]

        # boundaries and markings 
        right_boundary  = curr_ln.right_lane_marking.polyline[:, :2]
        left_boundary   = curr_ln.left_lane_marking.polyline[:, :2]
        
        cls = avm.get_lane_segment_centerline(lane_id)[:, :2]

        # convert to local view
        right_line = np.matmul(data['rot'], (right_boundary - data['orig']).T).T
        left_line  = np.matmul(data['rot'], (left_boundary - data['orig']).T).T
        centerline = np.matmul(data['rot'], (cls - data['orig']).T).T

        # control-based variables
        try:
            right_mark_type = lane_mark_dict[curr_ln.right_mark_type]
        except:
            right_mark_type = lane_mark_dict['NONE']
        try:
            left_mark_type  = lane_mark_dict[curr_ln.left_mark_type]
        except:
            left_mark_type = lane_mark_dict['NONE']

        intersect       = 1 if curr_ln.is_intersection else 0
        lane_semantic   = lane_type[curr_ln.lane_type]

        # local centerlines as feats
        l_ctrs.append(np.asarray((left_line[:-1] + left_line[1:]) / 2.0, np.float32))
        r_ctrs.append(np.asarray((right_line[:-1] + right_line[1:]) / 2.0, np.float32))
        ctrs.append(np.asarray((centerline[:-1] + centerline[1:]) / 2.0, np.float32))

        # yaw rates as feats
        l_feats.append(np.asarray(left_line[1:] - left_line[:-1], np.float32))
        r_feats.append(np.asarray(right_line[1:] - right_line[:-1], np.float32))
        feats.append(np.asarray(centerline[1:] - centerline[:-1], np.float32))

        # number of left and right nodes:

        num_nodes_per_lane   = len(centerline) - 1
        
        # left boundary control variables
        l_mark.append(left_mark_type * np.ones(num_nodes_per_lane, np.float32))
        # right boundary control variables
        r_mark.append(right_mark_type * np.ones(num_nodes_per_lane, np.float32))    
        
        # add these two controls in the centerline layer
        intersect_lst.append(intersect * np.ones(num_nodes_per_lane, np.float32))
        lane_semantic_lst.append(lane_semantic * np.ones(num_nodes_per_lane, np.float32))
    
    # node idcs features for graph connectivity features it is a number of node counting process
    l_node_idx, r_node_idx, cl_node_idx = [], [], []
    l_count, r_count, cl_count = 0, 0, 0
    
    for ctr in ctrs:
        cl_node_idx.append(range(cl_count, cl_count + len(ctr)))
        cl_count += len(ctr)

    for l_ctr in l_ctrs:
        l_node_idx.append(range(l_count, l_count + len(l_ctr)))
        l_count += len(l_ctr)
        
    for r_ctr in r_ctrs:
        r_node_idx.append(range(r_count, r_count + len(r_ctr)))
        r_count += len(r_ctr)
    
    l_num_nodes = l_count
    r_num_nodes = r_count
    cl_num_nodes = cl_count


    # connectivity in logitudinal way
    l_pre, l_suc = get_logitudinal_adjacent_pairs(lane_ids, avm, l_node_idx)
    r_pre, r_suc = get_logitudinal_adjacent_pairs(lane_ids, avm, r_node_idx)
    pre, suc     = get_logitudinal_adjacent_pairs(lane_ids, avm, cl_node_idx)
    
    # connectivity indices for graph dilations
    pre_pairs, suc_pairs, left_pairs, right_pairs = get_adjacent_pairs(lane_ids, avm)
    
    
    pre_pairs = np.asarray(pre_pairs, np.int64)
    suc_pairs = np.asarray(suc_pairs, np.int64)
    left_pairs = np.asarray(left_pairs, np.int64)
    right_pairs = np.asarray(right_pairs, np.int64)
    
    left_graph, right_graph, cl_graph = dict(), dict(), dict()

    ################# centerline graph features gathering ################# 
    cl_graph['ctrs'] = np.concatenate(ctrs, 0)
    cl_graph['feats'] = np.concatenate(feats, 0)
    cl_graph['num_nodes'] = cl_num_nodes
    cl_graph['lane_type'] = np.concatenate(lane_semantic_lst, 0)
    cl_graph['intersect'] = np.concatenate(intersect_lst, 0)
    cl_graph['left_mark'] = np.concatenate(l_mark, 0)
    cl_graph['right_mark']= np.concatenate(r_mark, 0)

    cl_graph['pre'] = [pre]
    cl_graph['suc'] = [suc]

    # lane idcs list
    cl_lane_idcs = []
    for i, idcs in enumerate(cl_node_idx):
        cl_lane_idcs.append(i * np.ones(len(idcs), np.int64))
    
    cl_lane_idcs = np.concatenate(cl_lane_idcs, 0)
    cl_graph['lane_idcs']   = cl_lane_idcs
    cl_graph['pre_pairs']   = pre_pairs
    cl_graph['suc_pairs']   = suc_pairs
    cl_graph['left_pairs']  = left_pairs
    cl_graph['right_pairs'] = right_pairs

    ################# left graph features gathering ################# 
    left_graph['ctrs'] = np.concatenate(l_ctrs, 0)
    left_graph['feats'] = np.concatenate(l_feats, 0)
    left_graph['num_nodes'] = l_num_nodes
    # left_graph['mark_type'] = np.concatenate(l_mark, 0)
    left_graph['pre'] = [l_pre]
    left_graph['suc'] = [l_suc]
    
    # left lane idcs list
    l_lane_idcs = []
    for i, idcs in enumerate(l_node_idx):
        l_lane_idcs.append(i * np.ones(len(idcs), np.int64))
    
    l_lane_idcs = np.concatenate(l_lane_idcs, 0)
    left_graph['lane_idcs']   = l_lane_idcs
    left_graph['pre_pairs']   = pre_pairs
    left_graph['suc_pairs']   = suc_pairs
    left_graph['left_pairs']  = left_pairs
    left_graph['right_pairs'] = right_pairs
    

    ################# right graph features gathering ################# 
    right_graph['ctrs']      = np.concatenate(r_ctrs, 0)
    right_graph['feats']     = np.concatenate(r_feats, 0)
    right_graph['num_nodes'] = r_num_nodes
    # right_graph['mark_type'] = np.concatenate(r_mark, 0)
    right_graph['pre'] = [r_pre]
    right_graph['suc'] = [r_suc]
    
    # lane index list
    r_lane_idcs = []
    for i, idcs in enumerate(r_node_idx):
        r_lane_idcs.append(i * np.ones(len(idcs), np.int64))
    r_lane_idcs = np.concatenate(r_lane_idcs, 0)
    right_graph['lane_idcs'] = r_lane_idcs

    right_graph['pre_pairs'] = pre_pairs
    right_graph['suc_pairs'] = suc_pairs
    right_graph['left_pairs'] = left_pairs
    right_graph['right_pairs'] = right_pairs


    for k1 in ['pre', 'suc']:
        for k2 in ['u', 'v']:
            left_graph[k1][0][k2] = np.asarray(left_graph[k1][0][k2], np.int64)

    for k1 in ['pre', 'suc']:
        for k2 in ['u', 'v']:
            right_graph[k1][0][k2] = np.asarray(right_graph[k1][0][k2], np.int64)

    for k1 in ['pre', 'suc']:
        for k2 in ['u', 'v']:
            cl_graph[k1][0][k2] = np.asarray(cl_graph[k1][0][k2], np.int64)


    data['lane_ids']    = lane_ids
    data['cl_graph']    = cl_graph   
    data['left_graph']  = left_graph
    data['right_graph'] = right_graph
    
    for key in ['pre', 'suc']:
        left_graph[key]  += dilated_nbrs(left_graph[key][0], left_graph['num_nodes'], 6)
        right_graph[key] += dilated_nbrs(right_graph[key][0], right_graph['num_nodes'], 6)
        cl_graph[key] += dilated_nbrs(cl_graph[key][0], cl_graph['num_nodes'], 6)
    
    return to_long(from_numpy(data))
    
    
def get_logitudinal_adjacent_pairs(lane_ids: List, avm: ArgoverseStaticMap, node_idcs: List):

    pre, suc = dict(), dict()

    for key in ['u', 'v']:
        pre[key], suc[key] = [], []

    for i, lane_id in enumerate(lane_ids):
        # connectivity
        lane         = avm.vector_lane_segments[lane_id]
        prev_ids     = lane.predecessors
        succ_ids     = lane.successors

        idcs = node_idcs[i]

        pre['u'] += idcs[1:]
        pre['v'] += idcs[:-1]
        if prev_ids is not None:
            for nbr_id in lane.predecessors:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre['u'].append(idcs[0])
                    pre['v'].append(node_idcs[j][-1])

        suc['u'] += idcs[:-1]
        suc['v'] += idcs[1:]
        
        if succ_ids is not None:
            for nbr_id in lane.successors:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc['u'].append(idcs[-1])
                    suc['v'].append(node_idcs[j][0])
    return pre, suc

def get_adjacent_pairs(lane_ids: Dict, avm: ArgoverseStaticMap):

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
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
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
        if cross_angle is not None:
            right_dist[hi[right_mask], wi[right_mask]] = 1e6

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
        
        if args.split == 'test':
            data = read_argo2_data(log_data_dirpath, True)
        else:
            data = read_argo2_data(log_data_dirpath, False)
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
    def __init__(self, root : str, split : str, preprocess = True):
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
