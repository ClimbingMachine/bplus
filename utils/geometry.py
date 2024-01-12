import torch

def side_to_directed_lineseg(
        query_point: torch.Tensor,
        start_point: torch.Tensor,
        end_point: torch.Tensor) -> str:
    cond = ((end_point[0] - start_point[0]) * (query_point[1] - start_point[1]) -
            (end_point[1] - start_point[1]) * (query_point[0] - start_point[0]))
    if cond > 0:
        return 'LEFT'
    elif cond < 0:
        return 'RIGHT'
    else:
        return 'CENTER'