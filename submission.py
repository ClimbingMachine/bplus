import os

import torch
import torch.nn as nn
from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
from torch.utils.data import DataLoader
from tqdm import tqdm

from ArgoData.data_centerline import Argo2Dataset
from models.structure.banet import config, get_banet
from utils.helper import collate_fn

test_data = Argo2Dataset(root="../", split="test")

test_loader = DataLoader(
    test_data,
    batch_size=8,
    num_workers=config["val_workers"],
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
)

config, collate_fn, net, loss, post_process, opt = get_banet()
checkpoint = torch.load("./models/results/ganet/36.000.ckpt")
net.load_state_dict(checkpoint["state_dict"], strict=False)
m = nn.Softmax(dim=0)

# import sys


if __name__ == "__main__":

    final_predictions = dict()
    for i, test_data in tqdm(enumerate(test_loader)):
        # print(test_data.keys())
        _, predictions = net(test_data)

        for batch in range(len(predictions["reg"])):
            probabilities = (
                m(predictions["cls"][batch][0]).detach().cpu().numpy()
            )
            agt_prediction = (
                predictions["reg"][batch][0].detach().cpu().numpy()
            )
            final_predictions[test_data["scenario_id"][batch]] = {
                test_data["track_id"][batch]: tuple(
                    [agt_prediction, probabilities]
                )
            }

    sub = ChallengeSubmission(final_predictions)
    # print("size of the submission file: ", sys.getsizeof(sub))

    save_path = "./tests/"
    os.makedirs(save_path, exist_ok=True)
    sub.to_parquet(os.path.join(save_path, "test.parquet"))

    final_predictions = dict()
