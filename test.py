"""Evaluate model on test set and report results"""

import os
import sys
from tqdm import tqdm
import torch
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
)
from pandas import DataFrame

def test(config,
         config_id,
         device,
         model,
         model_path,
         test_metric,
         test_loader,
         out_path):

    device = torch.device(config["device"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=6)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=6)])
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_inputs, test_labels = (
                test_data["image"].to(device),
                test_data["label"].to(device),
            )
            roi_size = config["crop_shape"]
            sw_batch_size=4
            test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
            test_labels = [post_label(i) for i in decollate_batch(test_labels)]
            # print(test_outputs[0].shape)
            # print(test_labels[0].shape)
            # # break
            # Accumulate dice
            test_metric(y_pred=test_outputs, y=test_labels)
        # Total dice over test set
        test_dices = test_metric.aggregate()

        # Record metrics and compute mean over test set
        class_means = torch.mean(test_dices, dim=0)
        mean = torch.mean(test_dices)
        test_dices_df = DataFrame(test_dices.detach().cpu().numpy())
        # test_dices_df.to_csv(out_path)

    # Log best dice
    # print(f"All scores: {test_dices_df}")
    print(f"Average class scores: {class_means}")
    print(f"Average score overall: {mean}")


if __name__ == "__main__":
    test(*sys.argv[1:])