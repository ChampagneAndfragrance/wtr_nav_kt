import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
from diffuser.datasets.multipath import BallWheelchairJointDataset
from diffuser.graders.traj_graders import joint_traj_grader
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.temporal_film import ConditionalUnet1D
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from global_utils.config_loader import config_loader
import argparse
import random
import time
from metrics import dtw, energy, icp, smoothness
import matplotlib.image as mpimg


# INFO: define some global variables
global_device_name = "cuda"
global_device = torch.device("cuda")
mse_loss_func = torch.nn.MSELoss()


def wheelchair_diffusion_train(config):

    # INFO: load the train, valid and test dataset
    diffusion_train_dataset = BallWheelchairJointDataset(folder_path=config["train_set"], 
                                         horizon=config["horizon"],
                                         use_padding=config["use_padding"] ,
                                         max_path_length=config["max_path_length"],
                                         dataset_type = "pixel",
                                         include_start_detection = True,
                                         condition_path = True,
                                         max_trajectory_length=config["max_trajectory_length"],
                                         max_detection_num = config["max_detection_num"],
                                         train_mode=config["train_option"],
                                         condition_mode = config["condition_mode"],
                                         prediction_mode = config["pred_mode"],)
    diffusion_valid_dataset = BallWheelchairJointDataset(folder_path=config["valid_set"], 
                                         horizon=config["horizon"],
                                         use_padding=config["use_padding"] ,
                                         max_path_length=config["max_path_length"],
                                         dataset_type = "pixel",
                                         include_start_detection = True,
                                         condition_path = True,
                                         max_trajectory_length=config["max_trajectory_length"],
                                         max_detection_num = config["max_detection_num"],
                                         train_mode=config["train_option"],
                                         condition_mode = config["condition_mode"],
                                         prediction_mode = config["pred_mode"],)

    # INFO: different conditions need different collection functions
    if config["train_option"] == 'ball_chair':
        train_collate_fn = diffusion_train_dataset.ball_chair_collate_fn
        valid_collate_fn = diffusion_valid_dataset.ball_chair_collate_fn
        global_cond_dim = 0
        lstm_out_dim=128

        if config["pred_mode"] == "2d":
            global_feature_num = 5
            lstm_dim = 5
            observation_dim = 4
            transition_dim = 4
        elif config["pred_mode"] == "3d":
            global_feature_num = 6
            lstm_dim = 6
            observation_dim = 5
            transition_dim = 5
        else:
            raise NotImplementedError

    elif config["train_option"] == 'ball':
        train_collate_fn = diffusion_train_dataset.ball_collate_fn
        valid_collate_fn = diffusion_valid_dataset.ball_collate_fn
        lstm_dim = 3
        observation_dim = 2
        transition_dim = 2
        global_cond_dim = 0
        global_feature_num = 5
        lstm_out_dim=128
    elif config["train_option"] == 'chair':
        train_collate_fn = diffusion_train_dataset.chair_collate_fn
        valid_collate_fn = diffusion_valid_dataset.chair_collate_fn
        lstm_out_dim=128
        if config["condition_mode"] == "pre":
            if config["pred_mode"] == "2d":
                lstm_dim = 5
                transition_dim = 2
                observation_dim = 2
                global_cond_dim = 0
                global_feature_num = 5
            elif config["pred_mode"] == "3d":
                lstm_dim = 6
                transition_dim = 3
                observation_dim = 3
                global_cond_dim = 0
                global_feature_num = 5
            else:
                raise NotImplementedError
        elif config["condition_mode"] == "post":
            lstm_dim = 3
            global_cond_dim = 32
            if config["pred_mode"] == "2d":
                transition_dim = 2
                observation_dim = 2
                global_feature_num = 5
            elif config["pred_mode"] == "3d":
                transition_dim = 3
                observation_dim = 3
                global_feature_num = 6
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # INFO: define folders to save model and logs   
    path = f'./ICRA_2025_logs/{config["folder_name"]}/benchmark_{config["train_option"]}_{config["condition_mode"]}_{config["pred_mode"]}'
    os.makedirs(path, exist_ok=True)
    writer = SummaryWriter(path+"/logs")

    # INFO: ball-chair diffusions
    diffusion_net = ConditionalUnet1D(transition_dim=transition_dim, horizon=config["horizon"], global_cond_dim=global_cond_dim, lstm_dim=lstm_dim, lstm_out_dim=lstm_out_dim, global_feature_num=global_feature_num)
    diffusion_model = GaussianDiffusion(model=diffusion_net, horizon=config["horizon"], observation_dim=observation_dim, action_dim=0, n_timesteps=config["n_timesteps"], predict_epsilon=False)
    diffusion_optimizer = Adam(diffusion_model.parameters(), lr=0.0001, weight_decay=0.0000)
    diffusion_train_dataloader = (torch.utils.data.DataLoader(diffusion_train_dataset, batch_size=32, num_workers=0, shuffle=True, pin_memory=False, collate_fn=train_collate_fn))
    diffusion_valid_dataloader = (torch.utils.data.DataLoader(diffusion_valid_dataset, batch_size=32, num_workers=0, shuffle=True, pin_memory=False, collate_fn=valid_collate_fn))

    # INFO: set training and validation epochs
    epoch_num = 1000
    validation_epoch_period = 2
    min_valid_loss = np.inf

    # INFO: set random seed for diffusion training
    seed = 2
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    for ep_i in range(epoch_num):
        diffusion_train_losses = []

        # INFO: training with samples in the training dataloader
        for traj_ep, agents_info_batch in tqdm(enumerate(diffusion_train_dataloader)):
            # INFO: This part is to train the diffusion model with the ground truth trajectories
            diffusion_loss, infos = diffusion_model.loss(*agents_info_batch)
            diffusion_optimizer.zero_grad()
            diffusion_loss.backward()
            torch.nn.utils.clip_grad_norm(diffusion_model.parameters(), 0.1)
            diffusion_optimizer.step()
            diffusion_train_losses.append(diffusion_loss)

        # INFO: validating with samples in the validation dataloader
        if ep_i % validation_epoch_period == 0:
            diffusion_valid_losses = []
            with torch.no_grad():
                for traj_ep, agents_info_batch in tqdm(enumerate(diffusion_valid_dataloader)):
                    # INFO: This part is to validate the training process
                    diffusion_loss, infos = diffusion_model.loss(*agents_info_batch)
                    diffusion_valid_losses.append(diffusion_loss)                
        if torch.Tensor(diffusion_valid_losses).mean().item() < min_valid_loss:
            min_valid_loss = torch.Tensor(diffusion_valid_losses).mean().item()
            torch.save(diffusion_model, path+f"/diffusion_epoch_{ep_i}.pth")
            torch.save(diffusion_model, path+f"/diffusion_final.pth")

        # INFO: write loss in and monitor with tensorboard
        writer.add_scalars('loss', {'train_loss': torch.Tensor(diffusion_train_losses).mean().item(), 'valid_loss': torch.Tensor(diffusion_valid_losses).mean().item()}, ep_i)
        print("Trajectory eps: ", ep_i)
        print("min_valid_loss: ", min_valid_loss)

def red_diffusion_eval(config):

    # INFO: load the test data into dataloader
    diffusion_test_dataset = BallWheelchairJointDataset(folder_path=config["test_set"],
                                         horizon=config["horizon"],
                                         use_padding=config["use_padding"] ,
                                         max_path_length=config["max_path_length"],
                                         dataset_type = "pixel",
                                         include_start_detection = True,
                                         condition_path = True,
                                         max_trajectory_length=config["max_trajectory_length"],
                                         max_detection_num = config["max_detection_num"],
                                         train_mode=config["train_option"],
                                         condition_mode = config["condition_mode"],
                                         prediction_mode = config["pred_mode"],)
    if config["train_option"] == 'ball_chair':
        test_collate_fn = diffusion_test_dataset.ball_chair_collate_fn_repeat
        sample_type = "original"
    elif config["train_option"] == 'ball':
        test_collate_fn = diffusion_test_dataset.ball_collate_fn_repeat
        sample_type = "original"
    elif config["train_option"] == 'chair':
        test_collate_fn = diffusion_test_dataset.chair_collate_fn_repeat
        sample_type = "original"
    else:
        raise NotImplementedError
    diffusion_test_dataloader = (torch.utils.data.DataLoader(diffusion_test_dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False, collate_fn=test_collate_fn))
    
    # INFO: load the saved diffusion model
    path = f'./ICRA_2025_logs/{config["folder_name"]}/benchmark_{config["train_option"]}_{config["condition_mode"]}_{config["pred_mode"]}'
    diffusion_model = torch.load(path +"/diffusion_final.pth")
    
    # INFO: metrics we need to collect
    rmses = []
    distances = []
    icp_rmses = []
    energy_consumeds = []
    jerks = []

    # INFO: draw samples from current diffusion model
    for traj_ep, agents_info_batch in tqdm(enumerate(diffusion_test_dataloader)):
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # INFO: sample from the diffusion model
        start_time = time.time()
        sample = diffusion_model.conditional_sample(global_cond=agents_info_batch[1], cond=agents_info_batch[2], sample_type=sample_type)
        end_time = time.time()
        print("Diffusion takes %f seconds." % (end_time-start_time))
        sample = diffusion_test_dataset.unnormalize(sample.detach())
        if config["train_option"] != "dynamic":
            detections = diffusion_test_dataset.unnormalize(agents_info_batch[1]["detections"].data[:,1:])
        gt_sample = diffusion_test_dataset.unnormalize(agents_info_batch[0])
        # INFO: plot 1) previous ball+chair traj in image space 2) predicted ball+chair traj in image space 3) GT trajs
        if config["pred_mode"] == "2d":
            if config["train_option"] == "ball_chair":
                chair_traj = sample[0,:,2:]
                chair_traj_gt = gt_sample[0,:,2:]
            else:
                chair_traj = sample[0,:]
                chair_traj_gt = gt_sample[0,:]
        elif config["pred_mode"] == "3d":
            if config["train_option"] == "ball_chair":
                chair_traj = sample[0,:,2:4]
                chair_traj_gt = gt_sample[0,:,2:4]
            else:
                chair_traj = sample[0,:,:2]
                chair_traj_gt = gt_sample[0,:,:2]
        else:
            raise NotImplementedError
        chair_traj_np = to_numpy(chair_traj)
        chair_traj_gt_np = to_numpy(chair_traj_gt)
        if config["pred_mode"] == "2d":
            projection_matrix = np.load("./parameters/constant/projection_matrix.npy")
            chair_traj_np_cartesian = convert_wheelchair_2d_to_3d(chair_traj_np, projection_matrix)
            chair_traj_gt_np_cartesian = convert_wheelchair_2d_to_3d(chair_traj_gt_np, projection_matrix)
        elif config["pred_mode"] == "3d":
            chair_traj_np_cartesian = chair_traj_np
            chair_traj_gt_np_cartesian = chair_traj_gt_np
        rmse = np.sqrt(np.square(chair_traj_gt_np_cartesian - chair_traj_np_cartesian).mean())
        distance, _ = dtw.dtw(chair_traj_gt_np_cartesian, chair_traj_np_cartesian)
        icp_rmse = icp.icp_error(chair_traj_gt_np_cartesian, chair_traj_np_cartesian)
        energy_consumed = energy.compute_energy(chair_traj_gt_np_cartesian, chair_traj_np_cartesian)
        jerk = smoothness.compute_jerk(chair_traj_np_cartesian)
        rmses.append(rmse)
        distances.append(distance)
        icp_rmses.append(icp_rmse)
        energy_consumeds.append(energy_consumed)
        jerks.append(jerk)
    return np.array(rmses).mean(), np.array(distances).mean(), np.array(icp_rmses).mean(), np.array(energy_consumeds).mean(), np.array(jerks).mean()

def set_padding(config):
    if config["train_option"] == 'ball_chair':
        config["use_padding"] = False
    elif config["train_option"] == 'ball':
        config["use_padding"] =True
    elif config["train_option"] == 'chair':
        config["use_padding"] =True
    elif config["train_option"] == 'dynamic':
        config["use_padding"] =True
    else:
        raise NotImplementedError

def convert_wheelchair_2d_to_3d(image_points, projection_matrix):
    """
    Project 2D image points to 3D world coordinates (z=0) using a given projection matrix.

    Args:
    - image_points (np.array): Nx2 array containing 2D image coordinates.
    - projection_matrix (np.array): 3x4 projection matrix combining camera intrinsics and extrinsics.

    Returns:
    - world_points (np.array): Nx3 array containing 3D world coordinates.
    """

    # Get homography matrix from projection matrix (for z=0)
    homography = projection_matrix[:, [0, 1, 3]]  # Take columns 0, 1, and 3 (ignoring z)

    # Calculate the inverse homography matrix
    homography_inverse = np.linalg.inv(homography)

    # Convert 2D image points to homogeneous coordinates (add an extra dimension)
    num_points = image_points.shape[0]
    
    homogeneous_image_points = np.hstack((image_points, np.ones((num_points, 1))))

    # Compute the 3D world points in homogeneous coordinates
    world_homogeneous = homography_inverse @ homogeneous_image_points.T  # 3xN

    # Normalize by the last row to convert to 3D world coordinates
    world_points = (world_homogeneous / world_homogeneous[2, :]).T  # Nx3

    # Set Z = 0 (currently it is normalized so it is set to 1)
    world_points = world_points[:, :2]

    return world_points
    
def to_numpy(array):
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

def check_config(config):
    # train_option = ball_chair can only use condition_mode=pre;   
    if config["train_option"] == "ball_chair" and config["condition_mode"] != "pre":
        return False    
    
    #ball can only use pre 2d
    if config["train_option"] == "ball":
        if config["condition_mode"] != "pre" or config["pred_mode"] != "2d":
            return False
    return True

def run(config):
    if config["stage"] == "train":
        # INFO: train the diffusion model
        wheelchair_diffusion_train(config)
    elif config["stage"] == "eval":
        # INFO: evaluate the diffusion model
        mse, distance, icp, energy, jerk = red_diffusion_eval(config)
        metrics = np.array([mse, distance, icp, energy, jerk])
        np.savetxt(f'./ICRA_2025_logs/{config["folder_name"]}/benchmark_{config["train_option"]}_{config["condition_mode"]}_{config["pred_mode"]}/metrics.txt', metrics, delimiter=',')      
    else:
        raise NotImplementedError

def set_config(config, to, cm, pm, tt):
    config["train_option"] = to
    config["condition_mode"] = cm
    config["pred_mode"] = pm
    config["stage"] = tt
    set_padding(config)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":  
    config = config_loader(path="./config/diffusion_config/chair_diffusion_train.yaml")

    # INFO: Specify the full training/testing config list
    # train_options = ["chair", "ball_chair", "ball"]
    # condition_modes = ["post", "pre"]
    # pred_modes = ["2d", "3d"]
    # train_or_test = ["eval"]

    # INFO: Specify training/testing config
    train_options = ["chair"]
    condition_modes = ["post"]
    pred_modes = ["2d"]
    train_or_test = ["train"]

    for to in train_options:
        for cm in condition_modes:
            for pm in pred_modes:
                for tt in train_or_test:
                    set_config(config, to, cm, pm, tt)
                    if check_config(config):
                        run(config)
         