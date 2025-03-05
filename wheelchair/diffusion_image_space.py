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


def red_diffusion_train(config):
    """ Collect demonstrations for the homogeneous gnn where we assume all agents are the same. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """

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
    elif config["train_option"] == 'dynamic':
        train_collate_fn = diffusion_train_dataset.dynamic_collate_fn
        valid_collate_fn = diffusion_valid_dataset.dynamic_collate_fn
        lstm_dim = 0
        transition_dim = 2
        observation_dim = 2
        global_cond_dim = 0
        global_feature_num = 5
        lstm_out_dim = 0
    else:
        raise NotImplementedError
    # INFO: define folders to save model and logs   
    path = f'./CoRL_2024_logs/{config["folder_name"]}/benchmark_{config["train_option"]}_{config["condition_mode"]}_{config["pred_mode"]}'
    os.makedirs(path, exist_ok=True)
    writer = SummaryWriter(path+"/logs")

    # INFO: ball-chair diffusions
    diffusion_net = ConditionalUnet1D(transition_dim=transition_dim, horizon=config["horizon"], global_cond_dim=global_cond_dim, lstm_dim=lstm_dim, lstm_out_dim=lstm_out_dim, global_feature_num=global_feature_num)
    diffusion_model = GaussianDiffusion(model=diffusion_net, horizon=config["horizon"], observation_dim=observation_dim, action_dim=0, n_timesteps=config["n_timesteps"], predict_epsilon=False)
    
    # diffusion_optimizer = Adam(diffusion_model.parameters(), lr=0.0002, weight_decay=0.0005)
    diffusion_optimizer = Adam(diffusion_model.parameters(), lr=0.0001, weight_decay=0.0000)

    diffusion_train_dataloader = (torch.utils.data.DataLoader(diffusion_train_dataset, batch_size=32, num_workers=0, shuffle=True, pin_memory=False, collate_fn=train_collate_fn))
    diffusion_valid_dataloader = (torch.utils.data.DataLoader(diffusion_valid_dataset, batch_size=32, num_workers=0, shuffle=True, pin_memory=False, collate_fn=valid_collate_fn))

    dataloader_init = True

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
        
        for traj_ep, batches_seqLen_agentLocations in tqdm(enumerate(diffusion_train_dataloader)):
            # print(traj_ep)
            # INFO: This part is to continue training the diffusion model with the new ground truth trajectories (prisoner follows guided sampling)
            # batches_seqLen_agentLocations = diffusion_model_dataloader.__next__()
            diffusion_loss, infos = diffusion_model.loss(*batches_seqLen_agentLocations)
            diffusion_optimizer.zero_grad()
            diffusion_loss.backward()
            torch.nn.utils.clip_grad_norm(diffusion_model.parameters(), 0.1)
            diffusion_optimizer.step()
            # print("diffusion_loss = ", diffusion_loss)
            diffusion_train_losses.append(diffusion_loss)

        if ep_i % validation_epoch_period == 0:
            diffusion_valid_losses = []
            with torch.no_grad():
                for traj_ep, batches_seqLen_agentLocations in tqdm(enumerate(diffusion_valid_dataloader)):
                    # INFO: This part is to valid the training process
                    # batches_seqLen_agentLocations = diffusion_model_dataloader.__next__()
                    diffusion_loss, infos = diffusion_model.loss(*batches_seqLen_agentLocations)
                    diffusion_valid_losses.append(diffusion_loss)                

        if torch.Tensor(diffusion_valid_losses).mean().item() < min_valid_loss:
            min_valid_loss = torch.Tensor(diffusion_valid_losses).mean().item()
            torch.save(diffusion_model, path+f"/diffusion_epoch_{ep_i}.pth")
            torch.save(diffusion_model, path+f"/diffusion_final.pth")

        writer.add_scalars('loss', {'train_loss': torch.Tensor(diffusion_train_losses).mean().item(), 'valid_loss': torch.Tensor(diffusion_valid_losses).mean().item()}, ep_i)
        print("Trajectory eps: ", ep_i)
        print("min_valid_loss: ", min_valid_loss)

def red_diffusion_test(config):

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
    elif config["train_option"] == 'dynamic':
        test_collate_fn = diffusion_test_dataset.dynamic_collate_fn_repeat
        sample_type = "original"
    else:
        raise NotImplementedError

    diffusion_test_dataloader = (torch.utils.data.DataLoader(diffusion_test_dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False, collate_fn=test_collate_fn))


    # INFO: load the saved diffusion model
    path = f'./CoRL_2024_logs/{config["folder_name"]}/benchmark_{config["train_option"]}_{config["condition_mode"]}_{config["pred_mode"]}'
    diffusion_model = torch.load(path +"/diffusion_final.pth")
    count_parameters(diffusion_model)
    # INFO: draw samples from current diffusion model
    for traj_ep, batches_seqLen_agentLocations in tqdm(enumerate(diffusion_test_dataloader)):
        # if traj_ep == config["test_traj_ep"]:
        #     break
        # INFO: set random seed for the diffusion denoise process
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # INFO: sample from the diffusion modeld
        # diffusion_model.n_timesteps = 10
        start_time = time.time()
        sample = diffusion_model.conditional_sample(global_cond=batches_seqLen_agentLocations[1], cond=batches_seqLen_agentLocations[2], sample_type=sample_type)
        end_time = time.time()
        print("Diffusion takes %f seconds." % (end_time-start_time))
        sample = diffusion_test_dataset.unnormalize(sample.detach())
        if config["train_option"] != "dynamic":
            detections = diffusion_test_dataset.unnormalize(batches_seqLen_agentLocations[1]["detections"].data[:,1:])
        gt_sample = diffusion_test_dataset.unnormalize(batches_seqLen_agentLocations[0])


        # INFO: plot 1) previous ball+chair traj in image space 2) predicted ball+chair traj in image space 3) GT trajs
        figure, axes = plt.subplots()
        tennis_court = mpimg.imread("./images/tennis_court.png")
        axes.imshow(tennis_court, extent=(0, 1280, 720, 0))
        if config["train_option"] != "dynamic":
            # axes.scatter(detections[:,0], detections[:,1], s=50, c=np.arange(len(detections[:,0])), cmap='Greens', label="Tennis Ball") # prev ball traj
            axes.scatter(detections[:,0], detections[:,1], s=50, c="lime", alpha=0.5, label="Tennis Ball") # prev ball traj
        if config["train_option"] == "ball_chair":
            axes.scatter(detections[:,2], detections[:,3], s=20, c=np.arange(len(detections[:,2])), cmap='Purples') # prev chair traj
            for i in range(1):
                axes.scatter(sample[i,:,0], sample[i,:,1], s=10, c=np.arange(config["horizon"]), cmap='Greys') # predicted ball traj
                axes.scatter(sample[i,:,2], sample[i,:,3], s=10, c=np.arange(config["horizon"]), cmap='Blues') # predicted chair traj
            axes.scatter(gt_sample[0,:,0], gt_sample[0,:,1], s=10, c='k') # GT ball traj
            axes.scatter(gt_sample[0,:,2], gt_sample[0,:,3], s=10, c='m') # GT chair traj
            chair_traj = sample[0,:,2:]
            chair_traj_gt = gt_sample[0,:,2:]
        else:
            chair_traj = sample[0,:]
            chair_traj_gt = gt_sample[0,:]
            for i in range(1):
                # axes.scatter(sample[i,:,0], sample[i,:,1], s=20, c=np.arange(config["horizon"]), cmap='Blues') # predicted ball traj
                axes.scatter(sample[i,:,0], sample[i,:,1], s=20, c='b', label="Predicted Wheelchair") # predicted ball traj
            # axes.scatter(gt_sample[0,:,0], gt_sample[0,:,1], s=10, c=np.arange(config["horizon"]), cmap='Reds') # GT ball traj
            axes.scatter(gt_sample[0,:,0], gt_sample[0,:,1], s=10, c='r', label="GT Wheelchair") # GT ball traj

        plt.axis('square')
        if config["pred_mode"] == "2d":
            axes.set_xticks([])
            axes.set_yticks([])
            plt.legend()
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            plt.gca().invert_yaxis()
        elif config["pred_mode"] == "3d":
            axes.set_xticks([])
            axes.set_yticks([])
            plt.legend()
            plt.ylim(-5, 25)
            plt.xlim(-6, 6)
            # plt.xlim(-5, 5)
            # plt.ylim(-5, 5)
            plt.gca().invert_xaxis()  # Invert x-axis
            plt.grid(visible=True)
        else:
            raise NotImplementedError
          # Invert x-axis
        plt.show()
        # plt.savefig("./images/trajectories/diverse_diffusion_paths_%d.png" % traj_ep, bbox_inches='tight')

def red_diffusion_eval(config):
    # INFO: load the test data into dataloader
    diffusion_test_dataset = BallWheelchairJointDataset(folder_path="./data/real_image_traj/splited_processed_trajs/test/",
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
    elif config["train_option"] == 'dynamic':
        test_collate_fn = diffusion_test_dataset.dynamic_collate_fn_repeat
        sample_type = "original"
    else:
        raise NotImplementedError
    diffusion_test_dataloader = (torch.utils.data.DataLoader(diffusion_test_dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False, collate_fn=test_collate_fn))
    # INFO: load the saved diffusion model
    path = f'./CoRL_2024_logs/{config["folder_name"]}/benchmark_{config["train_option"]}_{config["condition_mode"]}_{config["pred_mode"]}'
    diffusion_model = torch.load(path +"/diffusion_final.pth")
    # INFO: metrics we need to collect
    rmses = []
    distances = []
    icp_rmses = []
    energy_consumeds = []
    jerks = []
    # INFO: draw samples from current diffusion model
    for traj_ep, batches_seqLen_agentLocations in tqdm(enumerate(diffusion_test_dataloader)):
        # print("The trajectory ep is: ")
        # if traj_ep == config["test_traj_ep"]:
        #     break
        # INFO: set random seed for the diffusion denoise process
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # INFO: sample from the diffusion model
        # diffusion_model.n_timesteps = 10
        start_time = time.time()
        sample = diffusion_model.conditional_sample(global_cond=batches_seqLen_agentLocations[1], cond=batches_seqLen_agentLocations[2], sample_type=sample_type)
        end_time = time.time()
        print("Diffusion takes %f seconds." % (end_time-start_time))
        sample = diffusion_test_dataset.unnormalize(sample.detach())
        if config["train_option"] != "dynamic":
            detections = diffusion_test_dataset.unnormalize(batches_seqLen_agentLocations[1]["detections"].data[:,1:])
        gt_sample = diffusion_test_dataset.unnormalize(batches_seqLen_agentLocations[0])
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
            projection_matrix = np.load("./data/constant/projection_matrix.npy")
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

def red_diffusion_controller(config):

    # INFO: load the test data into dataloader
    diffusion_sim_dataset = BallWheelchairJointDataset(folder_path=config["test_set"], 
                                         horizon=config["horizon"],
                                         use_padding=config["use_padding"] ,
                                         max_path_length=config["max_path_length"],
                                         dataset_type = "pixel",
                                         include_start_detection = True,
                                         condition_path = True,
                                         max_trajectory_length=config["max_trajectory_length"],
                                         max_detection_num = config["max_detection_num"],
                                         condition_mode = config["condition_mode"],
                                         prediction_mode = config["pred_mode"],)

    if config["train_option"] == 'ball_chair':
        sim_collate_fn = diffusion_sim_dataset.ball_chair_collate_fn
        sample_type = "original"
    elif config["train_option"] == 'ball':
        sim_collate_fn = diffusion_sim_dataset.ball_collate_fn_repeat
        sample_type = "original"
    elif config["train_option"] == 'chair':
        sim_collate_fn = diffusion_sim_dataset.chair_collate_fn
        sample_type = "original"
    else:
        raise NotImplementedError

    # INFO: load the saved diffusion model
    path = f'./CoRL_2024_logs/{config["folder_name"]}/benchmark_{config["train_option"]}_{config["condition_mode"]}_{config["pred_mode"]}'
    diffusion_model = torch.load(path +"/diffusion_final.pth")


    # INFO: draw samples from current diffusion model
    for path_ind in range(0, diffusion_sim_dataset.path_num):
        normalized_sample = None
        ims = []
        figure, axes = plt.subplots()
        gt_ball_locs = diffusion_sim_dataset.unnormalize(diffusion_sim_dataset.agent_locs[path_ind][:,:2])
        gt_chair_locs = diffusion_sim_dataset.unnormalize(diffusion_sim_dataset.agent_locs[path_ind][:,2:])

        for start in range(0, diffusion_sim_dataset.path_lengths[path_ind]):

            # INFO: set random seed for the diffusion denoise process
            seed = 0
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            # INFO: update the diffusion path every replan period
            if start % config["replan_period"] == 0:
                batch = diffusion_sim_dataset.draw_with_path_point_ind(path_ind, start, normalized_sample)
                collected_batch = sim_collate_fn(batch)

                # INFO: sample from the diffusion model
                start_time = time.time()
                normalized_sample = diffusion_model.conditional_sample(global_cond=collected_batch[1], cond=collected_batch[2], sample_type=sample_type)
                end_time = time.time()
                print("Diffusion takes %f seconds." % (end_time-start_time))
                local_idx = 0
                sample = diffusion_sim_dataset.unnormalize(normalized_sample.detach())

            im_gt_ball = axes.scatter(gt_ball_locs[start,0], gt_ball_locs[start,1], s=50, c='g', animated=True) # GT ball traj
            # im_gt_ball = axes.scatter(gt_ball_locs[:,0], gt_ball_locs[:,1], s=50, c=np.arange(len(gt_ball_locs[:,0])), cmap='Greens', animated=True) # GT ball traj
            im_gt_chair = axes.scatter(gt_chair_locs[start,0], gt_chair_locs[start,1], s=50, c='purple', animated=True) # GT chair traj
            # im_gt_chair = axes.scatter(gt_chair_locs[:,0], gt_chair_locs[:,1], s=50, c=np.arange(len(gt_chair_locs[:,0])), cmap='Purples', animated=True, alpha=0.1) # GT chair traj
            im_pred_chair_pt = axes.scatter(sample[0,local_idx,0], sample[0,local_idx,1], s=20, c='b', animated=True) # predicted ball traj
            axes.legend(['GT Ball', 'GT Chair', 'Pred Chair'])
            axes.set_title("Court Navigation Animation")
            local_idx = local_idx + 1
            plt.axis('square')
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
            plt.grid(visible=True)
            plt.gca().invert_yaxis()  # Invert x-axis
            plt.tight_layout()
            # plt.show()

            ims.append([im_gt_ball, im_gt_chair, im_pred_chair_pt])
        ani = animation.ArtistAnimation(figure, ims, interval=50, blit=True, repeat_delay=1000)
        ani.save("movie_%d.mp4" % path_ind)

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
    # print("homogeneous_image_points = ", homogeneous_image_points)

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
        red_diffusion_train(config)
    elif config["stage"] == "test":
        red_diffusion_test(config)
    elif config["stage"] == "eval":
        mse, distance, icp, energy, jerk = red_diffusion_eval(config)
        metrics = np.array([mse, distance, icp, energy, jerk])
        np.savetxt(f'./CoRL_2024_logs/{config["folder_name"]}/benchmark_{config["train_option"]}_{config["condition_mode"]}_{config["pred_mode"]}/metrics.txt', metrics, delimiter=',')
    # elif config["stage"] == "vis":
        
    elif config["stage"] == "sim":
        # TODO: simulation for the real-time close loop controller!
        red_diffusion_controller(config)
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
    train_options = ["chair", "ball_chair", "ball"]
    condition_modes = ["post", "pre"]
    pred_modes = ["2d", "3d"]
    train_or_test = ["eval"]

    # INFO: Specify training/testing config
    train_options = ["chair"]
    condition_modes = ["post"]
    pred_modes = ["2d"]
    train_or_test = ["eval"]

    for to in train_options:
        for cm in condition_modes:
            for pm in pred_modes:
                for tt in train_or_test:
                    set_config(config, to, cm, pm, tt)
                    if check_config(config):
                        run(config)
    
    
    
    error_logs = {}

    # INFO: test some configurations
    # train_options = ["chair"]
    # condition_modes = ["post"]
    # pred_modes = ["2d"]

    # for to in train_options:
    #     for cm in condition_modes:
    #         for pm in pred_modes:
    #             try:
    #                 set_config(config, to, cm, pm, 'test')
    #                 if check_config(config):
    #                     run(config) 
    #             except Exception as e:
    #                 error_logs[(to, cm, pm)] = str(e)

    # # Print the error logs
    # for k, v in error_logs.items():
    #     print(f"{k} <- Error: {v}")             