""" This script preprocess trajectories of wheelchair and ball and write results into npz files as datasets"""
# import wandb
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from diffuser.datasets.multipath import BallWheelchairJointDataset
from diffuser.graders.traj_graders import joint_traj_grader
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.temporal_film import ConditionalUnet1D
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import argparse
import random
import time

def get_start_end_failure_num(positions):

    start_failure_num, end_failure_num = 0, 0
    raw_data_num = len(positions)

    # INFO: delete the first several detection failure rows
    for i in range(len(positions)):
        if np.array_equal(positions[i], [-1, -1]):
            positions = np.delete(positions, i, axis=0)
            start_failure_num = start_failure_num + 1
        else:
            break

    # INFO: delete the last several detection failure rows
    for i in reversed(range(len(positions))):
        if np.array_equal(positions[i], [-1, -1]):
            positions = np.delete(positions, i, axis=0)
            end_failure_num = end_failure_num + 1
        else:
            break
    
    return start_failure_num, end_failure_num

def interpolate_detect_failure(positions):

    # INFO: start interpolation!
    for i in range(len(positions)):
        if np.array_equal(positions[i], [-1, -1]):  # Check if the data point is missing
            # INFO: Find the indices of the nearest observed neighbors
            left_neighbor_idx = i - 1
            right_neighbor_idx = i + 1
            
            # INFO: Find the nearest observed neighbors with valid indices
            left_shift, right_shift = 1, 1
            while np.array_equal(positions[left_neighbor_idx], [-1, -1]):
                left_neighbor_idx -= 1
                left_shift = left_shift + 1
            while np.array_equal(positions[right_neighbor_idx], [-1, -1]):
                right_neighbor_idx += 1
                right_shift = right_shift + 1
            
            # INFO: Interpolate the missing value based on the observed neighbors
            positions[i] = positions[left_neighbor_idx]*(left_shift/(left_shift+right_shift)) + positions[right_neighbor_idx]*(right_shift/(left_shift+right_shift))
    return positions

def preprocess_traj(txt_folder_name, npz_folder_name, undetect_process_mode):

    # INFO: Get all files in this folder
    traj_txt_file_names = os.listdir(txt_folder_name)
    traj_num = len(traj_txt_file_names)
    train_traj_num = int(0.55 * traj_num)
    train_path = npz_folder_name + "train/"
    os.makedirs(train_path, exist_ok=True)
    valid_traj_num = int(0.3 * traj_num)
    valid_path = npz_folder_name + "valid/"
    os.makedirs(valid_path, exist_ok=True)
    test_traj_num = traj_num - train_traj_num - valid_traj_num
    test_path = npz_folder_name + "test/"
    os.makedirs(test_path, exist_ok=True)

    for traj_i, file_name in enumerate(traj_txt_file_names):

        # INFO: Construct the full file path
        gt_traj_full_path = txt_folder_name + file_name

        # INFO: Load the trajectory info from the first line of the txt file
        with open(gt_traj_full_path, 'r') as traj_info_file:
            traj_info = traj_info_file.readline().split()
            last_frame_id_before_hit = int(traj_info[0])

        # INFO: Load trajectory data from txt file
        data = np.loadtxt(gt_traj_full_path, skiprows=1)

        # INFO: Extract wheelchair and ball positions
        wheelchair_3d_positions = data[:last_frame_id_before_hit, :3]
        wheelchair_2d_positions = data[:last_frame_id_before_hit, 3:5]
        ball_2d_positions = data[:last_frame_id_before_hit, 5:7]

        if undetect_process_mode == "skip":
            valid_indices = np.where(np.all(ball_2d_positions != np.array([-1, -1]), axis=1))
            valid_data_num = valid_indices[0].shape[0]
            # print("res:", ball_positions == np.array([-1, -1]))
            # print("invalid_idx: ", invalid_idx)

            # INFO: Extract the valid datapoints
            wheelchair_3d_positions = wheelchair_3d_positions[valid_indices[0], :]
            wheelchair_2d_positions = wheelchair_2d_positions[valid_indices[0], :]
            ball_2d_positions = ball_2d_positions[valid_indices[0], :]

            timestep_observations = np.arange(0, valid_data_num)

        elif undetect_process_mode == "interpolate":

            # INFO: get start and failure num of all the data type
            ball_2d_start_failure_num, ball_2d_end_failure_num = get_start_end_failure_num(ball_2d_positions)
            wheelchair_2d_start_failure_num, wheelchair_2d_end_failure_num = get_start_end_failure_num(wheelchair_2d_positions)
            wheelchair_3d_start_failure_num, wheelchair_3d_end_failure_num = get_start_end_failure_num(wheelchair_3d_positions)
            start_delete_num = np.max([ball_2d_start_failure_num, wheelchair_2d_start_failure_num, wheelchair_3d_start_failure_num])
            end_delete_num = np.max([ball_2d_end_failure_num, wheelchair_2d_end_failure_num, wheelchair_3d_end_failure_num])
            ball_2d_positions = ball_2d_positions[start_delete_num:len(ball_2d_positions)-end_delete_num,:]
            wheelchair_2d_positions = wheelchair_2d_positions[start_delete_num:len(wheelchair_2d_positions)-end_delete_num,:]
            wheelchair_3d_positions = wheelchair_3d_positions[start_delete_num:len(wheelchair_3d_positions)-end_delete_num,:]

            # INFO: interpolate the ball 2d trajectory
            ball_2d_positions = interpolate_detect_failure(ball_2d_positions)

            # INFO: interpolate the wheelchair 2d and 3d trajectory
            wheelchair_2d_positions = interpolate_detect_failure(wheelchair_2d_positions)
            wheelchair_3d_positions = interpolate_detect_failure(wheelchair_3d_positions)

            timestep_observations = np.arange(0, len(ball_2d_positions))

        
        detected_locations = np.concatenate((ball_2d_positions, wheelchair_2d_positions, wheelchair_3d_positions), axis=1)

        if traj_i < train_traj_num:
            np.savez(train_path + "real_%d.npz" % traj_i, 
                timestep_observations=timestep_observations, 
                detected_locations=detected_locations,
                ball_locations=ball_2d_positions,
                chair_2d_locations=wheelchair_2d_positions,
                chair_3d_locations=wheelchair_3d_positions,)
        elif traj_i < train_traj_num + valid_traj_num:
            np.savez(valid_path + "real_%d.npz" % (traj_i), 
                timestep_observations=timestep_observations, 
                detected_locations=detected_locations,
                ball_locations=ball_2d_positions,
                chair_2d_locations=wheelchair_2d_positions,
                chair_3d_locations=wheelchair_3d_positions,)
        else:
            np.savez(test_path + "real_%d.npz" % (traj_i), 
                timestep_observations=timestep_observations, 
                detected_locations=detected_locations,
                ball_locations=ball_2d_positions,
                chair_2d_locations=wheelchair_2d_positions,
                chair_3d_locations=wheelchair_3d_positions,)

if __name__ == "__main__":  
    txt_folder_name = "./data/real_image_traj/splited_raw_trajs/automatic/" # "./data/real_image_traj/splited_raw_trajs/global/"
    npz_folder_name = "./data/real_image_traj/automatic_processed_trajs/" # "./data/real_image_traj/splited_processed_trajs/"
    undetect_process_mode = "interpolate"
    preprocess_traj(txt_folder_name, npz_folder_name, undetect_process_mode)