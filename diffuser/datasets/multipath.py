import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn.functional as F
import copy
import random
from collections import namedtuple
# from diffuser.utils.rendering import PrisonerRendererGlobe, PrisonerRenderer
from diffuser.datasets.prisoner import pad_collate_detections, pad_collate_detections_repeat

global_device_name = "cpu"
global_device = torch.device("cpu")

class StateOnlyDataset(torch.utils.data.Dataset):
    """ Single stream dataset where we cannot tell which agent is which in the detections"""
    def __init__(self, 
                 folder_path, 
                 horizon,
                 dataset_type = "sponsor",
                 include_start_detection = False,
                 condition_path = True,
                 max_trajectory_length = 4320,
                 ):
        print("Loading dataset from: ", folder_path)

        # assert global_lstm_include_start # this variable is a remnant from past dataset

        self.condition_path = condition_path

        print("Condition Path: ", self.condition_path)

        self.dataset_type = dataset_type
        self.observation_dim = 2
        self.horizon = horizon # how many timesteps we are using in one trajectory?
        self.max_trajectory_length = max_trajectory_length

        self.dones = []
        self.agent_locs = []
        self.process_first_file = True

        # INFO: load data from dataset to lists here
        self._load_data(folder_path)

        # INFO: the number of trajectories we have in this dataset
        self.indices = np.arange(self.file_num)


    def _load_data(self, folder_path):
        self.file_num = 0

        # INFO: load each trajectory with extend ??? into np_files list
        np_files = []
        fps = get_lowest_root_folders(folder_path)
        for fp in fps:
            for file_name in sorted(os.listdir(fp)):
                np_file = np.load(os.path.join(fp, file_name), allow_pickle=True)
                # print(np_file)
                # self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
                np_files.append(np_file)
                self.file_num = self.file_num + 1
        self.set_normalization_factors()
        for np_file in np_files:
            self._load_file(np_file)

        # print("Path Lengths: ")
        # print(max(self.path_lengths), min(self.path_lengths))
        
        # normalize hideout locations
        if self.dataset_type == "prisoner_globe":
            for i in range(len(self.hideout_locs)):
                # INFO: find the target hideout for each file
                target_hideout_loc = self.find_target_hideout(self.hideout_locs[i][0].flatten(), i)
                self.target_hideout_locs[i] = self.normalize(target_hideout_loc)
                self.hideout_locs[i] = self.normalize(self.hideout_locs[i])

    def find_target_hideout(self, hideout_loc, path_ind):
        # INFO: find the hideout the prisoner is reaching
        red_path_terminal_loc = self.unnormalize(self.red_locs[path_ind][-1,:2])
        hideout_num = len(hideout_loc) // 2
        hideout_reached_id = 0
        hideout_reached_dist = np.inf
        for hideout_id in range(hideout_num):
            candidate_hideout_loc = hideout_loc[2*hideout_id:2*hideout_id+2]
            candidate_terminal_error = np.linalg.norm(red_path_terminal_loc - candidate_hideout_loc)
            if candidate_terminal_error < hideout_reached_dist:
                hideout_reached_dist = candidate_terminal_error
                hideout_reached_id = hideout_id
            else:
                pass
        hideout_loc = hideout_loc[2*hideout_reached_id:2*hideout_reached_id+2]
        return hideout_loc

    def set_normalization_factors(self):
        self.min_x = 0
        self.max_x = 2428
        self.min_y = 0
        self.max_y = 2428

    def normalize(self, arr):
        x = arr[..., 0]
        arr[..., 0] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

        y = arr[..., 1]
        arr[..., 1] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        return arr

    def unnormalize(self, obs):
        obs = copy.deepcopy(obs)

        last_dim = obs.shape[-1]
        evens = np.arange(0, last_dim, 2)
        odds = np.arange(1, last_dim, 2)

        x = obs[..., evens]
        obs[..., evens] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., odds]
        obs[..., odds] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        # x = obs[..., 0]
        # obs[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        # y = obs[..., 1]
        # obs[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        # x_1 = obs[..., 2]
        # obs[..., 2] = ((x_1 + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        # y_1 = obs[..., 3]
        # obs[..., 3] = ((y_1 + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        return obs
    
    def unnormalize_single_dim(self, obs):
        x = obs[..., 0]
        obs[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., 1]
        obs[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        return obs
    
    def select_random_rows(self, array, n):
        b, m = array.shape

        if n >= b:
            return array

        indices = np.arange(b)
        np.random.shuffle(indices)

        selected_indices = indices[:n]
        remaining_indices = indices[n:]

        selected_rows = np.full((n, m), -np.inf)
        selected_rows[:len(selected_indices)] = array[selected_indices]

        result = np.copy(array)
        result[remaining_indices] = -np.inf

        return result

    def _load_file(self, file):

        timesteps = file["timestep_observations"]
        ball_locs = np.float32(file["ball_locations"])[0]
        chair_locs = np.float32(file["chair_2d_locations"])[0]
        path_length = len(ball_locs)

        # INFO: path length should be smaller than the limit
        if path_length > self.max_trajectory_length:
            raise ValueError("Path length is greater than max trajectory length")

        # INFO: normalize ball and wheelchair locations
        normalized_ball_loc = self.normalize(ball_locs)
        normalized_chair_loc = self.normalize(chair_locs)

        # INFO: load trajectories into lists
        if self.process_first_file:
            self.process_first_file = False
            self.timesteps = [timesteps]
            self.ball_locs = [normalized_ball_loc]
            self.chair_locs = [normalized_chair_loc]
            if self.dataset_type == "pixel" or self.dataset_type == "cartesian":
                self.hideout_locs = [hideout_locs]
                self.target_hideout_locs = [hideout_locs]
        else:
            self.timesteps.append(timesteps)
            self.ball_locs.append(normalized_ball_loc)
            self.chair_locs.append(normalized_chair_loc)
            if self.dataset_type == "pixel" or self.dataset_type == "cartesian":
                self.hideout_locs.append(hideout_locs)
                self.target_hideout_locs.append(hideout_locs)

    def convert_global_for_lstm(self, global_cond_idx, global_cond, start):
        """ Convert the indices back to timesteps and concatenate them together"""
        detection_num = min(self.max_detection_num, len(global_cond_idx))
        global_cond_idx = global_cond_idx[-detection_num:]
        global_cond = global_cond[-detection_num:]

        # no detections before start, just pad with -1, -1
        # assert len(global_cond_idx) != 0
            # return torch.tensor([[-1, -1, -1, -1, -1]])
        if len(global_cond_idx) == 0:
            return -1 * torch.ones((1, 213)) # 229 for 5s1h, 213 for 1s1h
        # convert the indices back to timesteps
        global_cond_idx_adjusted = (start - global_cond_idx) / self.max_trajectory_length
        global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)


        return torch.tensor(global_cond).float()

    def get_conditions(self, idx):
        '''
            condition on current observation for planning
        '''

        # INFO: get current path red loc
        red_loc = self.red_locs[idx]

        if self.condition_path:
            # always include the start of the path
            if self.include_start_detection:
                idxs = np.array([[0], [-1]])
                detects = np.array([red_loc[0], self.target_hideout_locs[idx]])
            else:
                idxs = np.array([])
                detects = np.array([])
        else:
            idxs = np.array([])
            detects = np.array([])

        return(idxs, detects)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # INFO: which path file we are chosing?
        path_ind = self.indices[idx]

        prisoner_locs = self.red_locs[path_ind]

        local_cond = self.get_conditions(path_ind)

        hideout_loc = self.hideout_locs[path_ind]

        global_cond = hideout_loc[0].flatten()

        prisoner_at_start = np.array(self.red_locs[path_ind][0,:2])

        batch = (prisoner_locs, global_cond, local_cond, prisoner_at_start)
        return batch

    def collate_fn(self, batch):
        (prisoner_locs, global_cond, local_cond, prisoner_at_start) = zip(*batch)

        path = torch.tensor(np.stack(prisoner_locs, axis=0))

        global_cond = torch.tensor(np.stack(global_cond, axis=0))

        # Pass this to condition our models rather than pass them separately
        global_dict = {"hideouts": global_cond.to(global_device_name), "red_start": torch.Tensor(prisoner_at_start).to(global_device_name)}

        return path, global_dict, local_cond
    
    def collate_fn_repeat(self, batch, num_samples):
        (global_cond, local_cond, prisoner_at_start) = zip(*batch)

        global_cond = torch.tensor(np.stack(global_cond, axis=0))

        global_cond = global_cond.repeat((num_samples, 1))
        local_cond = list(local_cond) * num_samples

        # INFO: This is for red traj only
        global_dict = {"hideouts": global_cond.to(global_device_name), 
            "red_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=num_samples, dim=0)}

        return global_dict, local_cond

class BallWheelchairJointDataset(torch.utils.data.Dataset):
    """ Single stream dataset where we cannot tell which agent is which in the detections"""
    def __init__(self, 
                 folder_path, 
                 horizon,
                 use_padding,
                 max_path_length,
                 dataset_type = "pixel",
                 include_start_detection = False,
                 condition_path = True,
                 max_detection_num = 32,
                 max_trajectory_length = 4320,
                 num_detections = 16,
                 train_mode="dynamic",
                 condition_mode = "pre",
                 prediction_mode = "2d",
                 ):

        # INFO: tell me where you are loading the dataset
        print("Loading dataset from: ", folder_path)

        # INFO: do we use constraints in the training
        self.condition_path = condition_path
        print("Condition Path: ", self.condition_path)
        self.mode = condition_mode
        self.train_mode = train_mode
        self.pred_mode = prediction_mode

        # INFO: set the dataset type from: 1) pixel 2) cartesian 3)
        self.dataset_type = dataset_type
        self.use_padding = use_padding
        self.observation_dim = 2
        self.horizon = horizon
        self.max_detection_num = max_detection_num
        self.max_trajectory_length = max_trajectory_length
        self.num_detections = num_detections

        # self.dones = []
        self.agent_locs = []
        self.process_first_file = True

        # INFO: load the timesteps and the padded ball-wheelchair poses into lists
        self._load_data(folder_path)

        # self.dones_shape = self.dones[0].shape
        # # These mark the end of each episode
        # self.done_locations = np.where(self.dones == True)[0]

        self.max_path_length = max_path_length
        self.include_start_detection = include_start_detection
        self.indices = self.make_indices(self.path_lengths, horizon)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        # INFO: divide trajectory i into indices ((i, start_ind=min_start+0, start + horizon), (i, start_ind=min_start+1, start + horizon), ... (i, max_start, max_start + horizon))
        for i, path_length in enumerate(path_lengths):
            min_start = 0 # 0, self.max_detection_num
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(min_start, max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def _load_data(self, folder_path):

        # INFO: load each trajectory with extend ??? into np_files list
        np_files = []
        fps = get_lowest_root_folders(folder_path)
        for fp in fps:
            for file_name in sorted(os.listdir(fp)):
                np_file = np.load(os.path.join(fp, file_name), allow_pickle=True)
                # print(np_file)
                # self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
                np_files.append(np_file)
        self.set_normalization_factors()

        # INFO: load trajectories into lists
        for np_file in np_files:
            self._load_file(np_file)

        # INFO: print out max-min path lengths for debug
        print("Path Lengths: ")
        print(max(self.path_lengths), min(self.path_lengths))

        # INFO: print out the number of paths in the dataset
        self.path_num = len(np_files)
        print("Path Num: ")
        print(self.path_num)

        # INFO: load the detections into self.detected_dics and now we assume fully observable so every ball-wheelchair pose is in
        self.process_detections()

        # after processing detections, we can pad
        if self.use_padding:
            for i in range(len(self.agent_locs)):
                # INFO: pad at the end of agent_locs such that we can still draw self.horizon out even from the last step in the traj
                # self.agent_locs[i] = np.pad(self.agent_locs[i], ((0, self.horizon), (0, 0)), 'constant', constant_values=self.agent_locs[i][-1])
                self.agent_locs[i] = np.pad(self.agent_locs[i], ((0, self.horizon), (0, 0)), 'edge')

    def set_normalization_factors(self):

        # INFO: set the size of the image space
        self.min_x = 0
        self.max_x = 1280
        self.min_y = 0
        self.max_y = 720

        # INFO: set the size of the court 3D space
        self.court_min_x = -5
        self.court_max_x = 5 # court length constraint
        self.court_min_y = -5
        self.court_max_y = 5 # court width constraint
        self.theta_min = 0
        self.theta_max = np.pi

    def normalize_2d(self, arr, last_dim):
        evens = np.arange(0, last_dim, 2)
        odds = np.arange(1, last_dim, 2)

        x = arr[..., evens]
        arr[..., evens] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

        y = arr[..., odds]
        arr[..., odds] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        return arr 

    def normalize_3d(self, arr):
        x_3d = arr[..., 0]
        arr[..., 0] = ((x_3d - self.court_min_x) / (self.court_max_x - self.court_min_x)) * 2 - 1

        y_3d = arr[..., 1]
        arr[..., 1] = ((y_3d - self.court_min_y) / (self.court_max_y - self.court_min_y)) * 2 - 1

        theta = arr[..., 2]
        arr[..., 2] = ((theta - self.theta_min) / (self.theta_max - self.theta_min)) * 2 - 1
        return arr 

    def unnormalize_2d(self, arr, last_dim):
        evens = np.arange(0, last_dim, 2)
        odds = np.arange(1, last_dim, 2)

        x = arr[..., evens]
        arr[..., evens] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = arr[..., odds]
        arr[..., odds] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y
        return arr

    def unnormalize_3d(self, arr):
        x_3d = arr[..., 0]
        arr[..., 0] = ((x_3d + 1) / 2) * (self.court_max_x - self.court_min_x) + self.court_min_x

        y_3d = arr[..., 1]
        arr[..., 1] = ((y_3d + 1) / 2) * (self.court_max_y - self.court_min_y) + self.court_min_y

        theta = arr[..., 2]
        arr[..., 2] = ((theta + 1) / 2) * (self.theta_max - self.theta_min) + self.theta_min

        return arr
    
    def normalize(self, arr):
        arr = copy.deepcopy(arr)
        last_dim = arr.shape[-1]
        if self.pred_mode == "2d":
            arr = self.normalize_2d(arr, last_dim)
        elif self.pred_mode == "3d":
            if last_dim == 2:
                arr = self.normalize_2d(arr, last_dim)             
            elif last_dim == 3:
                arr = self.normalize_3d(arr)
            elif last_dim == 7:
                part_2d = arr[..., :4]
                part_3d = arr[..., 4:]
                arr_2d = self.normalize_2d(part_2d, 4)
                arr_3d = self.normalize_3d(part_3d)
                arr = np.concatenate((arr_2d, arr_3d), axis=-1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return arr

    def unnormalize(self, obs):
        obs = copy.deepcopy(obs)
        last_dim = obs.shape[-1]
        if self.pred_mode == "2d":
            obs = self.unnormalize_2d(obs, last_dim)

        elif self.pred_mode == "3d":
            print('last_dim',last_dim)
            if last_dim == 2:
                obs = self.unnormalize_2d(obs, last_dim)
            elif last_dim == 3:
                obs = self.unnormalize_3d(obs)
            elif last_dim == 5:
                part_2d = obs[..., :2]
                part_3d = obs[..., 2:]
                arr_2d = self.normalize_2d(part_2d, 2)
                arr_3d = self.normalize_3d(part_3d)
                obs = np.concatenate((arr_2d, arr_3d), axis=-1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return obs
    
    def unnormalize_single_dim(self, obs):
        x = obs[..., 0]
        obs[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., 1]
        obs[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        return obs

    def process_detections(self):
        self.detected_dics = []
        # INFO: self.detected_locations is from dataset, it is indexed by (traj_id, detected_loc_id_in_each_traj)
        for detected_locs in self.detected_locations:
            indices = []
            detects = []
            for i in range(len(detected_locs)):
                loc = detected_locs[i]
                if 1: # we assume the ball-wheelchair trajectory is fully observable on the image
                    # INFO: we already normalized detections 
                    # loc[0:2] = loc[0:2] * 2 - 1
                    # loc[2:4] = loc[2:4] * 2 - 1
                    indices.append(i)
                    detects.append(loc)
            # INFO: stack all the detects and indices together and form the detected_dics. self.detected_dics structure dim is [traj_num, detect_num] with the structure
            # INFO: (cont'd) [[(indices_in_traj, detects), (indices_in_traj, detects), ..., (indices_in_traj, detects)], [], ..., []]
            detects = np.stack(detects, axis=0)
            indices = np.stack(indices, axis=0)
            self.detected_dics.append((indices, detects))
    
    def select_random_rows(self, array, n):
        b, m = array.shape

        if n >= b:
            return array

        indices = np.arange(b)
        np.random.shuffle(indices)

        selected_indices = indices[:n]
        remaining_indices = indices[n:]

        selected_rows = np.full((n, m), -np.inf)
        selected_rows[:len(selected_indices)] = array[selected_indices]

        result = np.copy(array)
        result[remaining_indices] = -np.inf

        return result

    def _load_file(self, file):

        timesteps = file["timestep_observations"]
        detected_locations = file["detected_locations"]
        ball_locs = np.float32(file["ball_locations"])
        if self.pred_mode == "2d":
            chair_locs = np.float32(file["chair_2d_locations"])
        elif self.pred_mode == "3d":
            chair_locs = np.float32(file["chair_3d_locations"])
            # chair_locs = np.concatenate((convert_wheelchair_2d_to_3d(np.float32(file["chair_2d_locations"]), np.load("./data/constant/projection_matrix.npy")), np.float32(file["chair_3d_locations"])[:,-1:]), axis=-1)
        else:
            raise NotImplementedError
        agent_locs = [ball_locs, chair_locs]
        # agent_locs = np.concatenate((ball_locs[:,np.newaxis,:], chair_locs[:,np.newaxis,:]), axis=1)

        if chair_locs.shape[-1] == 4:
            print(file)

        # timesteps = np.arange(agent_locs.shape[0]) / self.horizon
        
        # INFO: path length should be smaller than the limit
        path_length = len(agent_locs[0])
        if path_length > self.max_trajectory_length:
            raise ValueError("Path length is greater than max trajectory length")

        # INFO: normalize ball and wheelchair locations
        agents = []
        for i in range(len(agent_locs)):
            agent = self.normalize(agent_locs[i])
            # agent_b = self.normalize(blue_locs[:, 1, :])
            agents.append(agent)
        # r_locs_normalized = np.concatenate((agent_a, agent_b), axis=1)
        locs_normalized = np.concatenate(agents, axis=1)
        detected_locations_normalized = self.normalize(detected_locations)

        # INFO: each trajectory is an element in self.agent_locs
        if self.process_first_file:
            self.process_first_file = False
            self.timesteps = timesteps
            # self.dones = file["dones"]
            self.agent_locs = [locs_normalized]
            self.detected_locations = [detected_locations_normalized]
            self.path_lengths = [path_length]
            # if self.dataset_type == "pixel" or self.dataset_type == "cartesian":
            #     self.hideout_locs = [hideout_locs]
        else:
            self.agent_locs.append(locs_normalized)
            self.timesteps = np.append(self.timesteps, timesteps)
            # self.dones = np.append(self.dones, file["dones"])
            self.detected_locations.append(detected_locations_normalized)
            self.path_lengths.append(path_length)
            # if self.dataset_type == "pixel" or self.dataset_type == "cartesian":
            #     self.hideout_locs.append(hideout_locs)

    def convert_global_for_lstm(self, global_cond_idx, global_cond, start, mode="pre"):
        """ Convert the indices back to timesteps and concatenate them together"""
        if mode == "pre":
            # INFO: we predict the future traj based on at most previous self.max_detection_num steps
            detection_num = min(self.max_detection_num, len(global_cond_idx))
            global_cond_idx = global_cond_idx[-detection_num:]
            global_cond = global_cond[-detection_num:]

            # no detections before start, just pad with -1, -1
            # assert len(global_cond_idx) != 0
                # return torch.tensor([[-1, -1, -1, -1, -1]])
            if len(global_cond_idx) == 0:
                return -1 * torch.ones((1, 213)) # 229 for 5s1h, 213 for 1s1h
            # INFO: calculate normalized steps before the start of the sampled traj
            global_cond_idx_adjusted = (start - global_cond_idx) / self.max_trajectory_length
            # INFO: concat global_cond_idx_adjusted with detects before the sampled traj
            global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)
        elif mode == "post":
            # INFO: we also constrain the max detection num when we are using the post ball estimation 
            detection_num = min(self.max_detection_num, len(global_cond_idx))
            global_cond_idx = global_cond_idx[:detection_num]
            global_cond = global_cond[:detection_num]
            # INFO: calculate normalized steps after the start of the sampled traj
            global_cond_idx_adjusted = - (start - global_cond_idx) / self.max_trajectory_length
            # INFO: concat global_cond_idx_adjusted with detects before the sampled traj
            global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)
        else:
            raise NotImplementedError
        return torch.tensor(global_cond).float()

    def get_conditions(self, path_ind, start, end, trajectories):
        '''
            condition on current observation for planning
        '''
        # INFO: self.detected_dics structure dim is [traj_num, detect_num] with the structure
        # INFO: (cont'd) [[([indices_in_traj], [detects]), ([indices_in_traj], [detects]), ..., ([indices_in_traj], [detects])], [], ..., []]
        detected_dic = self.detected_dics[path_ind]

        if self.mode == "pre":
            # subtract off the start and don't take anything past the end
            start_idx_find = np.where(detected_dic[0] >= start)[0]
            end_idx_find = np.where(detected_dic[0] < end)[0]
            # These are global conditions where the global_cond_idx is the 
            # integer index within the trajectory of where the detection occured

            # INFO: Take the detections before the start of the trajectory
            before_start_detects = np.where(detected_dic[0] <= start)[0]
            if len(before_start_detects) == 0:
                global_cond_idx = np.array([])
                global_cond = np.array([])
            else:
                # INFO: indices_in_traj before the sampled traj
                global_cond_idx = detected_dic[0][:before_start_detects[-1]+1]
                # INFO: detects before the sampled traj
                global_cond = detected_dic[1][:before_start_detects[-1]+1]

            # INFO: concatenation of time and detects
            detection_lstm = self.convert_global_for_lstm(global_cond_idx, global_cond, start, mode=self.mode)
        elif self.mode == "post":
            # INFO: Take the detections before the start of the trajectory
            after_start_detects = np.where(detected_dic[0] >= start)[0]            
            if len(after_start_detects) == 0:
                global_cond_idx = np.array([])
                global_cond = np.array([])
            else:
                # INFO: indices_in_traj before the sampled traj
                global_cond_idx = detected_dic[0][after_start_detects[0]:]
                # INFO: detects before the sampled traj
                global_cond = detected_dic[1][after_start_detects[0]:]
            # INFO: concatenation of time and detects
            detection_lstm = self.convert_global_for_lstm(global_cond_idx, global_cond, start, mode=self.mode)
        else:
            raise NotImplementedError



        if self.condition_path:
            # always include the start of the path
            if self.include_start_detection:
                if self.train_mode == "dynamic":
                    idxs = np.array([[0], [-1]])
                    detects = np.array([[trajectories[0]], [trajectories[-1]]])
                elif self.train_mode == "ball" or self.train_mode == "chair" or self.train_mode == "ball_chair":
                    idxs = np.array([[0]])
                    detects = np.array([[trajectories[0]]])                    
            else:
                raise NotImplementedError
        else:
            idxs = np.array([])
            detects = np.array([])

        return detection_lstm, (idxs, detects)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        # INFO: each element in self.indices corresponds to one sample described by path_ind, start_ind_in_path, end_ind_in_path
        path_ind, start, end = self.indices[idx]

        # INFO: draw the sample described by path_ind, start_ind_in_path, end_ind_in_path OUT!
        trajectories = self.agent_locs[path_ind][start:end]

        # INFO: all_detections is the concatenation of time and detects; conditions is ???
        all_detections, conditions = self.get_conditions(path_ind, start, end, trajectories)

        # hideout_loc = self.hideout_locs[path_ind]
        # hideout_loc = self.find_target_hideout(hideout_loc, path_ind)
        # global_cond = hideout_loc

        ball_chair_at_start = np.concatenate((np.array([0]), np.array(trajectories[0])))

        batch = (trajectories, all_detections, conditions, ball_chair_at_start)
        return batch

    def draw_with_path_point_ind(self, path_ind, start, start_loc):
        end = start + self.horizon

        # INFO: draw the sample described by path_ind, start_ind_in_path, end_ind_in_path OUT!
        trajectories = self.agent_locs[path_ind][start:end]

        # INFO: all_detections is the concatenation of time and detects; conditions is ???
        all_detections, conditions = self.get_conditions(path_ind, start, end, trajectories)
        ball_chair_at_start = np.concatenate((np.array([0]), np.array(trajectories[0])))
        if start_loc is not None:
            conditions[1][0,0,2:] = start_loc[0,-1,:].detach().cpu().numpy()
            ball_chair_at_start[-2:] = start_loc[0,-1,:].detach().cpu().numpy()
        

        batch = (trajectories, all_detections, conditions, ball_chair_at_start)
        return [batch]

    def ball_chair_collate_fn(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        # global_cond = torch.tensor(np.stack(global_cond, axis=0))

        if self.pred_mode == "2d":
            all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
        elif self.pred_mode == "3d":
            all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
        else:
            raise NotImplementedError 
        
        # INFO: pad with the largest time_len?
        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        # Pass this to condition our models rather than pass them separately
        # global_dict = {"detections": detections.to(global_device_name), "red_start": torch.Tensor(ball_chair_at_start).to(global_device_name)}
        global_dict = {"detections": detections.to(global_device_name)}

        return data, global_dict, conditions

    def ball_chair_collate_fn_act(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        # global_cond = torch.tensor(np.stack(global_cond, axis=0))

        if self.pred_mode == "2d":
            all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
        elif self.pred_mode == "3d":
            all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
        else:
            raise NotImplementedError 
        
        # INFO: stack the trajectory samples together
        all_detections = [np.pad(all_detections[i][:,1:].detach().cpu().numpy(), ((0, self.max_detection_num-(all_detections[i][:,1:].detach().cpu().numpy().shape[0])), (0, 0)), 'edge') 
                                    for i in range(len(all_detections))]
        all_detections = torch.tensor(np.stack(all_detections, axis=0))

        # INFO: indicate if the value is a padding value
        appendix_padded_num = torch.sum(torch.all(data==data[:,-1:,:], axis=-1), axis=-1)
        is_pad = torch.zeros(len(data), self.horizon)
        for i in range(len(data)):
            is_pad[i, -appendix_padded_num[i]:] = 1

        return data, all_detections, is_pad.bool(), torch.Tensor(ball_chair_at_start)

    def ball_chair_collate_fn_repeat(self, batch):

        num_samples = 1

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        # global_cond = torch.tensor(np.stack(global_cond, axis=0))

        data = data.repeat((num_samples, 1, 1))
        all_detections = list(all_detections) * num_samples
        conditions = list(conditions) * num_samples
        # ball_chair_at_start = ball_chair_at_start * num_samples

        if self.pred_mode == "2d":
            all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
        elif self.pred_mode == "3d":
            all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
        else:
            raise NotImplementedError 
        
        # INFO: pad with the largest time_len?
        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        # Pass this to condition our models rather than pass them separately
        # global_dict = {"detections": detections.to(global_device_name), "red_start": torch.Tensor(ball_chair_at_start).to(global_device_name)}
        global_dict = {"detections": detections.to(global_device_name)}

        return data, global_dict, conditions

    def ball_collate_fn(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only ball states 
        data = [data[i][:,:2] for i in range(len(data))]
        all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        conditions = [((conditions[i][0]), conditions[i][1][...,:2]) for i in range(len(conditions))]
        ball_chair_at_start = [ball_chair_at_start[i][:3] for i in range(len(ball_chair_at_start))]
        
        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        # global_cond = torch.tensor(np.stack(global_cond, axis=0))

        # INFO: pad with the largest time_len?
        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        # Pass this to condition our models rather than pass them separately
        # global_dict = {"detections": detections.to(global_device_name), "red_start": torch.Tensor(ball_chair_at_start).to(global_device_name)}
        global_dict = {"detections": detections.to(global_device_name)}

        return data, global_dict, conditions

    def ball_collate_fn_act(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only ball states 
        data = [data[i][:,:2] for i in range(len(data))]
        all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        conditions = [((conditions[i][0]), conditions[i][1][...,:2]) for i in range(len(conditions))]
        ball_chair_at_start = [ball_chair_at_start[i][:3] for i in range(len(ball_chair_at_start))]
        
        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        batch_detection_num = [all_detections[i][:,1:].detach().cpu().numpy().shape[0] for i in range(len(all_detections))]
        all_detections = [np.pad(all_detections[i][:,1:].detach().cpu().numpy(), ((0, self.max_detection_num-(all_detections[i][:,1:].detach().cpu().numpy().shape[0])), (0, 0)), 'edge') 
                                    for i in range(len(all_detections))]
        all_detections = torch.tensor(np.stack(all_detections, axis=0))

        # INFO: indicate if the value is a padding value
        appendix_padded_num = torch.sum(torch.all(data==data[:,-1:,:], axis=-1), axis=-1)
        is_pad = torch.zeros(len(data), self.horizon)
        for i in range(len(data)):
            is_pad[i, -appendix_padded_num[i]:] = 1

        return data, all_detections, is_pad.bool(), torch.Tensor(ball_chair_at_start)

    def ball_collate_fn_repeat(self, batch):
        num_samples = 20

        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only ball states 
        data = [data[i][:,:2] for i in range(len(data))]
        all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        conditions = [((conditions[i][0]), conditions[i][1][...,:2]) for i in range(len(conditions))]
        ball_chair_at_start = [ball_chair_at_start[i][:3] for i in range(len(ball_chair_at_start))]

        data = torch.tensor(np.stack(data, axis=0))

        data = data.repeat((num_samples, 1, 1))
        all_detections = list(all_detections) * num_samples
        conditions = list(conditions) * num_samples

        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        # Pass this to condition our models rather than pass them separately
        # global_dict = {"detections": detections.to(global_device_name), "red_start": torch.Tensor(ball_chair_at_start).to(global_device_name)}
        global_dict = {"detections": detections.to(global_device_name)}
        # global_dict = {"hideouts": global_cond, "detections": detections}

        return data, global_dict, conditions

    def chair_collate_fn(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only wheelchair states 
        data = [data[i][:,2:] for i in range(len(data))]
        if self.mode == "post":
            all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        elif self.mode == "pre":
            if self.pred_mode == "2d":
                all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
            elif self.pred_mode == "3d":
                all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError   
        conditions = [((conditions[i][0]), conditions[i][1][...,2:]) for i in range(len(conditions))]
        
        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        # global_cond = torch.tensor(np.stack(global_cond, axis=0))

        # INFO: pad with the largest time_len?
        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        # Pass this to condition our models rather than pass them separately
        if self.mode == "post":
            global_dict = {"detections": detections.to(global_device_name), "red_start": torch.Tensor(ball_chair_at_start).to(global_device_name)}
        elif self.mode == "pre":
            global_dict = {"detections": detections.to(global_device_name)}
        else:
            raise NotImplementedError

        return data, global_dict, conditions

    def chair_mlp_collate_fn(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only wheelchair states 
        data = [data[i][:,2:] for i in range(len(data))]



        if self.mode == "post":
            all_detections = [all_detections[i][:,1:3] for i in range(len(all_detections))]
            
        elif self.mode == "pre":
            
            if self.pred_mode == "2d":
                # all_detections = [all_detections[i][:,1:5] for i in range(len(all_detections))]
                all_detections = [all_detections[i][:,1:3] for i in range(len(all_detections))]
            elif self.pred_mode == "3d":
                # all_detections = [torch.cat((all_detections[i][:,1:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
                all_detections = [all_detections[i][:,1:3] for i in range(len(all_detections))]
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError  

        padded_all_detections = []
        for i, detections in enumerate(all_detections):
            if self.mode == "post":
                chair_start = torch.Tensor(ball_chair_at_start[i][3:]).unsqueeze(0) if not isinstance(ball_chair_at_start[i][1:], torch.Tensor) else (ball_chair_at_start[i][1:]).unsqueeze(0).float()
                curr_detection_num = detections.shape[0]
                replicate = detections[-1:]
                padding_last = replicate.expand(self.max_detection_num-curr_detection_num, -1)
                detections = torch.cat((detections, padding_last), dim=0)
                padded_all_detections.append(detections.unsqueeze(0))
            elif self.mode == "pre":
                chair_start = torch.Tensor(ball_chair_at_start[i][1:]).unsqueeze(0) if not isinstance(ball_chair_at_start[i][1:], torch.Tensor) else (ball_chair_at_start[i][1:]).unsqueeze(0).float()
                curr_detection_num = detections.shape[0]
                replicate = detections[0:1]
                padding_first = replicate.expand(self.max_detection_num-curr_detection_num, -1)
                detections = torch.cat((padding_first, detections), dim=0)
                padded_all_detections.append(detections.unsqueeze(0))
            else:
                raise NotImplementedError

        padded_all_detections = torch.vstack(padded_all_detections)
        
        if self.mode == "post":
            ball_chair_at_start = torch.Tensor(ball_chair_at_start)[:,3:] if not isinstance(ball_chair_at_start[0], torch.Tensor) else ball_chair_at_start[0][np.newaxis,3:].float()
        elif self.mode == "pre":
            ball_chair_at_start = torch.Tensor(ball_chair_at_start)[:,3:] if not isinstance(ball_chair_at_start[0], torch.Tensor) else ball_chair_at_start[0][np.newaxis,3:].float()
        else:
            raise NotImplementedError

        
        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        # INFO: batch size
        batch_size = data.shape[0]

        # INFO: output should have a size of [batch_size, ]
        data = data.view(batch_size, -1)
        padded_all_detections = padded_all_detections.view(batch_size, -1)
        if self.mode == "post":
            padded_all_detections = torch.cat((ball_chair_at_start, padded_all_detections), dim=-1)
        elif self.mode == "pre":
            padded_all_detections = torch.cat((padded_all_detections, ball_chair_at_start), dim=-1)
        else:
            raise NotImplementedError


        return data, padded_all_detections

    def hitpt_collate_fn(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only wheelchair states 
        data = [data[i][:,2:] for i in range(len(data))]
        if self.mode == "post":
            all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        elif self.mode == "pre":
            if self.pred_mode == "2d":
                all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
            elif self.pred_mode == "3d":
                all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError   
        conditions = [((conditions[i][0]), conditions[i][1][...,2:]) for i in range(len(conditions))]
        
        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        # global_cond = torch.tensor(np.stack(global_cond, axis=0))

        # INFO: pad with the largest time_len?
        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        # Pass this to condition our models rather than pass them separately
        if self.mode == "post":
            global_dict = {"detections": detections.to(global_device_name), "red_start": torch.Tensor(ball_chair_at_start).to(global_device_name)}
        elif self.mode == "pre":
            global_dict = {"detections": detections.to(global_device_name)}
        else:
            raise NotImplementedError

        return data, global_dict, conditions

    def chair_collate_fn_act(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only ball states 
        data = [data[i][:,2:] for i in range(len(data))]

        if self.mode == "post":
            all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        elif self.mode == "pre":
            if self.pred_mode == "2d":
                all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
            elif self.pred_mode == "3d":
                all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError 
 
        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        all_detections = [np.pad(all_detections[i][:,1:].detach().cpu().numpy(), ((0, self.max_detection_num-(all_detections[i][:,1:].detach().cpu().numpy().shape[0])), (0, 0)), 'edge') 
                                    for i in range(len(all_detections))]
        all_detections = torch.tensor(np.stack(all_detections, axis=0))

        # INFO: indicate if the value is a padding value
        appendix_padded_num = torch.sum(torch.all(data==data[:,-1:,:], axis=-1), axis=-1)
        is_pad = torch.zeros(len(data), self.horizon)
        for i in range(len(data)):
            is_pad[i, -appendix_padded_num[i]:] = 1

        return data, all_detections, is_pad.bool(), torch.Tensor(ball_chair_at_start) if not isinstance(ball_chair_at_start[0], torch.Tensor) else ball_chair_at_start[0].unsqueeze(0).float()

    
    def chair_collate_fn_repeat(self, batch):
        num_samples = 5

        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only wheelchair states 
        data = [data[i][:,2:] for i in range(len(data))]
        if self.mode == "post":
            all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        elif self.mode == "pre":
            if self.pred_mode == "2d":
                all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
            elif self.pred_mode == "3d":
                all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError  
        conditions = [((conditions[i][0]), conditions[i][1][...,2:]) for i in range(len(conditions))]

        data = torch.tensor(np.stack(data, axis=0))

        data = data.repeat((num_samples, 1, 1))
        all_detections = list(all_detections) * num_samples
        conditions = list(conditions) * num_samples
        ball_chair_at_start = ball_chair_at_start * num_samples

        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        # Pass this to condition our models rather than pass them separately
        if self.mode == "post":
            ball_chair_at_start_tensor = torch.Tensor(ball_chair_at_start).to(global_device_name) if not isinstance(ball_chair_at_start[0], torch.Tensor) else torch.stack(ball_chair_at_start).float().to(global_device_name)
            global_dict = {"detections": detections.to(global_device_name), "red_start": ball_chair_at_start_tensor}
        elif self.mode == "pre":
            global_dict = {"detections": detections.to(global_device_name)}
        else:
            raise NotImplementedError
        return data, global_dict, conditions
        
    def dynamic_collate_fn(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only wheelchair states 
        data = [data[i][:,2:] for i in range(len(data))]
        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        conditions = [((conditions[i][0]), conditions[i][1][...,2:]) for i in range(len(conditions))]
        # Pass this to condition our models rather than pass them separately

        global_dict = {}

        return data, global_dict, conditions

    def dynamic_collate_fn_repeat(self, batch):
        num_samples = 5

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only wheelchair states 
        data = [data[i][:,2:] for i in range(len(data))]
        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        conditions = [((conditions[i][0]), conditions[i][1][...,2:]) for i in range(len(conditions))]
        # Pass this to condition our models rather than pass them separately

        data = data.repeat((num_samples, 1, 1))
        conditions = list(conditions) * num_samples

        global_dict = {}

        return data, global_dict, conditions

    def collate_fn_repeat(self):
        return pad_collate_detections_repeat


def pad_collate_detections_red_blue(batch):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    # all detections is [batch x n_agents x tensors]

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    n_agents = len(all_detections[0])
    detects = [[] for _ in range(n_agents)]
    for b in all_detections:
        for i in range(n_agents):
            detects[i].append(b[i])

    prisoner_at_start = torch.tensor(np.stack(prisoner_at_start, axis=0))

    global_dict = {"hideouts": global_cond, "red_start": prisoner_at_start}
    for i, d in enumerate(detects):
        x_lens = [len(x) for x in d]
        xx_pad = pad_sequence(d, batch_first=True, padding_value=0)
        ds = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)
        global_dict[f"d{i}"] = ds

    return data, global_dict, conditions

def pad_collate_detections_repeat_red_blue(batch, num_samples):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    data = data.repeat((num_samples, 1, 1))
    global_cond = global_cond.repeat((num_samples, 1, 1))
    conditions = list(conditions) * num_samples

    n_agents = len(all_detections[0])
    detects = [[] for _ in range(n_agents)]
    for b in all_detections:
        for i in range(n_agents):
            detects[i].append(b[i])

    prisoner_at_start = torch.tensor(np.stack(prisoner_at_start, axis=0))

    red_start = (prisoner_at_start).repeat((num_samples,1))

    global_dict = {"hideouts": global_cond, "red_start": red_start}
    for i, d in enumerate(detects):
        d_mult = d * num_samples
        x_lens = [len(x) for x in d_mult]
        xx_pad = pad_sequence(d_mult, batch_first=True, padding_value=0)
        ds = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)
        global_dict[f"d{i}"] = ds

    return data, global_dict, conditions

def pad_collate_detections_selHideout_red_blue(batch, num_samples_each_hideout):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    data = data.repeat((num_samples_each_hideout, 1, 1))

    global_cond = torch.tensor(global_cond[0])
    hideout_num = global_cond.shape[0]
    samples_num = num_samples_each_hideout * hideout_num
    global_cond = torch.cat([global_cond[i].repeat(num_samples_each_hideout, 1) for i in range(hideout_num)], dim=0)
    

    # all_detections = list(all_detections) * samples_num
    conditions = list(conditions) * samples_num
    conditions = [[np.concatenate((conditions[i][0], np.array([[-1]]))), np.concatenate((conditions[i][1], global_cond[i:i+1]*2-1))] for i in range(samples_num)]

    x_lens = [len(x) for x in all_detections]
    # xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    # detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    n_agents = len(all_detections[0])
    detects = [[] for _ in range(n_agents)]
    for b in all_detections:
        for i in range(n_agents):
            detects[i].append(b[i])

    # Pass this to condition our models rather than pass them separately
    # INFO: This is for red+blue trajs
    # global_dict = {"hideouts": global_cond, "detections": detections, "unpacked": torch.cat(all_detections, axis=0), "red_start": torch.Tensor(prisoner_at_start).repeat_interleave(repeats=samples_num, dim=0)}
    # INFO: This is for red traj only

    prisoner_at_start = torch.tensor(np.stack(prisoner_at_start, axis=0))

    # global_dict = {"hideouts": global_cond.to(global_device_name), 
    #     # "red_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=samples_num, dim=0)
    #     "red_start": prisoner_at_start.to(global_device_name).repeat((samples_num, 1))
    #     }

    red_start = prisoner_at_start.repeat((samples_num,1))
    global_dict = {"hideouts": global_cond, "red_start": red_start}
    for i, d in enumerate(detects):
        d_mult = d * samples_num
        x_lens = [len(x) for x in d_mult]
        xx_pad = pad_sequence(d_mult, batch_first=True, padding_value=0)
        ds = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)
        global_dict[f"d{i}"] = ds

    return data, global_dict, conditions

def pad_collate_detections(batch):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    x_lens = [len(x) for x in all_detections]
    xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    global_dict = {"hideouts": global_cond.to(global_device_name), "detections": detections.to(global_device_name), "red_start": torch.Tensor(prisoner_at_start).to(global_device_name)}

    return data, global_dict, conditions

def pad_collate_detections_multiHideout(batch, num_samples):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(global_cond[0])

    data = data.repeat((num_samples, 1, 1))

    hideout_ind_sel = torch.randint(low=0, high=global_cond.shape[0], size=(num_samples,))
    global_cond = global_cond[hideout_ind_sel,:]
    all_detections = list(all_detections) * num_samples
    conditions = list(conditions) * num_samples
    conditions = [[np.concatenate((conditions[i][0], np.array([[-1]]))), np.concatenate((conditions[i][1], global_cond[i:i+1]*2-1))] for i in range(num_samples)]

    # x_lens = [len(x) for x in all_detections]
    # xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    # detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    global_dict = {"hideouts": global_cond.to(global_device_name), 
        # "unpacked": torch.cat(all_detections, axis=0).to(global_device_name), 
            "red_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=num_samples, dim=0)}
    # global_dict = {"hideouts": global_cond, "detections": detections}

    return data, global_dict, conditions

def pad_collate_detections_selHideout(batch, num_samples_each_hideout):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    data = data.repeat((num_samples_each_hideout, 1, 1))

    global_cond = torch.tensor(global_cond[0])
    hideout_num = global_cond.shape[0]
    samples_num = num_samples_each_hideout * hideout_num
    global_cond = torch.cat([global_cond[i].repeat(num_samples_each_hideout, 1) for i in range(hideout_num)], dim=0)
    

    all_detections = list(all_detections) * samples_num
    conditions = list(conditions) * samples_num
    conditions = [[np.concatenate((conditions[i][0], np.array([[-1]]))), np.concatenate((conditions[i][1], global_cond[i:i+1]*2-1))] for i in range(samples_num)]

    # x_lens = [len(x) for x in all_detections]
    # xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    # detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    # INFO: This is for red+blue trajs
    # global_dict = {"hideouts": global_cond, "detections": detections, "unpacked": torch.cat(all_detections, axis=0), "red_start": torch.Tensor(prisoner_at_start).repeat_interleave(repeats=samples_num, dim=0)}
    # INFO: This is for red traj only

    prisoner_at_start = torch.tensor(np.stack(prisoner_at_start, axis=0))

    global_dict = {"hideouts": global_cond.to(global_device_name), 
        # "red_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=samples_num, dim=0)
        "red_start": prisoner_at_start.to(global_device_name).repeat((samples_num, 1))
        }

    return data, global_dict, conditions

def pad_collate_detections_repeat(batch, num_samples):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    data = data.repeat((num_samples, 1, 1))
    global_cond = global_cond.repeat((num_samples, 1))
    all_detections = list(all_detections) * num_samples
    conditions = list(conditions) * num_samples

    x_lens = [len(x) for x in all_detections]
    xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    # INFO: This is for red+blue trajs
    # global_dict = {"hideouts": global_cond, "detections": detections, "unpacked": torch.cat(all_detections, axis=0), "red_start": torch.Tensor(prisoner_at_start).repeat_interleave(repeats=num_samples, dim=0)}
    # INFO: This is for red traj only
    global_dict = {"hideouts": global_cond.to(global_device_name), 
        "red_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=num_samples, dim=0)}

    return data, global_dict, conditions

def get_lowest_root_folders(root_folder):
    lowest_folders = []
    
    # Get all items in the root folder
    items = os.listdir(root_folder)
    
    # Check if each item is a directory
    for item in items:
        item_path = os.path.join(root_folder, item)
        
        if os.path.isdir(item_path):
            # Recursively call the function for subfolders
            subfolders = get_lowest_root_folders(item_path)
            
            if not subfolders:
                # If there are no subfolders, add the current folder to the lowest_folders list
                lowest_folders.append(item_path)         
            lowest_folders.extend(subfolders)
    if len(lowest_folders) == 0:
        return [root_folder]
    return lowest_folders

class BallPositionDataset(torch.utils.data.Dataset):
    def __init__(self, directory, k=5, img_width=1280, img_height=720):
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]
        self.k = k
        self.img_width = img_width
        self.img_height = img_height
        self.data = []
        self._load_data()

    def _load_data(self):
        for file in self.files:
            ball_locs = np.load(file)['ball_locations']
            chair_locs = np.load(file)['chair_2d_locations']
            ball_locs = np.pad(ball_locs, ((self.k, self.k), (0, 0)), mode='edge')
            chair_locs = np.pad(chair_locs, ((self.k, self.k), (0, 0)), mode='edge')
            for i in range(self.k, len(ball_locs) - self.k):
                input_seq_ball = ball_locs[i-self.k:i]
                input_seq_chair = chair_locs[i:i+1]
                input_seq = np.concatenate((input_seq_chair, input_seq_ball), axis=0)
                output_seq = chair_locs[-1]
                self.data.append((input_seq, output_seq))

    def denormalize_data(self, data, width=1280, height=720):
        return data * np.array([width, height])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, output_seq = self.data[idx]
        input_seq = input_seq / [self.img_width, self.img_height]  # Normalize
        output_seq = output_seq / [self.img_width, self.img_height]  # Normalize

        input, output = torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)
        # print(input, output)
        return input, output

class NAgentsRewardDataset(torch.utils.data.Dataset):
    def __init__(self, traj_max_num=10192) -> None:
        super().__init__()
        self.traj_max_num = traj_max_num
        self.traj_num = 0
        self.traj_pt_counter = 0
        self.seq_len = 240
        self.max_steps = 350
        self.set_normalization_factors()
        self.agent_locations = []
        self.red_rewards = []
        self.hideout_loc = []
        self.dones = []
        self.traj_lens = []

    def set_normalization_factors(self):
        self.min_x = 0
        self.max_x = 2428
        self.min_y = 0
        self.max_y = 2428

    def normalize(self, arr):
        arr = np.array(arr).astype(float)

        x = arr[..., 0]
        arr[..., 0] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

        y = arr[..., 1]
        arr[..., 1] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        return arr
    
    def push(self, prisoner_loc, blue_locs, red_rew, done, red_hideout):
        normalized_prisonerLoc = self.normalize([prisoner_loc])
        normalized_blueLocs = self.normalize(blue_locs)
        agent_locs = np.concatenate((normalized_prisonerLoc, normalized_blueLocs), axis=0).reshape(-1)
        if self.traj_num >= self.traj_max_num:
            # INFO: pop out the first traj
            del self.agent_locations[:self.traj_lens[0]]
            del self.red_rewards[:self.traj_lens[0]]
            del self.dones[:self.traj_lens[0]]
            del self.hideout_loc[:self.traj_lens[0]]
            self.traj_lens.pop(0)
            self.traj_num = self.traj_num - 1
        if self.traj_num < self.traj_max_num:
            self.agent_locations.append(agent_locs)
            self.red_rewards.append(red_rew)
            self.dones.append(done)
            self.hideout_loc.append(red_hideout)
            self.traj_pt_counter = self.traj_pt_counter + 1
            if done:
                self.traj_num = self.traj_num + 1
                self.traj_lens.append(self.traj_pt_counter)
                self.traj_pt_counter = 0

    def __len__(self):
        return len(self.agent_locations)

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_len
        curr_batch_loc = self.agent_locations[start_idx:end_idx]
        curr_hideout_loc = self.hideout_loc[start_idx:end_idx]
        curr_batch_rew = self.red_rewards[start_idx:end_idx]
        if end_idx < len(self):
            traj_end_localized_step = np.where(self.dones[start_idx:end_idx])[0]
            # traj_start_localized_step = np.where(self.dones[start_idx-self.max_steps:start_idx])[-1]
            if len(traj_end_localized_step) != 0:
                step_loc = np.stack(curr_batch_loc[0:traj_end_localized_step[0]+1], axis=0)
                step_rew = np.stack(curr_batch_rew[0:traj_end_localized_step[0]+1], axis=0)
                hideout = curr_hideout_loc[traj_end_localized_step[0]]
                step_loc = np.pad(step_loc, ((0, self.seq_len-(traj_end_localized_step[0]+1)), (0, 0)), 'edge')
                step_rew = np.pad(step_rew, ((0, self.seq_len-(traj_end_localized_step[0]+1)), (0, 0)), 'constant')
            else:
                step_loc = np.stack(curr_batch_loc, axis=0)
                step_rew = np.stack(curr_batch_rew, axis=0)
                hideout = curr_hideout_loc[-1]
            # # INFO: 
            # for i in range(self.max_steps):
            #     if self.dones[start_idx-i] and i != 0:
            #         break
        else:
            if len(curr_batch_loc) != 0:
                step_loc = np.stack(curr_batch_loc, axis=0)
                step_rew = np.stack(curr_batch_rew, axis=0)
                hideout = curr_hideout_loc[-1]
                step_loc = np.pad(step_loc, ((0, end_idx-len(self)), (0, 0)), 'edge')
                step_rew = np.pad(step_rew, ((0, end_idx-len(self)), (0, 0)), 'constant')
            else:
                step_loc = np.stack([self.agent_locations[-1]], axis=0)
                step_rew = np.stack([self.red_rewards[-1]], axis=0)
                hideout = self.hideout_loc[-1]
                step_loc = np.pad(step_loc, ((0, self.seq_len-1), (0, 0)), 'edge')
                step_rew = np.pad(step_rew, ((0, self.seq_len-1), (0, 0)), 'constant')

        condition = (np.array([]), np.array([]))
        prisoner_at_start = np.concatenate((np.array([0]), np.array(step_loc[0,:2])))
        return step_loc, step_rew, hideout, condition, torch.Tensor(prisoner_at_start)

    def collate_loc_reward(self):
        return pad_loc_reward

    def collate_loc(self):
        return pad_loc

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

def pad_loc_reward(batch, gamma, period):
    step_loc, step_rew, _, _, _ = zip(*batch)

    batches_seqLen_agentLocations = torch.Tensor(np.stack(step_loc, axis=0)).to(global_device_name)
    red_rews = torch.Tensor(np.stack(step_rew, axis=0)).squeeze()
    seq_len = batches_seqLen_agentLocations.shape[1]
    discount_factors = torch.Tensor([gamma**i for i in range(seq_len)])
    batches_seqLen_redRews = torch.sum(red_rews*discount_factors, axis=-1, keepdim=True).to(global_device_name)

    return batches_seqLen_agentLocations[:,::period,:2], batches_seqLen_redRews

def pad_loc(batch):
    step_loc, step_rew, hideout, condition, prisoner_at_start = zip(*batch)
    batches_seqLen_agentLocations = torch.Tensor(np.stack(step_loc, axis=0)).to(global_device_name)
    hideout = torch.stack(hideout, dim=0).to(global_device_name)
    prisoner_at_start = torch.stack(prisoner_at_start, dim=0).to(global_device_name)
    # INFO: construct the global condition
    global_dict = {"hideouts": hideout, "red_start": prisoner_at_start}
    return batches_seqLen_agentLocations[:,:,:2], global_dict, condition

def update_raw_traj(raw_red_downsampled_traj, detected_blue_states, red_vel, perception_max_thresh=0.1, perception_min_thresh=0.05):
    raw_red_downsampled_traj = copy.deepcopy(raw_red_downsampled_traj)
    repulse_vec = torch.zeros_like(raw_red_downsampled_traj).to(global_device_name)
    for detects in detected_blue_states:
        detect_loc = torch.Tensor([detects[0]]).to(global_device_name)
        detect_vel = torch.Tensor(detects[1]).to(global_device_name)

        blue_to_pt = raw_red_downsampled_traj - detect_loc
        dist_from_blue_to_pt = torch.norm(blue_to_pt, dim=-1, keepdim=True)
        dist_from_blue_to_pt[dist_from_blue_to_pt>perception_max_thresh] = 1e6
        dist_from_blue_to_pt[dist_from_blue_to_pt<perception_min_thresh] = 0.05

        # INFO: vertical to relative vel
        relative_vel = detect_vel - red_vel
        if relative_vel[0] == 0 and relative_vel[1] == 0:
            repulse_direction_vec = blue_to_pt
            repulse_direction_vec_normalized = repulse_direction_vec / torch.norm(repulse_direction_vec, dim=-1, keepdim=True)
            repulse_vec = repulse_vec + 0.001 * repulse_direction_vec_normalized / dist_from_blue_to_pt
        else:
            repulse_direction_vec = blue_to_pt - torch.inner(blue_to_pt, relative_vel).unsqueeze(-1) / torch.norm(relative_vel) @ (relative_vel / torch.norm(relative_vel)).unsqueeze(0)
            repulse_direction_vec_normalized = repulse_direction_vec / torch.norm(repulse_direction_vec, dim=-1, keepdim=True)
            repulse_vec = repulse_vec + 0.001 * repulse_direction_vec_normalized / dist_from_blue_to_pt        

        # INFO: vertical to abs vel
        # if detect_vel[0] == 0 and detect_vel[1] == 0:
        #     repulse_direction_vec = blue_to_pt
        #     repulse_direction_vec_normalized = repulse_direction_vec / torch.norm(repulse_direction_vec, dim=-1, keepdim=True)
        #     repulse_vec = repulse_vec + 0.001 * repulse_direction_vec_normalized / dist_from_blue_to_pt
        # else:
        #     repulse_direction_vec = blue_to_pt - torch.inner(blue_to_pt, detect_vel).unsqueeze(-1) / torch.norm(detect_vel) @ (detect_vel / torch.norm(detect_vel)).unsqueeze(0)
        #     repulse_direction_vec_normalized = repulse_direction_vec / torch.norm(repulse_direction_vec, dim=-1, keepdim=True)
        #     repulse_vec = repulse_vec + 0.001 * repulse_direction_vec_normalized / dist_from_blue_to_pt

        raw_red_downsampled_traj = raw_red_downsampled_traj + repulse_vec
    return raw_red_downsampled_traj


if __name__ == "__main__":

    # data_path = "/home/sean/october_datasets/multiagent/rectilinear"
    # data_path = "/home/sean/PrisonerEscape/datasets/multiagent/rectilinear"
    # data_path = "/home/sean/october_datasets/multiagent/sinusoidal"
    data_path = "/home/wu/Research/Diffuser/data/prisoner_datasets/october_datasets/gnn_map_0_run_600_AStar_only_dr"

    # data_path = "/home/sean/PrisonerEscape/datasets/multiagent/AStar"
    dataset = NAgentsSingleDataset(data_path,                  
                 horizon = 60,
                 normalizer = None,
                 global_lstm_include_start=False,
                 condition_path = False)
    
    # print(dataset[0])

    def cycle(dl):
        while True:generate_path_samples
    train_batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True, collate_fn=dataset.collate_fn())

    for i in dataloader:
        pass

    # for i in range(len(dataset)):
    #     print(dataset[i])