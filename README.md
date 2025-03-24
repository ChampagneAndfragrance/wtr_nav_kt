# wtr_nav_kt
## Code Structure
The main code is in the folder `wheelchair`.

Currently, you have extracted the agent trajectories from the videos and save them into a folder - e.g. `./splited_raw_trajs/`.

The file `traj_preprocess_main.py` is used to process the raw trajectories extracted from the videos into a trainable format. 

The file `diffusion_image_space.py` is used to train the diffusion policy for the wheelchair. We load the data into the dataloader `BallWheelchairJointDataset` and then load the `GaussianDiffusion` to train the wheelchair imitation policy.

You can set the configuration file  `chair_diffusion_train.yaml` in the folder `./config/diffusion_config`. The trained model will be save in `./ICRA_2025_logs/...`.

## Citations
* We adapt our diffusion model training code from [`Diffuser`](https://github.com/jannerm/diffuser) with the following citation:
```c
@inproceedings{janner2022diffuser,
  title = {Planning with Diffusion for Flexible Behavior Synthesis},
  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}
```

* If you find our work helpful, please cite as:
```c
@misc{wu2024learningwheelchairtennisnavigation,
      title={Learning Wheelchair Tennis Navigation from Broadcast Videos with Domain Knowledge Transfer and Diffusion Motion Planning}, 
      author={Zixuan Wu and Zulfiqar Zaidi and Adithya Patil and Qingyu Xiao and Matthew Gombolay},
      year={2024},
      eprint={2409.19771},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.19771}, 
}
```
