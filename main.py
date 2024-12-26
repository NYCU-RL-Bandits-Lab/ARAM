import os
import argparse
from datetime import datetime
import gym
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
import random
import time
import wandb
from environments import reacher_v3_L2_005
from environments import half_cheetah_v3_O_20, half_cheetah_v3_O_20_ccw01
from environments import hopper_v3_M_10_goal_vel3
from environments import ant_v3_L2_2_goal_forward_ccw05
from environments import humanoid_v3_L2_2, humanoid_v3_O30, humanoid_v3_M30
from environments import gym_BSS_3zone, gym_BSS_5zone
from environments.NSFnet.NSFnet_multiV2 import SimulatedNetworkEnv
from environments.NSFnet.NSFnet_multiV3 import SimulatedNetwork20dEnv


import safety_gym

from agent import SacAgent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='MO_half_cheetah-v0')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--ver_number', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prefer', type=int, default=4)
    parser.add_argument('--buf_num', type=int, default=4)
    parser.add_argument('--q_freq', type=int, default=1000)
    parser.add_argument('--training_steps', type=int, default=1500000)
    parser.add_argument('--eval_interval', type=int, default=10000)    
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--model_saved_step', type=int, default=10000)
    parser.add_argument('--action_sample_number', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--augement_action_sample_number', type=int, default=100)
    parser.add_argument('--augment_ratio', type=float, default=0.2)
    parser.add_argument('--augment_ratio_decay', type=float, default=0.99)
    parser.add_argument('--augment_ratio_decay_freq', type=int, default=10000)
    parser.add_argument('--penalty_weight', type=float, default=0.2)
    parser.add_argument("--prob_id", action="store", default = "")
    parser.add_argument("--log_dir", action="store", default="tmp")
    parser.add_argument('--entropy_tuning', action='store_true', default=False)
    parser.add_argument('--eval_episode', type=int, default=10)
    parser.add_argument("--wandb-project-name", type=str, default="sb3", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="rl_redway", help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-info", type=str, default="rl_redway", help="the info of wandb's project")
    parser.add_argument("-tags", "--wandb-tags", type=str, default=[], nargs="+", help="Tags for wandb run, e.g.: -tags optimized pr-123")
    parser.add_argument('--hidden_sizes', type=int, nargs=2, default=[256, 256], help='The hidden units configuration')
    parser.add_argument('--lr', type=float, default=0.0003)


    args = parser.parse_args()

    # You can define configs in the external json or yaml file.
    configs = {
        'num_steps': args.training_steps,
        'batch_size': args.batch_size,#256
        'lr': args.lr,
        'hidden_units': args.hidden_sizes,
        'memory_size': 100,
        'prefer_num': args.prefer,
        'buf_num': args.buf_num,
        'gamma': args.gamma,
        'tau': args.tau,
        'entropy_tuning': args.entropy_tuning,
        'ent_coef': 0.2,  # It's ignored when entropy_tuning=True.
        'multi_step': 1,
        'per': False,  # prioritized experience replay
        'alpha': 0.6,  # It's ignored when per=False.
        'beta': 0.4,  # It's ignored when per=False.
        'beta_annealing': 0.0001,  # It's ignored when per=False.
        'grad_clip': None,
        'updates_per_step': 1,
        'start_steps': args.start_steps,
        'log_interval': 10,
        'target_update_interval': 1,
        'eval_interval': args.eval_interval,
        'eval_episode' : args.eval_episode,
        'cuda': args.cuda,
        'seed': args.seed,
        'cuda_device': args.cuda_device,
        'q_frequency': args.q_freq,
        'action_sample_number': args.action_sample_number,
        'augement_action_sample_number': args.augement_action_sample_number,
        'augment_ratio': args.augment_ratio,
        'penalty_weight': args.penalty_weight,
        'augment_ratio_decay': args.augment_ratio_decay,
        'augment_ratio_decay_freq': args.augment_ratio_decay_freq,
        'prob_id': args.prob_id,
        'model_saved_step': args.model_saved_step
    }
    
    env = gym.make(args.env_id)
    
    log_dir = os.path.join(
        args.log_dir, args.env_id,
        f'MOSAC-set{args.prefer}-buf{args.buf_num}-seed{args.seed}_freq{args.q_freq}')
    run_name = f"{args.env_id}__seed{args.seed}__{int(time.time())}__{args.prob_id}__MOSAC__buf{args.buf_num}__set{args.prefer}__{args.action_sample_number}__{args.augement_action_sample_number}__{args.augment_ratio}__ver{args.ver_number}__{args.wandb_info}"
    tags = args.wandb_tags
    wandb.login(key='901bb9f061cd7a361b05f301d6b5240177dcaaf9')
    run = wandb.init(
        name=run_name,
        dir='/SSD2/redoao/wandb',
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        tags=tags,
        config=vars(args),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    agent = SacAgent(env=env, log_dir=log_dir, **configs)
    agent.run()



if __name__ == '__main__':
    run()