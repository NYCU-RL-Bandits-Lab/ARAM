import os
import argparse
from datetime import datetime
import gym
import torch
import numpy as np
import random
import time
import wandb
import matplotlib.pyplot as plt
from matplotlib import animation
import cv2
from environments import half_cheetah_v3_O_20, ant_v3_L2_2_goal_forward_ccw05
import Constraint_Proj
import Constraint_Check

from agent import SacAgent
import os
from model import TwinnedQNetwork, GaussianPolicy

REPLAY_PATH = "./videos/ARAM"

def save_frames_as_gif(frames, path='./', filename='operator.gif'):
    frames = frames[2:]
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, fps=20)
    
RES = (70,70,1800,1130)

cuda = True
device = torch.device("cpu")

def explore(state, preference):
        # act with noisy
        state = torch.FloatTensor(state).unsqueeze(0)
        preference = torch.FloatTensor(preference).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)

def exploit(state, preference):
        # act without noisy
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        preference = preference.clone().detach().to(device).unsqueeze(0)
        with torch.no_grad():
            _, _, action = policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)


def constraintViolation_Proj(prob_id, observations, actions):
    if prob_id == "HC+N" or prob_id == "W+N" or prob_id == "H+N" or prob_id == "S+N" or prob_id == "Re+N" or prob_id == "MA_umaze+N" or prob_id == "MA_medium+N" or prob_id == "Pu+N" or prob_id == "An+N":
        actions = Constraint_Proj.Projection_X_N(observations, actions)
    elif prob_id == "Net+N":
        actions = Constraint_Proj.Projection_X_N(observations, actions)
    elif prob_id == "Re+L2_01":
        actions = Constraint_Proj.Projection_Re_L2_01(observations, actions)
    elif prob_id == "HC+O20" or prob_id == "HC+O20_ver3" or prob_id == "HC+O_20" or prob_id == "HC+O_20_ver3":
        actions = Constraint_Proj.Projection_HC_O20(observations, actions)
    elif prob_id == "HC+O10" or prob_id == "HC+O10_ver3" or prob_id == "HC+O_10" or prob_id == "HC+O_10_ver3":
        actions = Constraint_Proj.Projection_HC_O10(observations, actions)        
    elif prob_id == "HC+M_10" or prob_id == "M_10_ver3":
        actions = Constraint_Proj.Projection_HC_M10(observations, actions)
    elif prob_id == "H+M_10" or prob_id == "H+M_10_ver3":
        actions = Constraint_Proj.Projection_H_M_10(observations, actions)
    elif prob_id == "W+M_10" or prob_id == "W+M_10_ver3":
        actions = Constraint_Proj.Projection_W_M10(observations, actions)
    elif prob_id == "W+M_5" or prob_id == "W+M_5_ver3":
        actions = Constraint_Proj.Projection_W_M5(observations, actions)
    elif prob_id == "An+L2_2" or prob_id == "An+L2_2_ver3":
        actions = Constraint_Proj.Projection_An_L2_2(observations, actions)      
    elif prob_id == "BSS5z+S+D40":
        actions = Constraint_Proj.Projection_BSS5z_S_D40(observations, actions)            
    elif prob_id == "NSFnetV2+S":
        actions = Constraint_Proj.Projection_NSFnet_ver2(observations, actions)
    return actions

def constraintViolation_Check(prob_id, state, action):
    if prob_id == "HC+O20" or prob_id == "HC+O20_ver3" or prob_id == "HC+O_20" or prob_id == "HC+O_20_ver3":
        return Constraint_Check.Check_HC_O20(state, action)
    elif prob_id == "HC+O10" or prob_id == "HC+O10_ver3" or prob_id == "HC+O_10" or prob_id == "HC+O_10_ver3":
        return Constraint_Check.Check_HC_O10(state, action)        
    elif prob_id == "HC+M_10" or prob_id == "M_10_ver3":
        return Constraint_Check.Check_HC_M10(state, action)
    elif prob_id == "Re+L2_01":
        return Constraint_Check.Check_L2_01(state, action)
    elif prob_id == "Re+L2_005" or prob_id == "Re+L2_005_ver3":
        return Constraint_Check.Check_L2_005(state, action)
    elif prob_id == "Re+S_lr_L2_005" or prob_id == "Re+S_lr_L2_005_ver3":
        return Constraint_Check.Check_Re_S_lr_L2_005(state, action)
    elif prob_id == "H+L2_05" or prob_id == "H+L2_05_ver3":
        return Constraint_Check.Check_L2_05(state, action)
    elif prob_id == "H+L2_1" or prob_id == "H+L2_1_ver3":
        return Constraint_Check.Check_L2_1(state, action)
    elif prob_id == "H+M_10" or prob_id == "H+M_10_ver3":
        return Constraint_Check.Check_H_M10(state, action)
    elif prob_id == "W+M_10" or prob_id == "W+M_10_ver3":
        return Constraint_Check.Check_W_M10(state, action)
    elif prob_id == "W+M_5" or prob_id == "W+M_5_ver3":
        return Constraint_Check.Check_W_M5(state, action)
    elif prob_id == "An+O_20" or prob_id == "An+O_20_ver3":
        return Constraint_Check.Check_An_O20(state, action)
    elif prob_id == "An+O_30" or prob_id == "An+O_30_ver3":
        return Constraint_Check.Check_An_O30(state, action)
    elif prob_id == "An+L2_2" or prob_id == "An+L2_2_ver3":
        return Constraint_Check.Check_An_L2_2(state, action)
    elif prob_id == "HC+N" or prob_id == "Re+N" or prob_id == "MA_umaze+N" or prob_id == "MA_umaze+N" or prob_id == "Pu+N" or prob_id == "An+N":
        return Constraint_Check.Check_X_N(state, action)
    elif prob_id == "Net+N" or prob_id == "Re+N" or prob_id == "MA_umaze+N" or prob_id == "MA_umaze+N" or prob_id == "Pu+N" or prob_id == "An+N":
        return Constraint_Check.Check_X_N(state, action)
    elif prob_id == "Re+N":
        return Constraint_Check.Check_X_N(state, action)
    elif prob_id == "BSS3z+S":
        return Constraint_Check.Check_BSS3z_S(state, action)
    elif prob_id == "BSS5z+S":
        return Constraint_Check.Check_BSS5z_S(state, action)        
    elif prob_id == "BSS5z+S+D40":
        return Constraint_Check.Check_BSS5z_S_D40(state, action)
    elif prob_id == "BSS5z+S2":
        return Constraint_Check.Check_BSS5z_S2(state, action)
    elif prob_id == "BSS5z+S2+D40":
        return Constraint_Check.Check_BSS5z_S2_D40_ver2(state, action)   
    elif prob_id == "NSFnetV2+S":
        return Constraint_Check.Check_NSFnet_ver2(state, action)   



def wrapper_action_eval(prob_id, state, action):
    if prob_id == "BSS3z+S" or prob_id == "BSS5z+S":
        if(np.sum(action)>5):
            return constraintViolation_Proj(state, action)
        else:
            if prob_id == "BSS3z+S":
                action = 15 / 2 * action + 55 / 2
                return action
            else:
                action = 25 / 2 * action + 45 / 2
                return action
    elif prob_id == "BSS5z+S+D40":
        action = 20 * action + 20
        return action
    else:
        return action


def action_wrap_inter(prob_id, state, action):
    if prob_id == "BSS3z+S":
        action[0] = np.round(action[0])
        action[1] = np.round(action[1])
        action[2] = np.round(90 - action[0] - action[1])
        return action
    elif prob_id == "BSS5z+S2+D40" or prob_id == "BSS5z+S+D40":
        return Constraint_Proj.Projection_BSS5z_S2_INT40(state, action)
    else:
        return Constraint_Proj.Projection_BSS5z_S2_INT35(state, action)

parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type=str, default='MO_ant_L2_2_goal_forward_ccw05-v0')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--re', action='store_true', default=False)
parser.add_argument('--so_env', action='store_true', default=False)
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--prefer', type=int, default=4)
parser.add_argument('--buf_num', type=int, default=4)
parser.add_argument('--q_freq', type=int, default=1000)
parser.add_argument("--prob_id", action="store", default = "An+L2_2")
parser.add_argument("--model_path", action="store", default="tmp")
parser.add_argument('--eval_episode', type=int, default=5)
parser.add_argument('--pref', type=float, nargs='+', default=[0.9, 0.1])

args = parser.parse_args()

env_name = args.env_id

prob_id = args.prob_id

ref = args.ref_point

env = gym.make(env_name)
env.seed(args.seed)
reward_num = 2

policy = GaussianPolicy(env.observation_space.shape[0] + reward_num ,env.action_space.shape[0],[256,256]).to(device)

policy.load_state_dict(torch.load(args.model_path))

state=env.reset()
env.continuous = True
step = 0
epi = 0
epi_num = args.eval_episode

done = False
p = args.pref
preference = torch.tensor( p,dtype=torch.float32  )
episode_rewards = np.zeros((epi_num ,reward_num))
episode_reward = np.zeros(reward_num)
state = env.reset()
frames = []
count_reward2 = 0
for j in range(epi_num):
    done = False
    reward2_total = 0
    step = 0
    while not done:
        sample_count = 0
        reward2 =0
        reward2_total = 0
        step += 1
        while(sample_count<args.augement_action_sample_number):
            sample_count += 1 
            action = explore(state,preference)
            violate_check, _ = constraintViolation_Check(args.prob_id, state, action)
            if(violate_check):
                reward2_total += 1
            else:
                reward2 += 1
        action = constraintViolation_Proj(args.prob_id, state, action)
        action = wrapper_action_eval(args.prob_id, state, action)
        _, tes = constraintViolation_Check(args.prob_id, state, action)
        next_state, reward, done, info = env.step(action)
        if args.so_env:
            reward = np.array([reward, 0])
        reward[0] = reward[0] 
        reward[1] = reward2_total
        if args.re:
            env.render()
        episode_reward += reward
        state = next_state

        if done:
            state = env.reset()
            episode_reward[1]/=step
            print('reward', p)
            episode_rewards[j] = episode_reward
            print('Steps:',step,'Reward:',episode_reward)
            print('='*70)
            episode_reward = np.zeros(reward_num)
            step = 0

save_frames_as_gif(frames, path=REPLAY_PATH, filename=args.env_id+".gif")