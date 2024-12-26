import os
import numpy as np
import torch
import wandb
import copy
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory, PrioritizedMemory
from base import QMemory

from model import TwinnedQNetwork, GaussianPolicy
from utils import grad_false, hard_update, soft_update, to_batch,\
    update_params, RunningMeanStats

import Constraint_Proj
import Constraint_Check

import random
from multi_step import *
from datetime import datetime
import time
import copy

p_name= ['9505','9010','8515','8020','7525','7030','6535','6040','5545','5050','4555','4060','3565','3070','2575','2080','1585','1090','0595']
PREF = [[0.9, 0.1],  [0.5, 0.5], [0.1,0.9],]
        
class SacAgent:

    def __init__(self, env, log_dir, num_steps=3000000, batch_size=256, 
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6, prefer_num = 8, buf_num = 0,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 eval_interval=1000, eval_episode=10, cuda=True, seed=0, cuda_device=0, 
                 q_frequency=1000, prob_id="Re+L2_005_ver3", augment_ratio = 0.5, augment_ratio_decay = 0.99, augment_ratio_decay_freq = 10000, penalty_weight = 1, augement_action_sample_number = 100, action_sample_number = 100, model_saved_step=100000):
        self.env = env

        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False
        self.q_frequency = q_frequency
        
        self.device = torch.device(
            "cuda:"+str(cuda_device) if cuda and torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        print(torch.cuda.is_available())
        print(self.device)
        print(self.env.observation_space.shape[0])
        self.reward_num = 2
        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0]+self.reward_num,
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.reward_num,
            hidden_units=hidden_units).to(self.device)
        self.critic_target = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.reward_num,
            hidden_units=hidden_units).to(self.device).eval()
        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(
                self.env.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            # replay memory without prioritied experience replay
            self.memory = MOMultiStepMemory(
                memory_size, self.env.observation_space.shape, self.reward_num,
                self.env.action_space.shape, self.device, gamma, multi_step)
            self.augumented_memory = MOMultiStepMemory(
                memory_size, self.env.observation_space.shape, self.reward_num,
                self.env.action_space.shape, self.device, gamma, multi_step)
        self.critic_update_time = 0
        self.policy_update_time = 0
        self.critic_loss_time = 0
        self.policy_loss_time = 0
        self.mujoco_time = 0
        self.gp_time = 0
        self.sample_action_time = 0
        self.counter_time = time.perf_counter()
        self.eval_time = 0
        #Q Replay Buffer
        self.Q_memory = QMemory(buf_num)
        self.cur_p = 0
        self.cur_e = 0
        self.qmem_p = 0
        self.qmem_e = 0

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.monitor = []
        self.tot_t = []
        self.reward_v = []
        PREF_ = PREF
        for i in PREF_:
            self.tot_t.append([])
            self.reward_v.append([])

        self.set_num = prefer_num 
        
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.eval_episode = eval_episode
        self.action_sample_number = action_sample_number
        self.augement_action_sample_number = augement_action_sample_number
        self.augment_ratio = augment_ratio
        self.augment_ratio_decay = augment_ratio_decay
        self.augment_ratio_decay_freq = augment_ratio_decay_freq
        self.penalty_weight = penalty_weight
        self.model_saved_step = model_saved_step
        self.prob_id = prob_id
        self.action_number = 1
        self.eval_sample_number = 1
        self.mujoco = False
        self.goal_env = False
        self.so_env = False
        self.hard_env = False
        self.safe_env = False
        if self.prob_id == "HC+N" or self.prob_id == "HC+O20" or self.prob_id == "HC+O20_ver3" or self.prob_id == "HC+O10" or self.prob_id == "HC+O10_ver3" \
         or self.prob_id == "HC+O_10" or self.prob_id == "HC+O_10_ver3" or self.prob_id == "HC+O_20" or self.prob_id == "HC+O_20_ver3" \
         or self.prob_id == "HC+M_10" or self.prob_id == "HC+M_10_ver3" or self.prob_id == "HC+O_5":
            penalty = -1
            reward_offset1 = 1
            reward_offset2 = 0
            self.mujoco = True
        elif self.prob_id == "S+N" or self.prob_id == "S+L2_01" or self.prob_id == "S+L2_01_ver3" or self.prob_id == "S+L2_05" or self.prob_id == "S+L2_1":
            penalty = -1
            reward_offset1 = 1
            reward_offset2 = 0
            self.mujoco = True
        elif self.prob_id == "Re+N" or self.prob_id == "Re+L2_01" or self.prob_id == "Re+L2_005"  or self.prob_id == "Re+L2_005_ver3" or self.prob_id == "Re+S_lr_L2_005" or self.prob_id == "Re+S_lr_L2_005_ver3":
            penalty = -1
            reward_offset1 = 1
            reward_offset2 = 1
            self.mujoco = True
        elif self.prob_id == "H+N" or self.prob_id == "H+L2_05" or self.prob_id == "H+L2_05_ver3" or self.prob_id == "H+L2_1" \
        or self.prob_id == "H+L2_1_ver3" or self.prob_id == "H+M_10" or self.prob_id == "H+M_10_ver3":
            penalty = -1
            reward_offset1 = 1
            reward_offset2 = 0
            self.mujoco = True
        elif self.prob_id == "W+N" or self.prob_id == "W+M_10" or self.prob_id == "W+M_10_ver3" or self.prob_id == "W+M_5" or self.prob_id == "W+M_5_ver3":
            penalty = -1
            reward_offset1 = 1
            reward_offset2 = 0
            self.mujoco = True
        elif self.prob_id == "An+N" or self.prob_id == "An+O_20" or self.prob_id == "An+O_20_ver3" or self.prob_id == "An+O_30" or self.prob_id == "An+O_30_ver3" or self.prob_id == "An+L2_2" or self.prob_id == "An+L2_2_ver3":
            penalty = -1
            reward_offset1 = 1
            reward_offset2 = 0
            self.mujoco = True
        elif self.prob_id == "BSS3z+S" or self.prob_id == "BSS3z+S+D40" :
            penalty = -1
            reward_offset1 = 20
            reward_offset2 = 1
            self.hard_env = False   
        elif self.prob_id == "BSS5z+S" or self.prob_id == "BSS5z+S2" or self.prob_id == "BSS5z+S2+D40" or self.prob_id == "BSS5z+S+D40" or self.prob_id == "BSS5z+S+D35":
            penalty = -1
            reward_offset1 = 100
            reward_offset2 = 2
            self.hard_env = False      
        elif self.prob_id == "Net+N":
            penalty = -1
            reward_offset1 = 100
            reward_offset2 = 0        
        elif self.prob_id == "NSFnetV2+S":
            penalty = -1
            reward_offset1 = 10
            reward_offset2 = 0
            self.so_env = True
        elif self.prob_id == "NSFnetV3+S":
            penalty = -1
            reward_offset1 = 10
            reward_offset2 = 0
            self.so_env = True



        self.penalty = penalty * self.penalty_weight
        self.reward_offset1 = reward_offset1
        self.reward_offset2 = reward_offset2
        if self.prob_id == "Point+Safe" or self.prob_id == "Point+Safe2" or self.prob_id == "Point+Safe3":
            self.max_episode_steps = 1000
        else:
            self.max_episode_steps = self.env.max_episode_steps

    def constraintViolation_Proj(self, observations, actions):
        if self.prob_id == "HC+N" or self.prob_id == "W+N" or self.prob_id == "H+N" or self.prob_id == "S+N" or self.prob_id == "Re+N" or self.prob_id == "MA_umaze+N" or self.prob_id == "MA_medium+N" or self.prob_id == "An+N":
            actions = Constraint_Proj.Projection_X_N(observations, actions)
        elif self.prob_id == "Net+N" or self.prob_id == "Re+N" or self.prob_id == "Pu+N" or self.prob_id == "Pandp+N" or self.prob_id == "Sl+N":
            actions = Constraint_Proj.Projection_X_N(observations, actions)
        elif self.prob_id == "S+L2_01" or self.prob_id == "S+L2_01_ver3":
            actions = Constraint_Proj.Projection_S_L2_01(observations, actions)
        elif self.prob_id == "S+L2_05":
            actions = Constraint_Proj.Projection_S_L2_05(observations, actions)
        elif self.prob_id == "S+L2_1":
            actions = Constraint_Proj.Projection_S_L2_1(observations, actions)
        elif self.prob_id == "Re+L2_01":
            actions = Constraint_Proj.Projection_Re_L2_01(observations, actions)
        elif self.prob_id == "Re+L2_005" or self.prob_id == "Re+L2_005_ver3":
            actions = Constraint_Proj.Projection_Re_L2_005(observations, actions)
        elif self.prob_id == "Re+S_lr_L2_005" or self.prob_id == "Re+S_lr_L2_005_ver3":
            actions = Constraint_Proj.Projection_Re_S_lr_L2_005(observations, actions)
        elif self.prob_id == "HC+O20" or self.prob_id == "HC+O20_ver3" or self.prob_id == "HC+O_20" or self.prob_id == "HC+O_20_ver3":
            actions = Constraint_Proj.Projection_HC_O20(observations, actions)
        elif self.prob_id == "HC+O10" or self.prob_id == "HC+O10_ver3" or self.prob_id == "HC+O_10" or self.prob_id == "HC+O_10_ver3":
            actions = Constraint_Proj.Projection_HC_O10(observations, actions)        
        elif self.prob_id == "HC+O_5" or self.prob_id == "HC+O_5_ver3":
            actions = Constraint_Proj.Projection_HC_O5(observations, actions)        
        elif self.prob_id == "HC+M_10" or self.prob_id == "M_10_ver3":
            actions = Constraint_Proj.Projection_HC_M10(observations, actions)
        elif self.prob_id == "H+L2_05" or self.prob_id == "H+L2_05_ver3":
            actions = Constraint_Proj.Projection_H_L2_01(observations, actions)
        elif self.prob_id == "H+L2_1" or self.prob_id == "H+L2_1_ver3":
            actions = Constraint_Proj.Projection_H_L2_1(observations, actions)
        elif self.prob_id == "H+M_10" or self.prob_id == "H+M_10_ver3":
            actions = Constraint_Proj.Projection_H_M_10(observations, actions)
        elif self.prob_id == "W+M_10" or self.prob_id == "W+M_10_ver3":
            actions = Constraint_Proj.Projection_W_M10(observations, actions)
        elif self.prob_id == "W+M_5" or self.prob_id == "W+M_5_ver3":
            actions = Constraint_Proj.Projection_W_M5(observations, actions)
        elif self.prob_id == "An+O_20" or self.prob_id == "An+O_20_ver3":
            actions = Constraint_Proj.Projection_An_O20(observations, actions)
        elif self.prob_id == "An+O_30" or self.prob_id == "An+O_30_ver3":
            actions = Constraint_Proj.Projection_An_O30(observations, actions)
        elif self.prob_id == "An+L2_2" or self.prob_id == "An+L2_2_ver3":
            actions = Constraint_Proj.Projection_An_L2_2(observations, actions)
        elif self.prob_id == "BSS3z+S":
            actions = Constraint_Proj.Projection_BSS3z_S(observations, actions)
        elif self.prob_id == "BSS5z+S":
            actions = Constraint_Proj.Projection_BSS5z_S(observations, actions)         
        elif self.prob_id == "BSS2z+S+D40":
            actions = Constraint_Proj.Projection_BSS3z_S_D40(observations, actions)       
        elif self.prob_id == "BSS5z+S+D35":
            actions = Constraint_Proj.Projection_BSS5z_S_D35(observations, actions)
        elif self.prob_id == "BSS5z+S+D40":
            actions = Constraint_Proj.Projection_BSS5z_S_D40(observations, actions)
        elif self.prob_id == "BSS5z+S2":
            actions = Constraint_Proj.Projection_BSS5z_S2(observations, actions)        
        elif self.prob_id == "BSS5z+S2+D40":
            actions = Constraint_Proj.Projection_BSS5z_S2_D40_ver2(observations, actions)        
        elif self.prob_id == "NSFnetV2+S":
            actions = Constraint_Proj.Projection_NSFnet(observations, actions)
        elif self.prob_id == "NSFnetV3+S":
            actions = Constraint_Proj.Projection_NSFnetV3(observations, actions)        
        return actions

    def constraintViolation_Check(self, state, action):
        if self.prob_id == "HC+O20" or self.prob_id == "HC+O20_ver3" or self.prob_id == "HC+O_20" or self.prob_id == "HC+O_20_ver3":
            return Constraint_Check.Check_HC_O20(state, action)
        elif self.prob_id == "HC+O10" or self.prob_id == "HC+O10_ver3" or self.prob_id == "HC+O_10" or self.prob_id == "HC+O_10_ver3":
            return Constraint_Check.Check_HC_O10(state, action)        
        elif self.prob_id == "HC+O_5" or self.prob_id == "HC+O_5_ver3":
            return Constraint_Check.Check_HC_O5(state, action)     
        elif self.prob_id == "HC+M_10" or self.prob_id == "M_10_ver3":
            return Constraint_Check.Check_HC_M10(state, action)
        elif self.prob_id == "Re+L2_01":
            return Constraint_Check.Check_L2_01(state, action)
        elif self.prob_id == "Re+L2_005" or self.prob_id == "Re+L2_005_ver3":
            return Constraint_Check.Check_L2_005(state, action)
        elif self.prob_id == "Re+S_lr_L2_005" or self.prob_id == "Re+S_lr_L2_005_ver3":
            return Constraint_Check.Check_Re_S_lr_L2_005(state, action)
        elif self.prob_id == "S+L2_01" or self.prob_id == "S+L2_01_ver3":
            return Constraint_Check.Check_L2_01(state, action)
        elif self.prob_id == "S+L2_05" or self.prob_id == "S+L2_01_ver3":
            return Constraint_Check.Check_L2_05(state, action)
        elif self.prob_id == "S+L2_1" or self.prob_id == "S+L2_1_ver3":
            return Constraint_Check.Check_L2_1(state, action)
        elif self.prob_id == "H+L2_05" or self.prob_id == "H+L2_05_ver3":
            return Constraint_Check.Check_L2_05(state, action)
        elif self.prob_id == "H+L2_1" or self.prob_id == "H+L2_1_ver3":
            return Constraint_Check.Check_L2_1(state, action)
        elif self.prob_id == "H+M_10" or self.prob_id == "H+M_10_ver3":
            return Constraint_Check.Check_H_M10(state, action)
        elif self.prob_id == "W+M_10" or self.prob_id == "W+M_10_ver3":
            return Constraint_Check.Check_W_M10(state, action)
        elif self.prob_id == "W+M_5" or self.prob_id == "W+M_5_ver3":
            return Constraint_Check.Check_W_M5(state, action)
        elif self.prob_id == "An+O_20" or self.prob_id == "An+O_20_ver3":
            return Constraint_Check.Check_An_O20(state, action)
        elif self.prob_id == "An+O_30" or self.prob_id == "An+O_30_ver3":
            return Constraint_Check.Check_An_O30(state, action)
        elif self.prob_id == "An+L2_2" or self.prob_id == "An+L2_2_ver3":
            return Constraint_Check.Check_An_L2_2(state, action)
        elif self.prob_id == "HC+N" or self.prob_id == "Re+N" or self.prob_id == "MA_umaze+N" or self.prob_id == "MA_umaze+N" or self.prob_id == "Pu+N" or self.prob_id == "An+N":
            return Constraint_Check.Check_X_N(state, action)
        elif self.prob_id == "Net+N" or self.prob_id == "Re+N" or self.prob_id == "Pu+N" or self.prob_id == "Pandp+N" or self.prob_id == "Sl+N":
            return Constraint_Check.Check_X_N(state, action)
        elif self.prob_id == "Re+N":
            return Constraint_Check.Check_X_N(state, action) 
        elif self.prob_id == "BSS3z+S":
            return Constraint_Check.Check_BSS3z_S(state, action)
        elif self.prob_id == "BSS5z+S":
            return Constraint_Check.Check_BSS5z_S(state, action)          
        elif self.prob_id == "BSS3z+S+D40":
            return Constraint_Check.Check_BSS3z_S_D40(state, action)      
        elif self.prob_id == "BSS5z+S+D40":
            return Constraint_Check.Check_BSS5z_S_D40(state, action)
        elif self.prob_id == "BSS5z+S+D35":
            return Constraint_Check.Check_BSS5z_S_D35(state, action)
        elif self.prob_id == "BSS5z+S2":
            return Constraint_Check.Check_BSS5z_S2(state, action)
        elif self.prob_id == "BSS5z+S2+D40":
            return Constraint_Check.Check_BSS5z_S2_D40_ver2(state, action)
            
    def action_wrap_adju(self, state, action):
        if self.prob_id == "BSS3z+S+D40":
            action = 15 * action + 25
            return action    
        elif self.prob_id == "BSS5z+S+D40":
            action = 20 * action + 20
            return action    
        else:
            return action



    def action_wrap_inter(self, state, action):
        if self.prob_id == "BSS3z+S":
            action[0] = np.round(action[0])
            action[1] = np.round(action[1])
            action[2] = np.round(90 - action[0] - action[1])
            return action
        elif self.prob_id == "BSS5z+S2+D40" or self.prob_id == "BSS5z+S+D40":
            return Constraint_Proj.Projection_BSS5z_S2_INT40(state, action)
        elif self.prob_id == "BSS5z+S2+D35" or self.prob_id == "BSS5z+S+D35":
            return Constraint_Proj.Projection_BSS5z_S2_INT35(state, action)
        elif self.prob_id == "BSS3z+S+D40":
            return Constraint_Proj.Projection_BSS3z_S2_INT40(state, action)
        else:
            return action

    def get_pref(self):
        preference = np.random.dirichlet(np.ones(self.reward_num))
        preference = preference.astype(np.float32)
        return preference

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and \
            self.steps >= self.start_steps

    def augment_check(self, state, preference):
        action = self.explore(state, preference)
        for i in range(self.augement_action_sample_number):
            action = self.explore(state, preference)
            violate_check, _ = self.constraintViolation_Check(state, action)
            if(violate_check):
                return action
            else:
                self.augumented_memory.append(state, preference, action, [0, self.penalty], state, False, False)
        return action

    def augment_check_warm_up(self, state, preference):
        action = self.env.action_space.sample()
        for i in range(self.augement_action_sample_number):
            action = self.env.action_space.sample()
            violate_check, _ = self.constraintViolation_Check(state, action)
            if(violate_check):
                return action
            else:
                self.augumented_memory.append(state, preference, action, [0, self.penalty], state, False, False)
        return action
                    
    def act(self, state, preference=None):
        if preference is None:
            preference = self.get_pref()
        if self.start_steps > self.steps:
            action = self.augment_check_warm_up(state, preference)
        else:
            action = self.augment_check(state, preference)
        if self.prob_id == "NSFnetV3+S" and self.start_steps <= 3 * self.steps:
            action = Constraint_Proj.Projection_NSFnetV3(0, [-0.85]*20)
        return action

    def explore(self, state, preference):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preference = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state, preference):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preference = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, preference, actions, rewards, next_states, dones):

        curr_q1, curr_q2 = self.critic(states, actions, preference)
        
        return curr_q1, curr_q2

    def calc_target_q(self, states, preference, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states, preference)
            next_q1, next_q2 = self.critic_target(next_states, next_actions, preference)           
            
            #We choose argmin_Q (ωTQ)
            w_q1 = torch.einsum('ij,j->i',[next_q1, preference[0] ])
            w_q2 = torch.einsum('ij,j->i',[next_q2, preference[0] ])
            mask = torch.lt(w_q1,w_q2)
            mask = mask.repeat([1,self.reward_num])
            mask = torch.reshape(mask, next_q1.shape)

            minq = torch.where( mask, next_q1, next_q2)
                
            next_q = minq + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q
        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_ctrl_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        preference = self.get_pref()
        PREF_ = PREF
        while not done:
            ## Just fixed
            self.counter_time = time.perf_counter()
            action = self.act(state, preference)

            before_action = action
            violate_check, violate_diff = self.constraintViolation_Check(state, action)
            penalty = 0
            if(violate_check == False):
                penalty = self.penalty
                self.counter_time = time.perf_counter()
                action = self.constraintViolation_Proj(state, action)
                self.gp_time += time.perf_counter() - self.counter_time
            self.sample_action_time += time.perf_counter() - self.counter_time
            self.counter_time = time.perf_counter()
            after_action = self.action_wrap_adju(state, action)
            
            after_action = self.action_wrap_inter(state, after_action)
            print("a",torch.cuda.memory_summary())
            next_state, reward, done, info = self.env.step(after_action)
            print("b",torch.cuda.memory_summary())

            self.mujoco_time += time.perf_counter() - self.counter_time
            self.counter_time = time.perf_counter()
            self.steps += 1
            if(self.so_env):
                reward = np.array([reward, 0])
            if self.safe_env:
                reward[0] = reward[0] - info['cost_hazards']
            reward[1] = penalty
            episode_steps += 1
            episode_reward += reward
            if self.mujoco:
                episode_ctrl_reward += info['reward_ctrl_']
            reward[0] = reward[0] / self.reward_offset1
            reward[0] = reward[0] + self.reward_offset2
            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.max_episode_steps:
                masked_done = False
                done = True
            else:
                masked_done = done

            if self.per:
                batch = to_batch(
                    state, preference, action, reward, next_state, masked_done,
                    self.device)
     
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_q1 - target_q).item()
                self.memory.append(
                    state, preference, action, reward, next_state, masked_done, error,
                    episode_done=done)
            else:
                self.memory.append(
                        state, preference, before_action, reward, next_state, masked_done,
                        episode_done=done)
            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            if self.steps % self.eval_interval == 0 or self.steps == self.start_steps:
                for i in range(len(PREF_)):
                    self.counter_time = time.perf_counter()
                    self.evaluate(PREF_[i],i)
                    self.eval_time += time.perf_counter() - self.counter_time
            if self.steps % self.model_saved_step == 0:
                self.save_models(self.steps/self.model_saved_step)
            if self.steps % self.augment_ratio_decay_freq == 0:
                self.augment_ratio = self.augment_ratio * self.augment_ratio_decay
            state = next_state

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'episode weight: {preference}  '
              f'ctrl cost: {episode_ctrl_reward}  '
              f'reward:, {episode_reward} ')

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.learning_steps % self.q_frequency == 0 and self.learning_steps > 20000:
            co = copy.deepcopy(self.critic)
            self.Q_memory.append(co)
        
        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            augment_size = min(len(self.augumented_memory), int(self.batch_size * self.augment_ratio))
            real_size = self.batch_size - augment_size
            batch = self.memory.sample(real_size)
            augumented_batch = self.augumented_memory.sample(augment_size)
            concatenated_batch = tuple(torch.cat((batch[i], augumented_batch[i]), dim=0) for i in range(len(batch)))
            # set priority weights to 1 when we don't use PER.
            weights = 1.

        rand = random.randint(0, len(PREF)-1)
        PREF_SET = []
        # Form preference set W containing the updating preference
        preference = self.get_pref()
        preference = torch.tensor(preference ,device = self.device)
        PREF_SET.append(preference)
        for _ in range(self.set_num-1):
            p = self.get_pref()
            p = torch.tensor(p ,device = self.device)
            PREF_SET.append(p)

        self.counter_time = time.perf_counter()
        q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
            self.calc_critic_loss(concatenated_batch, weights, preference, PREF_SET)
        self.critic_loss_time += time.perf_counter() - self.counter_time
        self.counter_time = time.perf_counter()
        policy_loss, entropies = self.calc_policy_loss(concatenated_batch, weights, preference, PREF_SET)
        self.policy_loss_time += time.perf_counter() - self.counter_time
        self.counter_time = time.perf_counter()
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)
        self.policy_update_time += time.perf_counter() - self.counter_time
        self.counter_time = time.perf_counter()
        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        self.critic_update_time += time.perf_counter() - self.counter_time


        wandb.log({"total_timesteps": self.steps, "learning step" : self.learning_steps, "policy_loss": policy_loss, "q1_loss": q1_loss, "q2_loss": q2_loss, \
        "critic_update_time": self.critic_update_time, "policy_update_time": self.policy_update_time, "critic_loss_time": self.critic_loss_time,\
        "policy_loss_time": self.policy_loss_time, "mujoco_time": self.mujoco_time, "gp_time": self.gp_time, \
        "sample_action_time": self.sample_action_time, "eval_time": self.eval_time,
        })
        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            wandb.log({"total_timesteps": self.steps, "learning step" : self.learning_steps,"entropy loss": entropy_loss})
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())
        
    def calc_critic_loss(self, batch, weights, preference, PREF):
        
        states, _, actions, rewards, next_states, dones = batch
        
        D_pref = preference.repeat(self.batch_size,1)

        curr_q1, curr_q2 = self.calc_current_q(states, D_pref, actions, rewards, next_states, dones)
        
        target_q = self.calc_target_q(states, D_pref, actions, rewards, next_states, dones)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()
      
        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean(torch.tensordot((curr_q1 - target_q).pow(2), preference,dims=1) * weights)
        q2_loss = torch.mean(torch.tensordot((curr_q2 - target_q).pow(2), preference,dims=1) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights, preference, PREF):
        
        observations, _, actions, rewards, next_observations, dones = batch
        
        sample_number = self.action_number * self.action_sample_number
        preference_batch = preference.repeat(self.batch_size, 1) 
        preference_batch_sample = preference.repeat(sample_number * self.batch_size, 1)
        actions_pis = torch.tensor([],device=self.device)
        actions_pis_prime = torch.tensor([],device=self.device)
        cad_count = torch.tensor([],device=self.device)
        log_probs = torch.empty(0,device=self.device)
        total_count = 0
        tensor_true = torch.tensor(1,device=self.device)
        accept_count = 0
        
        # Action by the current actor for the sampled state
        ent_coef_loss = None
        tiled_observations = observations.unsqueeze(1).repeat(1, sample_number, 1).view(-1, observations.size(1))
        losses = []
        tiled_actions_pis, tiled_entropys, _ = self.policy.sample(tiled_observations, preference_batch_sample)
        tiled_entropys = tiled_entropys.reshape(-1)
        for a, c in enumerate([ self.critic]+self.Q_memory.sample() ): # Use critic from Q Replay Buffer
            for b, i in enumerate(PREF): #Get Q from preference set W
                p_batch = torch.tensor(i, device = self.device).repeat(sample_number * self.batch_size, 1)
                q1, q2 = c(tiled_observations, tiled_actions_pis, p_batch)
                q1 = torch.tensordot(q1, preference, dims = 1)
                q2 = torch.tensordot(q2, preference, dims = 1)
                q = torch.min(q1, q2)
                l = - q - self.alpha * tiled_entropys
                losses.append(l)
        
        losses = torch.stack(losses, dim = 1)
    
        policy_loss, idx =  torch.min(losses, 1)
        policy_loss = torch.mean(policy_loss)

        min_indices = torch.argmin(losses, dim=1).detach()
        policy_loss1 = losses[torch.arange(len(min_indices)), min_indices]
        policy_loss1 = torch.mean(policy_loss1)

        sampled_action, e, _ = self.policy.sample(observations, preference_batch)
        #return policy_loss, e, accept_count / total_count

        return policy_loss, e

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss

    def evaluate(self, preference, ind):
        episodes = self.eval_episode
        returns = np.empty((episodes,self.reward_num))
        preference = np.array(preference)
        count_reward2 = 0
        count_reward3 = 0
        total_count = 0
        for i in range(episodes):
            if self.prob_id == "MA_umaze+N" or self.prob_id == "MA_umaze+L2_08" or self.prob_id == "MA_umaze+L2_08_ver3":
                state = self.env.reset()
                state = self.env.reset_to_location([3,1])
            elif self.prob_id == "MA_medium+N" or self.prob_id == "MA_medium+L2_08" or self.prob_id == "MA_medium+L2_08_ver3":
                state = self.env.reset()
                state = self.env.reset_to_location([6,1])
            else:
                state = self.env.reset()
            episode_reward = np.zeros(self.reward_num)
            episode_ctrl_reward = 0
            done = False
            count = 0
            reward2_total = 0
            while not done:
                sample_count = 0
                count += 1
                reward2 = 0
                reward3 = 0
                while(sample_count<self.eval_sample_number):
                    sample_count += 1 
                    action = self.explore(state, preference)
                    violate_check, _ = self.constraintViolation_Check(state, action)
                    if(violate_check):
                        reward2_total += 1
                    else:
                        reward2 += 1
                action = self.exploit(state, preference)
                after_action = self.constraintViolation_Proj(state, action)
                after_action = self.action_wrap_inter(state, after_action)
                next_state, reward, done, info = self.env.step(after_action)
                if(self.so_env):
                    reward = np.array([reward, 0])
                if self.safe_env:
                    reward[0] = reward[0] - info['cost_hazards']
                if self.goal_env and info['is_success']:
                    reward3 = 1
                if self.mujoco:
                    episode_ctrl_reward += info['reward_ctrl_']
                reward[1] = -reward2
                episode_reward += reward
                state = next_state
            count_reward2 += reward2_total / count
            returns[i] = episode_reward
            total_count += count
            print(episode_reward)
        mean_return = np.mean(returns, axis=0)
        
        batch = self.memory.sample(self.batch_size) 
        p = torch.tensor(preference ,device = self.device, dtype=torch.float32)
        with torch.no_grad():
            q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
                            self.calc_critic_loss(batch, 1, p, 0)
        wandb.log({"total_timesteps": self.steps, "learning step" : self.learning_steps, "eval/" + str(preference) + "reward0": mean_return[0], "eval/" + str(preference) + "reward1": mean_return[1], "eval/" + str(preference) + "reward2": count_reward2 / self.eval_episode, "eval/" + str(preference) + "avg_steps": total_count/episodes})
        if self.goal_env:
            wandb.log({"total_timesteps": self.steps, "learning step" : self.learning_steps, "eval/" + str(preference) + "reward3": count_reward3 / self.eval_episode})
        if self.mujoco:
            wandb.log({"total_timesteps": self.steps, "learning step" : self.learning_steps, "eval/" + str(preference) + "ctrl_cost": episode_ctrl_reward})
        path = os.path.join(self.log_dir, 'summary')
        tot_path = os.path.join(path, f'{ind}total_log.npy')
        reward_path = os.path.join(path, f'{ind}reward_log.npy')
        self.tot_t[ind].append( np.dot(preference, mean_return) )
        self.reward_v[ind].append(mean_return)

        np.save(tot_path, np.array(self.tot_t[ind]) )
        np.save(reward_path, np.array(self.reward_v[ind]) )

        print('-' * 60)
        print(f'preference ', preference,
              f'Num steps: {self.steps:<5}  '
              f'reward:', mean_return,
              f'avg steps:', total_count/episodes)
        print('-' * 60)

    def save_models(self, num):
        self.policy.save(os.path.join(self.model_dir, 'policy_'+str(num)+'.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic_'+str(num)+'.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, 'critic_target.pth'))

    def __del__(self):
        self.env.close()
