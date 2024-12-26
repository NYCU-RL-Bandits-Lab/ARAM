import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register
from gym.utils import seeding

class ReacherEnv_L2_005(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.max_episode_steps = 50
        self._max_episode_steps = 50
        self.reward_num = 2
        self.reward_space = 2
        self.steps = 0
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def step(self, a):  
        self.steps += 1          
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        if(np.square(a).sum() <= 0.05+1e-6):
            reward = reward_dist + reward_ctrl
            self.do_simulation(a, self.frame_skip)
            ob = self._get_obs()
            done = False
            penalty = 0
        else:
            ob = self._get_obs()
            reward = 0
            penalty = 1
        if(self.steps >= self._max_episode_steps):
            done = True
        else:
            done = False
        return ob, np.array([reward, -penalty]), done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, reward_ctrl_=reward_ctrl, penalty = -penalty)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0


    def reset_model(self):
        self.steps = 0
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

register(id='MO_reacher_L2_005-v0', entry_point='environments.reacher_v3_L2_005:ReacherEnv_L2_005')