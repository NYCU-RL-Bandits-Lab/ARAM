import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register
from gym.utils import seeding


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class HopperEnv_M_10_vel3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="hopper.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self.np_random, seed = seeding.np_random(0)
        self._max_episode_steps = 1000
        self.max_episode_steps = 1000
        self._goal_vel = 3
        self.steps = 0
        self.reward_num = 2
        mujoco_env.MujocoEnv.__init__(self, xml_file, 4)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
        
    def control_cost_(self, action):
        control_cost = np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        if(self.steps >= self._max_episode_steps):
            done = True
        else:
            done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        state = self._get_obs()
        if(np.maximum((state[8:]*action),0).sum()<=10+1e-6):
            x_position_before = self.sim.data.qpos[0]
            self.do_simulation(action, self.frame_skip)
            x_position_after = self.sim.data.qpos[0]
            x_velocity = (x_position_after - x_position_before) / self.dt

            ctrl_cost = self.control_cost(action)
            ctrl_cost_ = self.control_cost_(action)

            forward_reward = -np.abs(x_velocity - self._goal_vel) + 1 # make it happy, not suicidal
            #forward_reward = self._forward_reward_weight * x_velocity
            healthy_reward = self.healthy_reward

            rewards = forward_reward + healthy_reward
            costs = ctrl_cost
            reward = rewards - costs
            penalty = 0
        else:
            x_position_before = self.sim.data.qpos[0]
            x_position_after = self.sim.data.qpos[0]
            x_velocity = (x_position_after - x_position_before) / self.dt
    
            ctrl_cost = self.control_cost(action)
            ctrl_cost_ = self.control_cost_(action)

            forward_reward = -np.abs(x_velocity - self._goal_vel) + 1 # make it happy, not suicidal
            #forward_reward = self._forward_reward_weight * x_velocity
            healthy_reward = self.healthy_reward
    
            rewards = forward_reward + healthy_reward
            costs = ctrl_cost
            reward = rewards - costs
            penalty = 1
        observation = self._get_obs()
        self.steps += 1

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "penalty": penalty,
            "reward_ctrl_": -ctrl_cost_,
        }
        return observation, np.array([reward, -penalty]), self.done, info

    def reset_model(self):
        self.steps = 0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

register(id='MO_hopper_M_10_goal_vel3-v0', entry_point='environments.hopper_v3_M_10_goal_vel3:HopperEnv_M_10_vel3')
