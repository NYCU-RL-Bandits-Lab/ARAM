import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}

class AntEnv_L2_2_goal_forward_ccw05(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self._max_episode_steps = 1000
        self.max_episode_steps = 1000
        self.reward_num = 2
        self.steps = 0
        self._goal_vel = 3.0
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def control_cost_(self, action):
        control_cost = np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        if self.steps > self._max_episode_steps:
            done = True
        return done

    def Check_An_L2_2(self, state, action):
        a1=action[0]
        a2=action[1]
        a3=action[2]
        a4=action[3]
        a5=action[4]
        a6=action[5]
        a7=action[6]
        a8=action[7]
        test = (a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7 + a8 * a8 <= 2+1e-6)
        dif = a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7 + a8 * a8 - 2
        return test

    def step(self, action):
        self.steps += 1
        xy_position_before = self.get_body_com("torso")[:2].copy()
        observation = self._get_obs()
        if(self.Check_An_L2_2(observation, action)):
            penalty = 0
        else:
            penalty = 1
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        comvel =  np.linalg.norm(xy_velocity, ord=2)
        x_velocity, y_velocity = xy_velocity
        forward_reward = -np.abs(x_velocity - self._goal_vel)+1  # make it happy, not suicidal

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        ctrl_cost_ = self.control_cost_(action)


        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            'reward_fwd': forward_reward,
            'reward_ctrl': -ctrl_cost,
            "reward_ctrl_": -ctrl_cost_,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),
            'penalty': penalty,
            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'xy_velocity': comvel,
            'forward_reward': forward_reward,
        }

        return observation, np.array([reward, -penalty]), done, info
        #return observation, reward, done, info

    def _get_obs(self):

        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        self.steps = 0

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

register(id='MO_ant_L2_2_goal_forward_ccw05-v0', entry_point='environments.ant_v3_L2_2_goal_forward_ccw05:AntEnv_L2_2_goal_forward_ccw05')
