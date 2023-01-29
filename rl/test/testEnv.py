from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from rl.util import *


class TestHumannoidEnv(HumanoidEnv):
    """
    see humanoid.py#HumandoidEnv for detailed descriptions about the environment
    this class customizes humanoid.py#HumandoidEnv
    """

    def __init__(
        self, 
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1, 
        healthy_reward=5, 
        terminate_when_unhealthy=True, 
        healthy_z_range=(1.0, 2.0), 
        reset_noise_scale=0.01, 
        exclude_current_positions_from_observation=True, 
        **kwargs):
        super().__init__(
            forward_reward_weight, 
            ctrl_cost_weight, 
            healthy_reward, 
            terminate_when_unhealthy, 
            healthy_z_range, 
            reset_noise_scale, 
            exclude_current_positions_from_observation, 
            **kwargs)

    @override(HumanoidEnv)
    def step(self, action):
        # TODO: customize reward function as one sees fit
        # things to consider:
        #   - penalizing energy / force differences spent on two legs (together with total force penalization to try to result in efficient walks)
        #   - normalize each reward component and assign wegiht to each component. this way, we can specify a reward function to more intentionally guide agent learning 
        #   - add contact_cost in cost function?
        return super().step(action)


class TestPendulumEnv(PendulumEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(PendulumEnv)
    def step(self, action):
        return super().step(action)