from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.classic_control import *
from gymnasium.envs.box2d import *
from gymnasium.envs.mujoco.inverted_double_pendulum_v4 import InvertedDoublePendulumEnv
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv
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
        #   - penalize energy / force differences spent on two legs (together with total force penalization to try to result in efficient walks)
        #   - penalize two-feet-in-the-air situation (it's walker not runner. hmm does not seem like a good reward component)
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


class TestCartPoleEnv(CartPoleEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(CartPoleEnv)
    def step(self, action):
        return super().step(action)


class TestContinuous_MountainCarEnv(Continuous_MountainCarEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(Continuous_MountainCarEnv)
    def step(self, action):
        return super().step(action)


class TestCarRacing(CarRacing):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(CarRacing)
    def step(self, action):
        return super().step(action)


class TestInvertedDoublePendulumEnv(InvertedDoublePendulumEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(InvertedDoublePendulumEnv)
    def step(self, action):
        return super().step(action)


class TestWalker2dEnv(Walker2dEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(Walker2dEnv)
    def step(self, action):
        return super().step(action)


class TestSwimmerEnv(SwimmerEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(SwimmerEnv)
    def step(self, action):
        return super().step(action)


class TestHalfCheetahEnv(HalfCheetahEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(HalfCheetahEnv)
    def step(self, action):
        return super().step(action)


class TestAntEnv(AntEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(AntEnv)
    def step(self, action):
        return super().step(action)


class TestHopperEnv(HopperEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(HopperEnv)
    def step(self, action):
        return super().step(action)


class TestReacherEnv(ReacherEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(ReacherEnv)
    def step(self, action):
        return super().step(action)