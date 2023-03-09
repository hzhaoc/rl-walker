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
from gymnasium.envs.mujoco.humanoidstandup_v4 import HumanoidStandupEnv
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker, BipedalWalkerHardcore
from rl.util import *
import numpy as np
import math


class TestHumannoidEnv(HumanoidEnv):
    """
    see humanoid.py#HumandoidEnv for detailed descriptions about the environment
    this class customizes humanoid.py#HumandoidEnv
    """

    def __init__(
        self, 
        forward_reward_weight=2.0,
        ctrl_cost_weight=0.15, 
        healthy_reward=5, 
        terminate_when_unhealthy=True, 
        healthy_z_range=(0.6, 2.0), 
        reset_noise_scale=0.1, 
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
        #   - add contact_cost in cost function?
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        observation = self._get_obs()
        reward_stable_height = -(observation[0] - 1.31)**2 * 10
        #reward_head_straight = -math.acos(observation[1])**2 * 10  # convert quaternion to eular angle
        reward = forward_reward + healthy_reward - ctrl_cost + reward_stable_height# + reward_head_straight
        terminated = self.terminated

        info = {
            "rFwd": forward_reward,
            "rCtrl": -ctrl_cost,
            "rAlive": healthy_reward,
            "rHeight": reward_stable_height,
            #"rHead": reward_head_straight,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

    
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


class TestHumanoidStandupEnv(HumanoidStandupEnv):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(HumanoidStandupEnv)
    def step(self, action):
        return super().step(action)


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

HULL_INIT_POS_Y = TERRAIN_HEIGHT + 2 * LEG_H


class TestBipedalWalker(BipedalWalker):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    @override(BipedalWalker)
    def step(self, action):
        assert self.hull is not None

        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
            )
            self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
            )
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        shaping = (
            130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 50.0 * abs(
            state[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        terminated = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True

        if self.render_mode == "human":
            self.render()

        reward += -max(abs(pos.y - HULL_INIT_POS_Y) - 1.0, 0.0)**2 * 50  # head height maintains same level
        return np.array(state, dtype=np.float32), reward, terminated, False, {}


class TestBipedalWalkerHardcore(BipedalWalkerHardcore):
    def __init__(
        self, 
        **kwargs):
        super().__init__(
            **kwargs)

    def step(self, action):
        return super().step(action)