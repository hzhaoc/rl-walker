from rl import Env


env = Env.make(id="TestHumannoidEnv", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(10000):
    action = env.action_space.sample()
    #print('action dimension', action.shape)
    observation, reward, terminated, truncated, info = env.step(action)
    #print('obs dimension', observation.shape)

    if terminated or truncated:
        observation, info = env.reset()
env.close()