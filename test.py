from rl import Env
from rl.agent import *
from rl.agent.agentDDPG import AgentDDPG
from rl.agent.agentTD3 import AgentTD3
from rl.util.utils import Queue
from params import params, EnvNames


envi = 121
# agentClass = AgentDDPG
agentClass = AgentTD3
version = 5
startEpisode = 0

torch.autograd.set_detect_anomaly(mode=False, check_nan=True)
env = Env.make(name=EnvNames[envi], render_mode="human")
agent = agentClass(env=env, 
                  critic_loss_weight_regularization_l2=params.agent.critic_loss_weight_regularization_l2,
                  critic_gradient_clip=params.agent.critic_gradient_clip,
                  actor_noise_switch=params.agent.actor_noise_switch,  # exploration switch
                  actor_noise_sigma=params.agent.actor_noise_sigma,  # volatility of noise in actor
                  actor_noise_theta=params.agent.actor_noise_theta, # temporal scaler in actor noise
                  policy_noise=params.agent.policy_noise, # noise for training only
                  noise_clip=params.agent.noise_clip, # noise for training only
                  tau=params.agent.tau, # target network weight update rate
                  gamma=params.agent.gamma, # net present value discount rate. the closer the 1, the longer term of reward memory
                  critic_lr=params.agent.critic_lr, # critic network weight learning rate
                  actor_lr=params.agent.actor_lr, # actor network weight learning rate
                  exp_sample_size=params.agent.exp_sample_size,
                  bufsize=params.agent.bufsize, # experiance buffer size for sampling
                  optim_momentum=params.agent.optim_momentum,  # stochastic gradient descent momentum
                  actor_last_layer_weight_init=params.agent.actor_last_layer_weight_init, # actor network last layer initial uniform distribution radius around 0
                  critic_last_layer_weight_init=params.agent.critic_last_layer_weight_init, # critic network last layer initial uniform distribution range
                  critic_bn_eps=params.agent.critic_bn_eps,  # critic network batch norm epsilon for scaling stability
                  critic_bn_momentum=params.agent.critic_bn_momentum,  # critic network batch norm running mean and standard deviation momentum 
                  #critic_layer_depth=3,  # critic network number of layers. action will be entered in the last second layer. MIN=3
                  #actor_layer_depth=3,  # actor network layer depth. MIN=3
                  actor_loss_weight_regularization_l2=params.agent.actor_loss_weight_regularization_l2,
                  actor_gradient_clip=params.agent.actor_gradient_clip,
                  update_delay=params.agent.update_delay,  # how many steps it take to update online actor, target actor, target critics once.
        )

if startEpisode > 0:
    agent.load(f"models/{EnvNames[envi]}.{agent.__class__.__name__}.v{version}.e{startEpisode}")


# training
fileRewards = open(rf"play/{EnvNames[envi]}.txt", "w+", 1)
for j in range(startEpisode+1, startEpisode+1+params.train.episodes):
    r = 0.0
    state, _ = env.reset()
    agent.reset()

    for i in range(params.train.steps):
        action = agent.act(state)
        # print(action)
        next_state, reward, terminated, truncated, info = env.step(action)

        r += reward
        #print(f"epoch {j} step {i}: rFwd {round(info['rFwd'], 2)}, rCtrl {round(info['rCtrl'], 2)}, rAlive {round(info['rAlive'], 2)}, rHeight {round(info['rHeight'], 2)}, rHead: {round(0.0, 2)}")
        agent.buf.push(state, action, reward, next_state, terminated)
        agent.update()

        if terminated:
            break
        state = next_state

    line = "{},{}\n".format(j, round(r, 2))
    fileRewards.write(line)
    print(line, end="")
    if j % 100 == 0:
        modelDir = f"models/{EnvNames[envi]}.{agent.__class__.__name__}.v{version}.e{j}"
        agent.save(modelDir)
    
fileRewards.close()
env.close()