import sys
import gym

from elegantrl.train.run import train_and_evaluate, train_and_evaluate_mp
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentDDPG import AgentDDPG, AgentDDPGHterm

def demo_ddpg_h_term(gpu_id, drl_id, env_id):  
    env_name = ['Hopper-v2',
                'Swimmer-v3',
                'Ant-v3',
                'Humanoid-v3', 
                'HalfCheetah-v3',
                'Walker2d-v3',
                ][env_id]
    agent_class = [AgentDDPG, AgentDDPGHterm][drl_id]

    if env_name == 'Humanoid-v3':
        from rl.envs.CustomGymEnv import HumanoidEnv
        env_func = HumanoidEnv
        env_args = {
            'env_num': 1,
            'env_name': 'Humanoid-v3',
            'max_step': 1000,
            'state_dim': 376,
            'action_dim': 17,
            'if_discrete': False,
            'target_return': 3000.,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.eval_times = 2 ** 2
        args.reward_scale = 2 ** -2
        args.max_memo = 2 ** 21
        args.learning_rate = 2 ** -14
        args.lambda_a_log_std = 2 ** -6

        args.target_step = args.max_step
        args.worker_num = 4

        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim // 2
        args.num_layer = 3
        args.repeat_times = 2 ** 0
        args.gamma = 0.96
        args.if_act_target = False
        import numpy as np
        args.target_entropy = np.log(env_args['action_dim'])

        args.if_allow_break = False
        args.break_step = int(4e6)
    else:
        raise ValueError('env_name:', env_name)
 
    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU
    AGENT_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    demo_ddpg_h_term(GPU_ID, AGENT_ID, ENV_ID)