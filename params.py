d = \
{
    "train": {
        "episodes": 5000,
        "steps": 2000,
        "batchUpdate": False, # if true, update by epoch
    },

    "agent": { # comment is benchmark proved to have worked
        "critic_loss_weight_regularization_l2": 0.0,
        "critic_gradient_clip": 1e16, # prevent overfitting
        "noise_type": 'normal', # "ou": ou noise, "normal": normal noise, "empty": empty noise
        "actor_noise_sigma" :0.1,
        "actor_noise_theta": 0.05,
        "actor_output_size": None,  # if set at 0 or None, by default it will be same as action space
        "actor_do_batch_norm": True, 
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "tau": 0.005, # stablize traning
        "gamma": 0.99,
        "critic_lr": 1e-3,
        "actor_lr": 1e-3,
        "exp_sample_size": 256,
        "bufsize": 500000,
        "optim_momentum": 1e-1,
        "actor_last_layer_weight_init": 3e-3,
        "critic_last_layer_weight_init": 3e-4,
        "critic_bn_eps": 1e-4,
        "critic_bn_momentum": 1e-2,
        "actor_loss_weight_regularization_l2": 0.0,
        "actor_gradient_clip": 1e16, # prevent overfitting
        "update_delay":2,
        "optim_type": "adam", # default sgd
    }
}


#################################################################################  A: working. B: barely working. C: not working
#                                               task            DDPG        TD3
EnvNames = {
    110: "TestReacherEnv",                    # locomotion      C           C
    111: "TestContinuous_MountainCarEnv",     # locomotion      B           A
    112: "TestHumannoidEnv",                  # locomotion                  C
    120: "TestWalker2dEnv",                   # locomotion 
    121: "TestBipedalWalker",                 # locomotion      C           C
    130: "TestHalfCheetahEnv",                # locomotion                  A
    140: "TestAntEnv",                        # locomotion      
    150: "TestHopperEnv",                     # locomotion     
    160: "TestSwimmerEnv",                    # locomotion      

    220: "TestInvertedDoublePendulumEnv",     # stablization    
    230: "TestPendulumEnv",                   # stablization    A-          A
}
#################################################################################


class Params:
# TODO: implement __repr__
# TODO: implement __get__, __set__ or __getter__, __setter__?
    def __init__(self, kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

params = {k: Params(v) for k, v in d.items()}
params = Params(params)