# TODO: compare our TD3 to benchmar in bipedalwalker, and fine tine it
# test:
# - optimizer                   
# - noise level                 
# - noise type                  
# - batch norm                  little effect, nice to have
# - grad clip                   little effect
# - done reward                 use close to 0 done reward
# - network size                use larger network
# - epoch or step update        use step update
# - batch size                  little effect, nice to set to 1

#  TODO: try different agents:
#       - A3C: did well on leg balancing
#       - SAC (I may have already implemented in in DDPG & TD3. may be just need more epochs to train), 
#       - PPO
#       - Adv. POET 

# TODO：exploration
        # - add intrinsic reward: curiosity driven 
        #    - maximize knolwedge about environment (forward dynamics. error between f(s_t, a_t)=a'_{t+1} and a_{t+1})
        #    - frequency based
        #    - maximize information gain. entroy based
        #    - memory-based. encourage state more distant to current visited state to be explored
        #    - drawbakcs
        #      - knolwedge fading. when states cease to be novel no exploration anymore
        #      - in addition to above, function approximator slow to catch up
        # noise based
        #   - can we make noise bigger when reward is high, and vice versa?
        # 
        
# TODO: when updating online actor, use online or offline critic?
# TODO: a Param class to contain all hyper parameters
# TODO: PPO and other potentially intrinsically better agents
# TODO: fix TD3 action output action with nan