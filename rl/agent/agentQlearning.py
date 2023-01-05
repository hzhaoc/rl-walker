# TODO: remove or redo the file


from rl.agent import Agent, Actor, Critic
from rl.updatable.actionUpdatable import ActionUpdatableFinite
from rl.state import State
from rl.action import Action
from rl.util import *
from rl.env import EnvFeedback


class AgentQlearning(Agent):
    """
    Q-learning policy / greedy expected SARSA
    
    **finite** aciton space
    **finite** state space

    see note: Discrete Action - TD, exp model#SARSA
    see note: Discrete Action - TD, exp model#Q-learning
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.actor = _ActorQlearning(self, **kwargs)
        self.critic = _CriticQlearning(self, **kwargs)

    def act(self, S: State) -> Action:
        return self.actor.act(S)

    def update(self, feedback: EnvFeedback) -> None:
        self.critic.update(feedback)  # value function should be updated first to provde latest estimation adjustment to actor
        self.actor.update(feedback)


class _ActorQlearning(Actor, ActionUpdatableFinite):
    def __init__(self, agent: Agent, 
                       action_space: dict = {}) -> None:
        Actor.__init__(self, agent = agent)
        ActionUpdatableFinite.__init__(self, action_space = action_space)

    @override
    def act(self, s: State) -> Action:
        """at current state, take the action that gives maximum value"""
        self.validate()
        s = self._getState(s)
        self._a = Action(action=max(self._agent.critic.Q[s], key = self._agent.critic.Q[s].get))
        return self._a

    @override
    def update(self) -> None:
        pass


class _CriticQlearning(Critic, ActionUpdatableFinite):
    def __init__(self, agent: Agent, 
                       rate: float = None,
                       action_space: dict = {},
                       discount: float = 0.95
                ) -> None:
        Critic.__init__(self, agent = agent)
        ActionUpdatableFinite.__init__(self, rate = rate, action_space = action_space, discount = discount)
        self.Q = defaultdict(defaultdict(float))  # outer key is state, inner key is action, value is long term reward / rstate-action value

    @override
    def update(self, s_t1: State, r_t1: float) -> None:
        """
        action value update rule uses a TD variant:
        $Q(S_t,A_t) = Q(S_t,A_t) + \alpha(R_{t+1} + \gamma max_{\alpha'}Q(S_{t+1},\alpha') - Q(S_t,A_t))$

        different from SARSA, it is greedy w.r.t uses next action that maximzies value of next state to estimate future return
        """
        try:
            s_t0 = self._getState(self._agent._feedback.state)
            a_t0 = self._getAction(self._agent._a)
            q_t0 = self._agent.critic.Q[s_t0][a_t0]

            s_t1 = self._getState(s_t1)
            #a_t1 = self._getAction(a_t1) # this is SARSA
            a_t1 = Action(action=max(self._agent.critic.Q[s_t1], key = self._agent.critic.Q[s_t1].get)).val  # argmax needs to find max from given finite action space, needs redo
            q_t1 = self._agent.critic.Q[s_t1][a_t1]
            
            r_t1 = r_t1
            
            gamma = self._discount
            alpha = self._rate

            q_t0 = q_t0 + alpha* (r_t1 + gamma * q_t1 - q_t0)
            self._agent.critic.Q[s_t0][q_t0] = q_t0
        except:
            print("cannot update critic")
    