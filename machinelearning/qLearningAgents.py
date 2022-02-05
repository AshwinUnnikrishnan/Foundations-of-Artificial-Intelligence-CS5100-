import models
import nn
from backend import ReplayMemory
import copy

import numpy as np

import layout
from game import *
from learningAgents import ReinforcementAgent

import random,util

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qVals = {}
        self.eval = False

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        stateQVals = self.qVals.setdefault(state,util.Counter())
        return stateQVals[action]

    def getPolicyAndValue(self, state):
        bestAct, bestVal = [None], float("-inf")
        actions = self.getLegalActions(state)
        if len(actions) == 0: return (None, 0.0)
        for act in actions:
            val = self.getQValue(state,act)
            if val > bestVal:
                bestAct, bestVal = [act], val
            elif val == bestVal:
                bestAct.append(act)

        return random.choice(bestAct), bestVal

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        return self.getPolicyAndValue(state)[1]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        return self.getPolicyAndValue(state)[0]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        oldQVal = self.getQValue(state, action)
        newQVal = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qVals[state][action] += self.alpha * (newQVal-oldQVal)
        return

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class PacmanDeepQAgent(PacmanQAgent):
    def __init__(self, layout_input="smallGrid", target_update_rate=300, doubleQ=True, **args):
        PacmanQAgent.__init__(self, **args)
        self.model = None
        self.target_model = None
        self.target_update_rate = target_update_rate
        self.update_amount = 0
        self.epsilon_explore = 1.0
        self.epsilon0 = 0.05
        self.epsilon = self.epsilon0
        self.discount = 0.9
        self.update_frequency = 3
        self.counts = None
        self.replay_memory = ReplayMemory(50000)
        self.min_transitions_before_training = 10000
        self.td_error_clipping = 1

        # Initialize Q networks:
        if isinstance(layout_input, str):
            layout_instantiated = layout.getLayout(layout_input)
        else:
            layout_instantiated = layout_input
        self.state_dim = self.get_state_dim(layout_instantiated)
        self.initialize_q_networks(self.state_dim)

        self.doubleQ = doubleQ
        if self.doubleQ:
            self.target_update_rate = -1

    def get_state_dim(self, layout):
        pac_ft_size = 2
        ghost_ft_size = 2 * layout.getNumGhosts()
        food_capsule_ft_size = layout.width * layout.height
        return pac_ft_size + ghost_ft_size + food_capsule_ft_size

    def get_features(self, state):
        pacman_state = np.array(state.getPacmanPosition())
        ghost_state = np.array(state.getGhostPositions())
        capsules = state.getCapsules()
        food_locations = np.array(state.getFood().data).astype(np.float32)
        for x, y in capsules:
            food_locations[x][y] = 2
        return np.concatenate((pacman_state, ghost_state.flatten(), food_locations.flatten()))

    def initialize_q_networks(self, state_dim, action_dim=5):
        import models
        self.model = models.DeepQModel(state_dim, action_dim)
        self.target_model = models.DeepQModel(state_dim, action_dim)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) as predicted by self.model
        """
        feats = self.get_features(state)
        legalActions = self.getLegalActions(state)
        action_index = legalActions.index(action)
        state = nn.Constant(np.array([feats]).astype("float64"))
        return self.model.run(state).data[0][action_index]


    def shape_reward(self, reward):
        if reward > 100:
            reward = 10
        elif reward > 0 and reward < 10:
            reward = 2
        elif reward == -1:
            reward = 0
        elif reward < -100:
            reward = -10
        return reward


    def compute_q_targets(self, minibatch, network = None, target_network=None, doubleQ=False):
        """Prepare minibatches
        Args:
            minibatch (List[Transition]): Minibatch of `Transition`
        Returns:
            Q_target: a (batch_size x num_actions) numpy array
        """
        if network is None:
            network = self.model
        if target_network is None:
            target_network = self.target_model
        states_np = np.vstack([x.state for x in minibatch])
        states = nn.Constant(states_np)
        states_np = states_np.astype('int')
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        next_states = np.vstack([x.next_state for x in minibatch])
        next_states = nn.Constant(next_states)
        dones = np.array([x.done for x in minibatch])

        "*** YOUR CODE STARTS HERE ***"
        model1 = models.DeepQModel(len(states), len(actions))
        model2 = None
        # These are placeholders. You should assign correct values for these two variables according to the specs.
        # Note that this doesn't mean you should only use two lines of code. For your reference, the staff solution uses 10 lines.
        Q_predict =  model1.run(states) # a numpy array of Q network's prediction of size (batch_size x num_actions) on states.
        Q_target = None # a numpy array of Q target of size (batch_size x num_actions)

        "*** YOUR CODE ENDS HERE ***"

        if self.td_error_clipping is not None:
            Q_target = Q_predict + np.clip(
                     Q_target - Q_predict, -self.td_error_clipping, self.td_error_clipping)

        return Q_target

    def update(self, state, action, nextState, reward):
        legalActions = self.getLegalActions(state)
        action_index = legalActions.index(action)
        done = nextState.isLose() or nextState.isWin()
        reward = self.shape_reward(reward)

        if self.counts is None:
            x, y = np.array(state.getFood().data).shape
            self.counts = np.ones((x, y))

        state = self.get_features(state)
        nextState = self.get_features(nextState)
        self.counts[int(state[0])][int(state[1])] += 1

        transition = (state, action_index, reward, nextState, done)
        self.replay_memory.push(*transition)

        if len(self.replay_memory) < self.min_transitions_before_training:
            self.epsilon = self.epsilon_explore
        else:
            self.epsilon = max(self.epsilon0 * (1 - self.update_amount / 20000), 0)

        if len(self.replay_memory) > self.min_transitions_before_training and self.update_amount % self.update_frequency == 0:
            minibatch = self.replay_memory.pop(self.model.batch_size)
            states = np.vstack([x.state for x in minibatch])
            states = nn.Constant(states.astype("float64"))
            Q_target1 = self.compute_q_targets(minibatch, self.model, self.target_model, doubleQ=self.doubleQ)
            Q_target1 = nn.Constant(Q_target1.astype("float64"))

            if self.doubleQ:
                Q_target2 = self.compute_q_targets(minibatch, self.target_model, self.model, doubleQ=self.doubleQ)
                Q_target2 = nn.Constant(Q_target2.astype("float64"))
            
            self.model.gradient_update(states, Q_target1)
            if self.doubleQ:
                self.target_model.gradient_update(states, Q_target2)

        if self.target_update_rate > 0 and self.update_amount % self.target_update_rate == 0:
            self.target_model.set_weights(copy.deepcopy(self.model.parameters))

        self.update_amount += 1

    def final(self, state):
        """Called at the end of each game."""
        PacmanQAgent.final(self, state)