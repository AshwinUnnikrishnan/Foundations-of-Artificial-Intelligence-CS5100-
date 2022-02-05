# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        i = 0
        while i < self.iterations:
            i += 1
            newQ = self.values.copy()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    newQ[state] = self.computeQValueFromValues(state, self.computeActionFromValues(state))
            self.values = newQ


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        possibleStates = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0
        for nextS, prob in possibleStates:
            QValue += prob * (self.mdp.getReward(state, action, nextS) + self.discount * self.getValue(nextS))
        return QValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        actionQValues = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            actionQValues[action] = self.computeQValueFromValues(state, action)
        return actionQValues.argMax()
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        i, n = 0, len(states)
        for iteration in range(self.iterations):
            if not self.mdp.isTerminal(states[i]):
                self.values[states[i]] = self.computeQValueFromValues(states[i], self.computeActionFromValues(states[i]))
            i = (i + 1) % n

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        priorityQ = util.PriorityQueue()
        prev, states = {}, self.mdp.getStates()
        for state in states:
            if not self.mdp.isTerminal(state):
                for act in self.mdp.getPossibleActions(state):
                    for next, prob in self.mdp.getTransitionStatesAndProbs(state, act):
                        if next in prev:
                            prev[next].add(state)
                        else:
                            prev[next] = {state}

        for state in states:
            if not self.mdp.isTerminal(state):
                val = []
                for action in self.mdp.getPossibleActions(state):
                    val.append(self.computeQValueFromValues(state, action))
                priorityQ.update(state, -abs(max(val) - self.values[state]))

        for i in range(self.iterations):
            if priorityQ.isEmpty():
                break
            temp = priorityQ.pop()
            if not self.mdp.isTerminal(temp):
                val = []
                for action in self.mdp.getPossibleActions(temp):
                    val.append(self.computeQValueFromValues(temp, action))
                self.values[temp] = max(val)

            for p in prev[temp]:
                if not self.mdp.isTerminal(p):
                    val = []
                    for action in self.mdp.getPossibleActions(p):
                        val.append(self.computeQValueFromValues(p, action))
                    diff = abs(max(val) - self.values[p])
                    if diff > self.theta:
                        priorityQ.update(p, -diff)