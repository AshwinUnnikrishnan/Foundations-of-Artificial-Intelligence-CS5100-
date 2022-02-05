# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currentScore = successorGameState.getScore()
        for food in newFood:
            foodDist = util.manhattanDistance(food, newPos)
            if (foodDist) != 0:
                currentScore += 5/foodDist #More value for the closer food and lesser for farther
        currentScore -= len(currentGameState.getCapsules())
        for ghost in newGhostStates:
            ghostDist = util.manhattanDistance(ghost.getPosition(), newPos)
            if ghostDist > 1:
                currentScore += 5/ghostDist #
            else:
                currentScore -= 100                 # if the next state ghost present then decrease the score more
        return currentScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is completed
                return self.evaluationFunction(gameState)
            if agent == 0: #ghost
                utilities = []
                for newState in gameState.getLegalActions(agent):
                    utilities.append(minimax(1, depth, gameState.generateSuccessor(agent, newState))) #foreach Branching call the minimax with ghost as agent
                return max(utilities) #maximizing pacman value
            else: #Any agent other than 0 will be ghost or adverserial
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent: #if we have completed the ghost reset to pacman
                    nextAgent = 0
                    depth += 1              #increase depth to next once all the agents are done in this level
                utilities = []
                for newState in gameState.getLegalActions(agent):#for each ghost get the utilities
                    utilities.append(minimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)))
                return min(utilities)#return the minimum of the utility

        maxUtility = None
        for branching in gameState.getLegalActions(0): #Building the minmaxTree from top
            utility = minimax(1, 0, gameState.generateSuccessor(0, branching))
            if maxUtility == None or utility > maxUtility:
                branch = branching
                maxUtility = utility
        return branch

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(agent, depth, game_state, alpha, beta):  #maximizing function
            v = float("-inf")
            for branching in game_state.getLegalActions(agent):
                v = max(v, alphabetapruning(1, depth, game_state.generateSuccessor(agent, branching), alpha, beta))
                if v > beta: # if the v is more than beta we dont have to look forward in successors of the node
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(agent, depth, game_state, alpha, beta):  # minimizing function
            v = float("inf")
            next_agent = agent + 1
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
                depth += 1
            for branching in game_state.getLegalActions(agent):
                v = min(v, alphabetapruning(next_agent, depth, game_state.generateSuccessor(agent, branching), alpha, beta))
                if v < alpha: # if the v is less than alpha we dont have to look forward in the branching as the value will be less than or equal to v which is less than alpha
                    return v
                beta = min(beta, v)
            return v

        def alphabetapruning(agent, depth, game_state, a, b):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:
                return self.evaluationFunction(game_state)
            if agent == 0:
                return max_value(agent, depth, game_state, a, b)
            else:
                return min_value(agent, depth, game_state, a, b)

        utility = None
        alpha = float("-inf")
        beta = float("inf")
        for branching in gameState.getLegalActions(0):
            ghostMax = alphabetapruning(1, 0, gameState.generateSuccessor(0, branching), alpha, beta)
            if utility == None or ghostMax > utility:
                utility = ghostMax
                action = branching
            alpha = max(alpha, utility)
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agent, depth, gameState): #Same as the minimax but with probability of branching multiplied to the utility
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is Completed
                return self.evaluationFunction(gameState)
            if agent == 0:  #pacman
                utilities = []
                for branching in gameState.getLegalActions(agent):
                    utilities.append(expectimax(1, depth, gameState.generateSuccessor(agent, branching)))
                return max(utilities)
            else:
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                    depth += 1
                val = 0
                for newState in gameState.getLegalActions(agent):
                    val += expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState))
                return val/len(gameState.getLegalActions(agent))    # Probability of chosing a branch would be 1/(# of actions)

        maxUtility = None
        for branching in gameState.getLegalActions(0):
            utility = expectimax(1, 0, gameState.generateSuccessor(0, branching))
            if maxUtility == None or utility > maxUtility:
                maxUtility = utility
                action = branching
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #we need to eat ghost, capsules to get extra points
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    minDistanceFood = -1
    for food in newFood:
        distance = util.manhattanDistance(newPos, food)
        if minDistanceFood == -1 or minDistanceFood >= distance:
            minDistanceFood = distance

    distGhost = 1
    nearGhost = 0
    for ghost in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(newPos, ghost)
        distGhost += distance
        if distance <= 1:
            nearGhost += 1
    numberOfCapsules = len(currentGameState.getCapsules())
    #combining above metrics in some weight if mindistance is more the food value is less that we add, the distGhost is less then the ghost value is more that we subtract and we want to reduce the number of capsules in order to eat ghost
    return currentGameState.getScore() + (1 / minDistanceFood) - (
                1 / distGhost) - nearGhost - numberOfCapsules
# Abbreviation
better = betterEvaluationFunction
