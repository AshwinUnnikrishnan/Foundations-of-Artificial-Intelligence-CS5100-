# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    from util import Stack
    dfsStack = Stack()
    visited = []
    dfsStack.push((problem.getStartState(), []))

    while not dfsStack.isEmpty():
        currentNode, actionsToReach = dfsStack.pop()
        if problem.isGoalState(currentNode):
            return actionsToReach
        if currentNode not in visited:
            visited.append(currentNode)
            for node in problem.getSuccessors(currentNode):
                dfsStack.push((node[0], actionsToReach + [node[1]]))
            "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    bfsQueue = Queue()
    visited = []
    bfsQueue.push((problem.getStartState(), []))
    while not bfsQueue.isEmpty():
        currentNode, actionsToReach = bfsQueue.pop()
        if problem.isGoalState(currentNode):
            return actionsToReach
        if currentNode not in visited:
            visited.append(currentNode)
            for i in problem.getSuccessors(currentNode):
                bfsQueue.push((i[0], actionsToReach + [i[1]]))
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    ucsQueue = PriorityQueue()
    visited = []
    ucsQueue.push((problem.getStartState(),[],0),0) #Pushing a set of ( StateValue, HowToReachIt, Cost )
    while not ucsQueue.isEmpty():
        currentNode, actionsToReach, prevCost = ucsQueue.pop()
        if problem.isGoalState(currentNode):
            return actionsToReach
        visited.append(currentNode)
        for value, action, cost in problem.getSuccessors(currentNode):   #for each child need to see if the child is already visited or has better cost to reach in the frontier
            if value not in visited:
                #check if child in Frontier and update based on the value
                for index, (p, c, i) in enumerate(ucsQueue.heap):
                    if i[0] == value: # if there is the same node present already in the heap
                        #check if the current priority is lesser than existing
                        if p > (prevCost + cost):
                            #delete the existing and add the new value to heap
                            del ucsQueue.heap[index]
                            ucsQueue.push((value,actionsToReach + [action],prevCost + cost),prevCost + cost)
                        break
                else:
                    ucsQueue.push((value, actionsToReach + [action], prevCost + cost),prevCost + cost)
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    ucsQueue = PriorityQueue()
    visited = []
    ucsQueue.push((problem.getStartState(),[],0),0) #Pushing a set of ( StateValue, HowToReachIt, Cost )
    while not ucsQueue.isEmpty():
        currentNode, actionsToReach, prevCost = ucsQueue.pop()
        prevCost -= heuristic(currentNode, problem)                 #Subtracting the previous heuristic added to reach this
        if problem.isGoalState(currentNode):
            return actionsToReach
        visited.append(currentNode)
        for value, action, cost in problem.getSuccessors(currentNode):   #for each child need to see if the child is already visited or has better cost to reach in the frontier
            if value not in visited:
                #check if child in Frontier and update based on the value
                for index, (p, c, i) in enumerate(ucsQueue.heap):
                    if i[0] == value: # if there is the same node present already in the heap
                        #check if the current priority is lesser than existing Need to check why we cannot skip if already present is explored or frontier
                        if p > (prevCost + cost + heuristic(value,problem)):
                            #delete the existing and add the new value to heap
                            ucsQueue.push((value,actionsToReach + [action],prevCost + cost+ heuristic(value,problem)),prevCost + cost+ heuristic(value,problem))
                        break
                else:
                    ucsQueue.push((value, actionsToReach + [action], prevCost + cost + heuristic(value,problem)),prevCost + cost+ heuristic(value,problem))
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
