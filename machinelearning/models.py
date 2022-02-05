import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        result = -1
        if nn.as_scalar(self.run(x)) >= 0:
            result = 1
        return result

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            correctPrediction = True
            for x, y in dataset.iterate_once(1):
                if nn.as_scalar(y) != self.get_prediction(x):
                    correctPrediction = False
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
            if correctPrediction:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 20
        self.W0 = nn.Parameter(1, 20)
        self.B0 = nn.Parameter(1, 20)
        self.W1 = nn.Parameter(20, 1)
        self.B1 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        Z0 = nn.Linear(x, self.W0)
        A0 = nn.ReLU(nn.AddBias(Z0, self.B0))
        Z1 = nn.Linear(A0, self.W1)
        return nn.AddBias(Z1, self.B1)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradient = nn.gradients(loss, [self.W0, self.W1, self.B0, self.B1])
                self.W0.update(gradient[0], -0.020)
                self.W1.update(gradient[1], -0.020)
                self.B0.update(gradient[2], -0.020)
                self.B1.update(gradient[3], -0.020)

                if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02:
                    return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 20
        self.W0 = nn.Parameter(784, 100)
        self.B0 = nn.Parameter(1, 100)
        self.W1 = nn.Parameter(100, 10)
        self.B1 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        Z0 = nn.Linear(x, self.W0)
        A0 = nn.ReLU(nn.AddBias(Z0, self.B0))
        Z1 = nn.Linear(A0, self.W1)
        return nn.AddBias(Z1, self.B1)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = -float('inf')
        while accuracy < 0.978:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradient = nn.gradients(loss, [self.W0, self.W1, self.B0, self.B1])
                self.W0.update(gradient[0], -0.10)
                self.W1.update(gradient[1], -0.10)
                self.B0.update(gradient[2], -0.10)
                self.B1.update(gradient[3], -0.10)
            accuracy = dataset.get_validation_accuracy()
            print(accuracy)

class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.numTrainingGames = None
        self.batch_size = 20


        self.learning_rate = 0.04166

        # hidden layer size
        self.hidden_layer_size = 50
        # these four parameters are going to be trained
        '''self.W1 = nn.Parameter(self.state_size, self.hidden_layer_size)
        self.b1 = nn.Parameter(self.hidden_layer_size,10)
        self.W2 = nn.Parameter(self.hidden_layer_size, self.num_actions)
        self.b2 = nn.Parameter(self.batch_size,self.num_actions)'''
        self.W0 = nn.Parameter(self.state_size, 100)
        self.B0 = nn.Parameter(1, 100)
        self.W1 = nn.Parameter(100, 10)
        self.B1 = nn.Parameter(1, self.num_actions)

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(states), Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"
        print("ASD")
        return
        dqGraph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_x = nn.Input(dqGraph, states)
        xW1 = nn.MatrixMultiply(dqGraph, input_x, self.W1)
        xW1_plus_b1 = nn.MatrixVectorAdd(dqGraph, xW1, self.b1)
        ReLU_1 = nn.ReLU(dqGraph, xW1_plus_b1)
        R1W2 = nn.MatrixMultiply(dqGraph, ReLU_1, self.W2)
        R1W2_plus_b2 = nn.MatrixVectorAdd(dqGraph, R1W2, self.b2)
        return dqGraph.get_output(R1W2_plus_b2)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        temp = self.run(states)

