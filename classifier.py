"""
PROBLEM BACKGROUND:
    For a person who wants to visit a circus, take judgement params
    to be WEATHER_BOOLEAN, FRIEND_BOOLEAN and PROXIMITY_BOOLEAN.
    On 6 out of the 8 possibilities, we train a neural network which
    tries out different combinations of weights, and updates variables
    (weights for now, not including biases for simplicty) on correct
    prediction on test data.
"""

from itertools import permutations as P
import numpy as np
from sklearn.metrics import accuracy_score


class SimpleNetwork(object):
    """
    Here, the network should figure out by itself
    what weight to assign to which parameter (out of
    WEATHER_BOOLEAN, FRIEND_BOOLEAN and PROXIMITY_BOOLEAN,
    as per the testing example), so that the same
    configuration of vars when used in testing data, yields
    a good accuracy.
    """

    def __init__(self):
        self.optimum_weight = []   # Stored after training
        # self.bias = 0            # Model bias, excluded for simplicity
        self.threshold = -8

    def return_weighted_sum(inputs, weights):
        """
        A helper function that does exactly what it's name says.

        : param: inputs:        List of inputs
        : param: weights:       List of corresponding weights

        Returns:
            A float value, viz. the weighted sum.
        """
        zipped_pair = zip(inputs, weights)
        weighted_sum = 0.
        for pair in zipped_pair:
            weighted_sum += zipped_pair[0] * zipped_pair[1]
        return weighted_sum

    def fit(self, X_train, y_true, rate=0.1, max_epochs=1000):
        """
        The training method.

        Tries out possible combinations of weights,
        and updates them whenever accuracy increases.

        By "tries out", it is meant:
            - Make a prediction, and compare it's accuracy with the
            set with the correct label (training set).

        Further, by making the prediction, it is meant:
            - Calculate the weighted sum of input-weight pair.
            - Check if this sum is greater than (for now) 0 (a constant
            bias).

        ----------------------------------------------------------------------
        :param: X_train:        2D list of training examples
        :param: y_true:        List of corresponding labels (preferably 0-1)
        :param: rate:           Learning rate of model (default 0.1)
        :param: max_epochs:     As the name says (default 1000)

        """

        # Assuming for simplicity that optimum learning rate lies
        # between -5 and 5.
        wt_range_lower = -5
        wt_range_upper = 6

        total_epochs = float(wt_range_upper-wt_range_lower) / rate

        weights = np.arange(wt_range_lower, wt_range_upper, step=rate)
        # Values between `wt_range_lower` and `wt_range_upper` are
        # checked for accuracy. The value where accuracy is max is finally
        # set as the optimum weight.

        '''
        Better solution:
            Create a temporary list of (here) 3 size.
            Multiply corresponding elements to all examples to update them.
        # Status: not happening.
        Using `itertools` instead.
        '''

        num_features = len(X_train[0])
        # LogicalError when ``r > len(weights)``
        wt_permutations = np.array(list(P(weights, r=num_features)))

        '''
        Multiplying each feature by a permutation, and
        calculating accuracy for each weight-permutation,
        by calculating the weighted sum for each weight-permutation.
        '''
        best_accuracy_so_far = 0.

        for weight in wt_permutations:
            modified_X_train = []
            for feature_set in X_train:
                modified_X_train.append(np.multiply(feature_set, weight).tolist())

            # Finding weighted sum and corresponding output
            this_wt_sum = 0.
            y_pred = []
            for feature in modified_X_train:
                for val in feature:
                    this_wt_sum += val
                if this_wt_sum >= self.threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

            this_accuracy = accuracy_score(y_true, y_pred)
            if this_accuracy > best_accuracy_so_far:
                best_accuracy_so_far = this_accuracy
                self.optimum_weight = weight
            print('\n\n=========================')
            # print('Epochs: {} out of {}'.format(wt_permutations.index(weight),
            #                                     total_epochs))
            print('Accuracy: {}'.format(this_accuracy))
            print('=============================')

        print('Max accuracy achieved: {}'.format(best_accuracy_so_far))
        print('Weights set: {}'.format(self.optimum_weight))


"""
INTUITION:

    case_n = [<WEATHER_BOOLEAN>, <FRIEND_BOOLEAN>, <PROXIMITY_BOOLEAN>]

    A list of such cases behaves like the training set.
    It's like a list of all cases we want our model to know.

"""

# FEATURES (kinda)
case1 = [0, 0, 0]    # No event favours.
case2 = [0, 0, 1]    # Only `Proximity` favours.
case3 = [0, 1, 0]    # Only `Friend` favours.
case4 = [0, 1, 1]    # Only `Weather` resists.
case5 = [1, 0, 0]    # Only `Weather` favours.
case6 = [1, 0, 1]    # Only `Friend` resists.
case7 = [1, 1, 0]    # Only `Proximity` resists.
case8 = [1, 1, 1]    # All events favour.

# LABELS (kinda)
output1 = 0
output2 = 1
output3 = 1
output4 = 1
output5 = 0
output6 = 1
output7 = 1
output8 = 1
# `output_n` corresponds to `case_n`

X_train = [case1, case2, case3, case6, case7, case8]
y_train = [output1, output2, output3, output6, output7, output8]

X_test = [case4, case5]
y_test_correct = [output4, output5]
# `y_test` to be predicted against `y_test_correct` (#supervised_learning).
