import numpy as np
from scipy.stats import norm
import math
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd


def load(filename, include_demographics=True):
    df = pd.read_csv(filename)
    if not include_demographics:
        df = df.drop(columns=["Demographic"])
    multinomial_count("3 Idiots", df)
    return df


def multinomial_count(x_column, df):
    #get the frequency of each class label in the column
    result = df.groupby(x_column).size()
    result = result.div(df[x_column].shape[0])
    return result


def get_p_x_given_y(x_column, y_column, df):
    """
    Computes P(X = 1 | Y = 1) and P(X = 1 | Y = 0), where X is a single feature (column).
    x_column: name of the column containing the feature X.
    y_column: name of the class containing the class label.

    return: [P(X = 1 | Y = 1), P(X = 1 | Y = 0)]
    """

    ### YOUR CODE HERE
    # Hint: you can check which rows of a feature (e.g. "x2") are equal to some value (e.g. 1) by doing
    # df["x2"] == 1.
    # Hint: remember to use Laplace smoothing

    y_0 = 0
    y_1 = 0
    x_1_and_y_0 = 0
    x_1_and_y_1 = 0
    #getting list of indices0

    for index, row in df.iterrows():
       if row[y_column] == 0:
           y_0 += 1
           if row[x_column] == 1:
                x_1_and_y_0 += 1
       elif row[y_column] == 1:
           y_1 += 1
           if row[x_column] == 1:
               x_1_and_y_1 += 1

    p_0 = (x_1_and_y_0 + 1)/(y_0 + 2)
    p_1 = (x_1_and_y_1 + 1) / (y_1 + 2)

    ### END OF YOUR CODE
    return [p_0, p_1]


def get_all_p_x_given_y(y_column, df):
    # We want to store P(X_i=1 | Y=y) in p_x_given_y[i][y]
    all_p_x_given_y = np.zeros((df.shape[1] - 1, 2))

    ### YOUR CODE HERE
    # Hint: df.columns gives a list of all the columns of the DataFrame.
    # Hint: remember to skip the "Label" column.

    for i, column in enumerate(df.columns[:-1]):
        #get_p_x_given_y gives p_0 and p_1
        list = get_p_x_given_y(i, y_column, df)
        all_p_x_given_y[i][0] = list[0]
        all_p_x_given_y[i][1] = list[1]
        print(list[1])


    ### END OF YOUR CODE
    return all_p_x_given_y


def get_p_y(y_column, df):
    """
    Compute P(Y = 1)
    """

    ### YOUR CODE HERE
    #find all places where y takes on some value, then divide by the value of y = 1

    y_0 = 0
    y_1 = 0

    for index, row in df.iterrows():
       if row[y_column] == 0:
           y_0 += 1
       elif row[y_column] == 1:
           y_1 += 1

    p_y_1 = (y_1)/(y_0 + y_1)
    print (p_y_1)
    return p_y_1


def joint_prob(xs, y, all_p_x_given_y, p_y):
    """
    Computes the joint probability of a single row and y
    """

    ### YOUR CODE HERE
    # Hint: P(X, Y) = P(Y) * P(X_1 | Y) * P(X_2 | Y) * ... * P(X_n | Y)
    #what is xs

    prob = p_y

    for i in range(len(all_p_x_given_y)):
        if xs[i] == 0:
            prob = prob * (1 - all_p_x_given_y[i][y])
        else:
           prob = prob * all_p_x_given_y[i][y]
    ### END OF YOUR CODE

    return prob


def get_prob_y_given_x(y, xs, all_p_x_given_y, p_y):
    """
    Computes the probability of y given a single row.
    """

    n, _ = all_p_x_given_y.shape  # n is the number of features/columns

    ### YOUR CODE HERE
    # Hint: use the joint probability function.


    #just return the joint probability`
    prob_y_given_x = joint_prob(xs, y, all_p_x_given_y, p_y)
    prob_y_given_x /= (joint_prob(xs, 0, all_p_x_given_y, 1 - p_y) + joint_prob(xs, 1, all_p_x_given_y, p_y))

    ### END OF YOUR CODE
    return prob_y_given_x


def compute_accuracy(all_p_x_given_y, p_y, df):
    # split the test set into X and y. The predictions should not be able to refer to the test y's.
    X_test = df.drop(columns="Label")
    y_test = df["Label"]

    num_correct = 0
    total = len(y_test)

    ### YOUR CODE HERE
    # Hint: we predict 1 if P(Y=1|X) >= 0.5.
    # Hint: to loop over the rows of X_test, use:
    #       for i, xs in X_test.iterrows():

    topThree = {}
    for i, xs in X_test.iterrows():
       num = get_prob_y_given_x(1, xs, all_p_x_given_y, p_y)
       if num >= 0.5 and y_test[i] == 1:
           num_correct += 1
       elif num < 0.5 and y_test[i] == 0:
           num_correct += 1

    accuracy = num_correct / total


    j = 0
    for i in range(len(all_p_x_given_y)):
        factor = get_p_x_given_y(df.columns[j], "Label", df)[0] * (1 - get_p_y("Label", df)) + get_p_x_given_y(df.columns[j], "Label", df)[1] * (get_p_y("Label", df))
        topThree[i] = (all_p_x_given_y[i][1] * (1 - factor))/ ((1 - all_p_x_given_y[i][1]) * factor)
        if j <= 19 :
            j += 1
    print("List: ", topThree)
    sorted_result = sorted(topThree.items(), key=lambda x: x[1], reverse=True)
    print("Result: ", sorted_result)


    ### END OF YOUR CODE
    return accuracy


def main():
    # load the training set
    df_train = load("/Users/agam/Downloads/pset6data/netflix-train.csv", include_demographics=False)

    # compute model parameters (i.e. P(Y), P(X_i|Y))
    all_p_x_given_y = get_all_p_x_given_y("Label", df_train)

    p_y = get_p_y("Label", df_train)

    # load the test set
    df_test = load("/Users/agam/Downloads/pset6data/netflix-test.csv", include_demographics=False)

    print(f"Training accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_train)}")
    print(f"Test accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_test)}")


if __name__ == "__main__":
    main()



