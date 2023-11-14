import numpy as np
from scipy.stats import norm
import math
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd


def load(filename):
    df = pd.read_csv(filename)
    return df


def multinomial_count(x_column, df):
    #get the frequency of each class label in the column
    result = df.groupby(x_column).size()
    result = result.div(df[x_column].shape[0])
    return result


def get_p_x_given_y(x_column, y_column, df):
    """
   Definition : computes [P(Xi = x | Y = 1), P(Xi = x | Y = 0)] for all Xi = x
    """
    #Get all x that xi can take on
    p_1 = df.groupby(x_column).apply(lambda x: ((x[x_column] == x.name) & (x[y_column] == 1)).sum())
    p_1 = p_1.div(df[x_column].shape[0])
    p_0 = df.groupby(x_column).apply(lambda x: ((x[x_column] == x.name) & (x[y_column] == 0)).sum())
    p_0 = p_0.div(df[x_column].shape[0])
    return [p_0, p_1]


def get_all_p_x_given_y(x_column, y_column, df):
    # We want to store P(X_i=1 | Y=y)
    #This function combines all conditional values into a matrix for each variables
    new = pd.DataFrame()
    #get_p_x_given_y gives p_0 and p_1
    list = get_p_x_given_y(x_column, y_column, df)
    # Add the new DataFrame as a new column to the original DataFrame
    new[x_column, '0'] = list[0]
    new[x_column, '1'] = list[1]
    return new

def matrices(y_column, df) :
    #creates dictionary of matrices for each variable
    variables = {}
    for i,columns in enumerate(df.columns[:5]):
        variables[i] = get_all_p_x_given_y(columns, y_column, df)
    return variables

def get_p_y(y_column, df):
    """
    Compute P(Y = 1)
    """
    y_0 = 0
    y_1 = 0
    for index, row in df.iterrows():
       if row[y_column] == 0:
           y_0 += 1
       elif row[y_column] == 1:
           y_1 += 1
    p_y_1 = (y_1)/(y_0 + y_1)
    return p_y_1


def joint_prob(xs, y_column, df, y, p_y):
    """
    Computes the joint probability of a single row and y
    """

    ### YOUR CODE HERE
    # Hint: P(X, Y) = P(Y) * P(X_1 | Y) * P(X_2 | Y) * ... * P(X_n | Y)
    variables = matrices(y_column, df)
    prob = p_y
    j = 0
    for i in xs:
       values = variables[j]
       if y == 0:
           if j == 0:
               #Blade
               if i == 19:
                   prob *= values.iloc[0, 0]
               elif i == 26:
                   prob *= values.iloc[1,0]
               elif i == 52:
                   prob *= values.iloc[2,0]
               elif i == 76:
                   prob *= values.iloc[3,0]
               else:
                   prob *= values.iloc[4, 0]
           elif j == 1:
               #Velocity
               if i == 5:
                   prob *= values.iloc[0, 0]
               elif i == 8:
                   prob *= values.iloc[1, 0]
               else:
                   prob *= values.iloc[2, 0]
           elif j == 2:
               #Location
               if i == "Con":
                   prob *=  values.iloc[0, 0]
               elif i == "H":
                   prob *= values.iloc[1, 0]
               elif i == "M":
                   prob *= values.iloc[2, 0]
               else:
                   prob *= values.iloc[3, 0]
           elif j == 3:
               #Orient
               if i == "Con":
                   prob *= values.iloc[0, 0]
               elif i == "D":
                   prob *= values.iloc[1, 0]
               elif i == "L":
                   prob *= values.iloc[2, 0]
               else:
                   prob *= values.iloc[3, 0]
           else:
               #Angle
               if i == 135:
                   prob *= values.iloc[0, 0]
               elif i == 45:
                   prob *= values.iloc[1, 0]
               elif i == 90:
                   prob *= values.iloc[2, 0]
               else:
                   prob *= values.iloc[3, 0]
       else:
           if j == 0:
               if i == 19:
                   prob *= values.iloc[0, 1]
               elif i == 26:
                   prob *= values.iloc[1, 1]
               elif i == 52:
                   prob *= values.iloc[2, 1]
               elif i == 76:
                   prob *= values.iloc[3, 1]
               else:
                   prob *= values.iloc[4, 1]
           elif j == 1:
               if i == 5:
                   prob *= values.iloc[0, 1]
               elif i == 8:
                   prob *= values.iloc[1, 1]
               else:
                   prob *= values.iloc[2, 1]
           elif j == 2:
               if i == "Con":
                   prob *=  values.iloc[0, 1]
               elif i == "H":
                   prob *= values.iloc[1, 1]
               elif i == "M":
                   prob *= values.iloc[2, 1]
               else:
                   prob *= values.iloc[3, 1]
           elif j == 3:
               if i == "Con":
                   prob *= values.iloc[0, 1]
               elif i == "D":
                   prob *= values.iloc[1, 1]
               elif i == "L":
                   prob *= values.iloc[2, 1]
               else:
                   prob *= values.iloc[3, 1]
           else:
               if i == 135:
                   prob *= values.iloc[0, 1]
               elif i == 45:
                   prob *= values.iloc[1, 1]
               elif i == 90:
                   prob *= values.iloc[2, 1]
               else:
                   prob *= values.iloc[3, 1]
       if j < 4:
           j += 1
    ### END OF YOUR CODE
    return prob


def get_prob_y_given_x(y, y_column, xs, p_y, df):
    """
    Computes the probability of y given a single row.
    """
    prob_y_given_x = joint_prob(xs, y_column, df, y, p_y)
    prob_y_given_x /= (joint_prob(xs, y_column, df, 0, 1 - p_y) + joint_prob(xs, y_column, df, 1, p_y))

    ### END OF YOUR CODE
    return prob_y_given_x


def compute_accuracy(p_y, df):
    # split the test set into X and y. The predictions should not be able to refer to the test y's.
    X_test = df.drop(columns="Mort")
    y_test = df["Mort"]

    num_correct = 0
    total = len(y_test)

    # we predict 1 if P(Y=1|X) >= 0.5.
    for i, xs in X_test.iterrows():
       num = get_prob_y_given_x(1, "Mort", xs, p_y, df)
       if num >= 0.5 and y_test[i] == 1:
           num_correct += 1
       elif num < 0.5 and y_test[i] == 0:
           num_correct += 1

    accuracy = num_correct / total
    ### END OF YOUR CODE
    return accuracy


def main():
    # load the training set
    df_train = load("/Users/agam/Documents/ORNL_Fish_Draft_Train.csv")

    p_y = get_p_y("Mort", df_train)

    # load the test set
    df_test = load("/Users/agam/Documents/ORNL_Fish_Draft_Test.csv")
    print(matrices("Mort", df_test))
    print(f"Training accuracy: {compute_accuracy(p_y, df_train)}")
    print(f"Test accuracy: {compute_accuracy(p_y, df_test)}")


if __name__ == "__main__":
    main()



