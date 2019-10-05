import matplotlib

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


# constants
N_TRAIN = 100
FOLD_COUNT = 10
POLYNOMIAL_DEGREE = 2
INCLUDE_BIAS = True
VALIDATION_COUNT = 10
LAMBDA_VALUES = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]


(countries, features, values) = a1.load_unicef_data()


targets = values[:,1]
x = values[:,7:40]
x_n = a1.normalize_data(x) # x normalized
x_n_train = x_n[0:N_TRAIN,:]
x_n_test = x_n[N_TRAIN:,:]

t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


validation_errors = np.zeros(len(LAMBDA_VALUES))


for lambda_index in range(len(LAMBDA_VALUES)):

    for fold in range(FOLD_COUNT, 0, -1):
        # indexes for partitioning
        idx1 = (fold - 1) * 10
        idx2 = fold * 10

        # x_train = [0:90,:] , [0:80,] + [90:100,] , [0:70,] + [80:100,]
        x_train_part1 =  x_n_train[:idx1,]
        x_train_part2 =  x_n_train[idx2:,]
        x_train_final = np.concatenate((x_train_part1, x_train_part2),axis=0)

        # t_train = [0:90] , [0:80] + [90:100] , [0:70] + [80:100] , ...
        t_train_part1 = t_train[:idx1]
        t_train_part2 = t_train[idx2:]
        t_train_final = np.concatenate((t_train_part1, t_train_part2), axis=0)

        # x_val, t_val = [90:100] , [80:90] , ...
        x_val = x_n_train[idx1:idx2]
        t_val = t_train[idx1:idx2]

        # train training set
        (w, y, tr_error) = a1.linear_regression(x_train_final, t_train_final, 'polynomial', LAMBDA_VALUES[lambda_index], degree=POLYNOMIAL_DEGREE, include_bias=INCLUDE_BIAS)

        # validate againt validation set
        (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, 'polynomial', degree=POLYNOMIAL_DEGREE, include_bias=INCLUDE_BIAS)

        # add error to current lambda errors
        validation_errors[lambda_index] = validation_errors[lambda_index] + te_err

    # average errors for current lambda
    validation_errors[lambda_index] = validation_errors[lambda_index] / FOLD_COUNT


def plot(title, validation_errors):
    # Produce a plot of results.
    plt.semilogx(LAMBDA_VALUES,validation_errors,color='green')
    plt.ylabel('RMS')
    plt.legend(['Training error','Validation error'])
    plt.title(title)
    plt.xlabel('Features')
    plt.show()


min_error = str("%0.2f" % validation_errors.min())
min_lambda = LAMBDA_VALUES[validation_errors.argmin()]
title = "Min Error: " + str(min_error) + " for lambda : " + str(min_lambda)
plot('Errors for Cross Validation: ' + title, validation_errors)
