import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


# constants
N_TRAIN = 100
POLYNOMIAL_DEGREE = 6
INCLUDE_BIAS = True


(countries, features, values) = a1.load_unicef_data()


targets = values[:,1]
x = values[:,7:]
x_n = a1.normalize_data(x) # x normalized

x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]

x_n_train = x_n[0:N_TRAIN,:]
x_n_test = x_n[N_TRAIN:,:]

t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


training_errors = {}
training_n_errors = {}

test_errors = {}
test_n_errors = {}

for i in range(1, POLYNOMIAL_DEGREE + 1):
    # Pass the required parameters to these functions
    (w, y, tr_error) = a1.linear_regression(x_train, t_train, 'polynomial', 0, degree=i, include_bias=INCLUDE_BIAS)
    training_errors[i] = tr_error

    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', degree=i, include_bias=INCLUDE_BIAS)
    test_errors[i] = te_err

    (w_n, y_n, tr_n_error) = a1.linear_regression(x_n_train, t_train, 'polynomial', 0, degree=i, include_bias=INCLUDE_BIAS)
    training_n_errors[i] = tr_n_error

    (t_n_est, te_n_err) = a1.evaluate_regression(x_n_test, t_test, w_n, 'polynomial', degree=i, include_bias=INCLUDE_BIAS)
    test_n_errors[i] = te_n_err


def plot(title, trn_errors, tst_errors):
    # Produce a plot of results.
    plt.rcParams.update({'font.size': 12})
    plt.plot(list(trn_errors.keys()), list(trn_errors.values()))
    plt.plot(list(tst_errors.keys()), list(tst_errors.values()))
    plt.ylabel('RMS')
    plt.legend(['Training error','Testing error'])
    plt.title(title)
    plt.xlabel('Polynomial degree')
    plt.show()


plot('Without Normalization - Polynomial (1 to 6)', training_errors, test_errors)
plot('Normalized - Polynomial (1 to 6)', training_n_errors, test_n_errors)
