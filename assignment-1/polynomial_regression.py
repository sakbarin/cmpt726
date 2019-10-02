import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)


POLYNOMIAL_DEGREE = 6
N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


# Pass the required parameters to these functions
training_errors = {}
test_errors = {}

for i in range(1, POLYNOMIAL_DEGREE + 1):
    (w, tr_error) = a1.linear_regression(x_train, t_train, 'polynomial', 0, i)
    training_errors[i] = tr_error

    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, i)
    test_errors[i] = te_err


# Produce a plot of results.
plt.rcParams.update({'font.size': 12})
plt.plot(list(training_errors.keys()), list(training_errors.values()))
plt.plot(list(test_errors.keys()), list(test_errors.values()))
plt.ylabel('RMS')
plt.legend(['Training error','Testing error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
