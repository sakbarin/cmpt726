import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


# constants
N_TRAIN = 100
POLYNOMIAL_DEGREE = 3
FEATURES = [11, 12, 13]
PLOT_MAX_X_TO_SHOW = {11: 175000, 12: 200, 13: 200}


(countries, features, values) = a1.load_unicef_data()


targets = values[:,1]
x = values[:,7:15]
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


# variable for holding feature names for the plot
feature_names_for_plot = []

# variables for fitting without bias term
training_errors = []
test_errors = []

# variables for fitting with bias term
training_b_errors = []
test_b_errors = []


for feature in range(np.size(x, 1)):
    # Pass the required parameters to these functions
    (w, y, tr_error) = a1.linear_regression(x_train[:,feature], t_train, 'polynomial', 0, degree=POLYNOMIAL_DEGREE, include_bias=False)
    training_errors.append(tr_error)

    (t_est, te_err) = a1.evaluate_regression(x_test[:, feature], t_test, w, 'polynomial', degree=POLYNOMIAL_DEGREE, include_bias=False)
    test_errors.append(te_err)

    (w_b, y_b, tr_b_error) = a1.linear_regression(x_train[:,feature], t_train, 'polynomial', 0, degree=POLYNOMIAL_DEGREE, include_bias=True)
    training_b_errors.append(tr_b_error)

    (t_b_est, te_b_err) = a1.evaluate_regression(x_test[:, feature], t_test, w_b, 'polynomial', degree=POLYNOMIAL_DEGREE, include_bias=True)
    test_b_errors.append(te_b_err)

    feature_names_for_plot.append("f" + str(feature + np.size(x, 1)))


# Produce a plot of results.
def plot_errors(title, trn_errors, tst_errors):
    index = np.arange(len(trn_errors))
    bar_width = 0.35
    plt.rcParams.update({'font.size': 12})
    plt.bar(index, trn_errors, bar_width, color='g', label='Training error')
    plt.bar(index + bar_width + 0.02, tst_errors, bar_width, color='r', label='Training error')
    plt.ylabel('RMS')
    plt.xticks(index, feature_names_for_plot)
    plt.legend(['Training error','Testing error'])
    plt.title(title)
    plt.xlabel('Feature No.')
    plt.show()


def plot_curve(title, w, feature_no, max_x_limit):
    # Plot a curve showing learned function.

    # Use linspace to get a set of samples on which to evaluate
    x2_ev = np.linspace(np.asscalar(min(x[:,feature_no])), np.asscalar(max(x[:,feature_no])), num=500)
    x2_ev = np.reshape(x2_ev, (-1, 1))

    # TO DO::
    phi = a1.design_matrix(x2_ev, 'polynomial', poly_degree=POLYNOMIAL_DEGREE, include_bias=True)
    y_ev = phi * w

    plt.xlim(0, max_x_limit)
    plt.plot(x_train, t_train, 'bo', markersize=3)
    plt.plot(x_test, t_test, 'g^', markersize=3)
    plt.plot(x2_ev, y_ev, 'r.-')

    plt.xlabel('x')
    plt.ylabel('t')

    plt.title(title)
    plt.show()


plot_errors('Single Feature, Polynomial, With Bias Term', training_b_errors, test_b_errors)
plot_errors('Single Feature, Polynomial, Without Bias Term', training_errors, test_errors)


for i in FEATURES:
    max_x_limit = PLOT_MAX_X_TO_SHOW[i]
    feature_no = i - 8
    (w, y, tr_error) = a1.linear_regression(x_train[:,feature_no], t_train, 'polynomial', 0, degree=POLYNOMIAL_DEGREE, include_bias=True)
    plot_curve('Fit for feature ' + features[i - 1] + ' - polynomial deg. 3', w, feature_no, max_x_limit)
