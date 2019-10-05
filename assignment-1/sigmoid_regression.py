import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


# constants
N_TRAIN = 100
POLYNOMIAL_DEGREE = 3
FEATURE = 11
MU = [100, 10000]
S = 2000


(countries, features, values) = a1.load_unicef_data()


targets = values[:,1]
x = values[:,FEATURE-1]
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


feature_names_for_plot = []
training_errors = []
test_errors = []


# Pass the required parameters to these functions
(w, y, tr_error) = a1.linear_regression(x_train, t_train, 'sigmoid', 0, mu=MU, s=S, include_bias=True)
training_errors.append(tr_error)

(t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'sigmoid', mu=MU, s=S, include_bias=True)
test_errors.append(te_err)


def plot_bar_chart(title, trn_errors, tst_errors):
    print("Training Error: %s , Testing Error: %s" % (training_errors , test_errors))

    # Produce a plot of results.
    index = np.arange(len(trn_errors))
    bar_width = 0.01
    plt.rcParams.update({'font.size': 12})
    plt.bar(index, trn_errors, bar_width, color='g', label='Training error')
    plt.bar(index + bar_width + 0.02, tst_errors, bar_width, color='r', label='Training error')
    plt.ylabel('RMS')
    plt.xticks(index, [])
    plt.legend(['Training error','Testing error'])
    plt.title(title)
    plt.xlabel('Error')
    plt.show()


def plot_learned_function(title):
    # Use linspace to get a set of samples on which to evaluate
    x_ev = np.linspace(np.asscalar(min(x)), np.asscalar(max(x)), num=500)

    # learn y_ev based on x_ev
    x_ev_reshaped=np.reshape(x_ev, (-1,1))
    phi = a1.design_matrix(x_ev_reshaped, 'sigmoid', miu=MU, s=S, include_bias=True)
    y_ev = phi * w

    plt.plot(x_train, t_train, 'bo', markersize=3)
    plt.plot(x_test, t_test,'g^', markersize=3)
    plt.plot(x_ev, y_ev, 'r.-')

    plt.xlim(0,200000)
    plt.ylim(0, 200)

    plt.xlabel('x')
    plt.ylabel('t')

    plt.title(title)
    plt.show()


plot_learned_function('Fit for (GNI per capita (US$) 2011) - sigmoid')
plot_bar_chart('Error comparison for (GNI per capita (US$) 2011) - sigmoid', training_errors, test_errors)
