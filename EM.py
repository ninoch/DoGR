import numpy as np
import statsmodels.api as sm
import sys
import scipy
import matplotlib.pyplot as plt
from collections import Counter

MAX_DF_SIZE = 10 * 1000

def k_means(data, number_of_components, max_iter = 30): 
    centroids = data[np.random.choice(np.arange(len(data)), number_of_components, replace=False), :]
    for i in range(max_iter):
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in data])
        centroids = [data[C == k].mean(axis = 0) for k in range(number_of_components)]
    C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in data])
    li = []
    for k in range(number_of_components):
        li.extend(np.abs(data[C == k] - centroids[k]))
    dist = np.mean(li)
    return np.array(centroids), C, dist

def EM_init(data, number_of_components):
    passed = False
    while passed == False:
        try:
            N, D = np.shape(data)

            centers, data_closest, dist = k_means(data, number_of_components)
            min_error = dist
            for itr in range(10): #TODO
                a, b, dist = k_means(data, number_of_components)
                if dist < min_error:
                    min_error = dist
                    centers = np.copy(a)
                    data_closest = np.copy(b)

            mu = centers # The last mu ? 
            priors = np.ndarray(shape=(number_of_components, 1))
            sigma = np.ndarray(shape=(number_of_components, D, D))
            coefficients = np.zeros(shape=(number_of_components, D))
            y_sigma = np.ndarray(shape=(number_of_components, 1))


            for k in range(number_of_components):
                idtmp = np.nonzero(data_closest==k)
                idtmp = list(idtmp)
                idtmp = np.reshape(idtmp,(np.size(idtmp)))
                priors[k] = np.size(idtmp)

                sigma[k, :, :] = np.cov(data[idtmp, :].T, bias=1)
                sigma[k, :, :] = sigma[k, :, :] + 0.00001 * np.diag(np.diag(np.ones((D, D))))

                X, Y = data[idtmp, :-1], data[idtmp, -1]

        # Sometimes failed: ValueError: zero-size array to reduction operation maximum which has no identity
                X = sm.add_constant(X, has_constant='add')
                model = sm.OLS(Y,X)
                res = model.fit()
                coefficients[k, :] = res.params
                y_sigma[k] = np.mean((Y - np.dot(X, coefficients[k, :]))**2)

            priors = priors / N

            mu = mu[:, :-1]
            sigma = sigma[:, :-1, :-1]
            passed = True
        except:
            pass

    return (priors, mu, sigma, coefficients, y_sigma)

# Last column must be Y value 
def EM(data, number_of_components, debug=False):
    N, D = np.shape(data)
    D = D - 1

    X, Y = data[:, :-1], data[:, -1]

    if D == 1:
        X = X.reshape((N, 1))
    Y = Y.reshape((N, 1))
    if debug:
        print ("N = ", N, " D = ", D, " K = ", number_of_components)
        print ("Data size: ", data.shape)
        print ("X size: ", X.shape)
        print ("Y size: ", Y.shape)

    if len(data) > MAX_DF_SIZE:
        n = min(len(data), number_of_components*1000)
        sampled_data = data[np.random.randint(data.shape[0], size=n), :]
        priors, mu, sigma, coefficients, y_sigma = EM_init(sampled_data, number_of_components)
    else:
        priors, mu, sigma, coefficients, y_sigma = EM_init(data, number_of_components)

    if debug:
        print ("Priors0 ", priors.shape, " =\n", priors)
        print ("Mu0 ", mu.shape, " =\n", mu)
        print ("Sigma0 ", sigma.shape, " =\n", sigma)
        print ("Coefficients0 ", coefficients.shape, " =\n", coefficients)
        print ("Ysigma0 ", y_sigma.shape, " =\n", y_sigma)

    gamma = np.ndarray(shape=(N, number_of_components))

    min_value = sys.float_info.min
    max_value = sys.float_info.max
    loglikelihood_threshold = 1e-10
    old_loglikelihood = -1 * max_value
    loglikelihood = 0

    iteration = 1

    while True:
        # Expectation
        for k in range(number_of_components):
            mu_large = np.append(mu[k, :], 0)
            sigma_large = np.insert(np.insert(sigma[k, :, :], D, 0, axis=1), D, 0, axis=0)
            sigma_large[-1, -1] = y_sigma[k]

            keep_y = np.copy(data[:, -1])
            data[:, -1] -= coefficients[k, 0] + np.dot(data[:, :-1], coefficients[k, 1:])
            gamma[:, k] = priors[k] * scipy.stats.multivariate_normal.pdf(data, mu_large, sigma_large, allow_singular=True)
            data[:, -1] = np.copy(keep_y)

        denominator = np.sum(gamma, axis=1) # CHECK
        denominator = denominator.reshape((N, 1))   

        denominator = np.where(denominator < min_value, min_value, denominator)

        gamma = gamma / denominator

        if debug:
            print ("GAMMA ~ min: {}, max: {}".format(np.min(gamma), np.max(gamma)))

        n_component = np.sum(gamma, axis=0)
        n_component = n_component.reshape((number_of_components, 1))

        # Maximization 
        for k in range(number_of_components):
            priors[k] = n_component[k] / N
            mu[k, :] = np.dot(gamma[:, k], X) / n_component[k]
            sigma[k, :, :] = np.matmul((X - mu[k]).T, gamma[:, k, np.newaxis] * (X - mu[k])) / n_component[k] + 0.00001 * np.diag(np.diag(np.ones((D, D))))

            model = sm.WLS(Y, sm.add_constant(X), weights=gamma[:, k])
            res = model.fit()
            coefficients[k, :] = res.params
            y_sigma[k] = np.dot(gamma[:, k, np.newaxis].T, np.power(Y - np.dot(coefficients[k, :], sm.add_constant(X).T)[:, np.newaxis], 2)) / n_component[k] # SQRT? 
       
        if debug:
            print ("\t Updated all parameters!")

        # Log-likelihood
        new_gamma = np.ndarray(shape=(N, number_of_components))
        for k in range(number_of_components):
            mu_large = np.append(mu[k, :], 0)
            sigma_large = np.insert(np.insert(sigma[k, :, :], D, 0, axis=1), D, 0, axis=0)
            sigma_large[-1, -1] = y_sigma[k]

            keep_y = np.copy(data[:, -1])
            data[:, -1] -= coefficients[k, 0] + np.dot(data[:, :-1], coefficients[k, 1:])
            new_gamma[:, k] = scipy.stats.multivariate_normal.pdf(data, mu_large, sigma_large, allow_singular=True) # CHECK * priors[k]
            data[:, -1] = np.copy(keep_y)

        if debug:
            print ("\t Computed New Gamma!")

        probs = np.dot(new_gamma, priors)

        probs = np.where(probs < min_value, min_value, probs)
        probs = np.reshape(probs, (N, 1))

        if debug:
            print ("\t Computed probs!")
        loglikelihood = np.mean(np.log10(probs), 0)

        ret_ll = np.sum(np.log(probs))
        if np.absolute((loglikelihood / old_loglikelihood) - 1) < loglikelihood_threshold: 
            break

        iteration += 1
        if iteration > 1000:
            break
        old_loglikelihood = loglikelihood
    return priors, mu, sigma, coefficients, y_sigma, ret_ll
