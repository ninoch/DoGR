import pickle
import numpy as np

from EM import *

class DogR(object):
    """
        - number_of_components: number of mixture components ( K )
        - data: N * D matrix, where N is number of data points and D is number of dimensions
        - prioris: K * 1 column-vector, the priors of each component
        - mu: K * D matrix, a 1 * D vector for each component
        - sigma: K * D * D tensor, a D * D matrix for each component
        - coefficient: a K * (D + 1) matrix, for each component the coefficient of dimesion + intercept 
        - y_sigma: a K * 1 column-vector, the variance of Y for each component
    """
    def __init__(self, number_of_components):
        self.number_of_components = number_of_components

    def fit(self, data):
        self.D = np.shape(data)[1]
        self.priors, self.mu, self.sigma, self.coefficients, self.y_sigma, self.ll = EM(data, self.number_of_components)

    def get_ll(self, input_data):
        inpN, inpD = np.shape(input_data)
        gamma = np.ndarray(shape=(inpN, self.number_of_components))

        for k in range(self.number_of_components):
            mu_large = np.append(self.mu[k, :], 0)
            sigma_large = np.insert(np.insert(self.sigma[k, :, :], self.D - 1, 0, axis=1), self.D - 1, 0, axis=0)
            sigma_large[-1, -1] = self.y_sigma[k]

            keep_y = np.copy(input_data[:, -1])
            input_data[:, -1] -= self.coefficients[k, 0] + np.dot(input_data[:, :-1], self.coefficients[k, 1:])
            gamma[:, k] = scipy.stats.multivariate_normal.pdf(input_data, mu_large, sigma_large, allow_singular=True) # CHECK * priors[k]
            input_data[:, -1] = np.copy(keep_y)

        probs = np.dot(gamma, self.priors)

        min_value = sys.float_info.min
        indexes = np.nonzero(probs < min_value)
        indexes = list(indexes)
        indexes = np.reshape(indexes,np.size(indexes))
        probs[indexes] = min_value
        probs = np.reshape(probs, (inpN, 1))
        loglikelihood = np.mean(np.log10(probs), 0)
        return loglikelihood

    def predict(self, input_data):
        inpN, inpD = np.shape(input_data)

        y_pred_components = np.ndarray(shape=(inpN, self.number_of_components))
        for k in range(self.number_of_components):
            y_pred_components[:, k] = self.priors[k] * scipy.stats.multivariate_normal.pdf(input_data, self.mu[k, :], self.sigma[k, :, :], allow_singular=True)

        min_value = sys.float_info[3]
        denominator = np.sum(y_pred_components, axis=1) + min_value # CHECK
        denominator = denominator.reshape((inpN, 1))
        y_pred_components = y_pred_components / denominator

        y_pred = np.sum(np.multiply(y_pred_components, np.matmul(sm.add_constant(input_data), self.coefficients.T)), axis=1)
        return y_pred

    def get_probs(self, input_data):
        inpN, inpD = np.shape(input_data)

        y_pred_components = np.ndarray(shape=(inpN, self.number_of_components))

        for k in range(self.number_of_components):
            mu_large = np.append(self.mu[k, :], 0)
            sigma_large = np.insert(np.insert(self.sigma[k, :, :], self.D - 1, 0, axis=1), self.D - 1, 0, axis=0)
            sigma_large[-1, -1] = self.y_sigma[k]

            keep_y = np.copy(input_data[:, -1])
            input_data[:, -1] -= self.coefficients[k, 0] + np.dot(input_data[:, :-1], self.coefficients[k, 1:])
            y_pred_components[:, k] = self.priors[k] * scipy.stats.multivariate_normal.pdf(input_data, mu_large, sigma_large, allow_singular=True)
            input_data[:, -1] = np.copy(keep_y)

        return y_pred_components

    def get_groups(self, input_data):
        inpN, inpD = np.shape(input_data)

        y_pred_components = np.ndarray(shape=(inpN, self.number_of_components))

        for k in range(self.number_of_components):
            mu_large = np.append(self.mu[k, :], 0)
            sigma_large = np.insert(np.insert(self.sigma[k, :, :], self.D - 1, 0, axis=1), self.D - 1, 0, axis=0)
            sigma_large[-1, -1] = self.y_sigma[k]

            keep_y = np.copy(input_data[:, -1])
            input_data[:, -1] -= self.coefficients[k, 0] + np.dot(input_data[:, :-1], self.coefficients[k, 1:])
            y_pred_components[:, k] = self.priors[k] * scipy.stats.multivariate_normal.pdf(input_data, mu_large, sigma_large, allow_singular=True)
            input_data[:, -1] = np.copy(keep_y)

        return np.argmax(y_pred_components, axis=1)

    def save(self, file_name):
        params = [self.number_of_components, self.D, self.priors, self.mu, self.sigma, self.coefficients, self.y_sigma, self.ll]
        pickle.dump(params, open(file_name, "wb"))

    def load(self, file_name):
        self.number_of_components, self.D, self.priors, self.mu, self.sigma, self.coefficients, self.y_sigma, self.ll = pickle.load(open(file_name, "rb"))
