import numpy as np
import numexpr as ne

from abc import ABC, abstractmethod

class Kernel():
    def kernel(self, X, Y, gamma, var):
        X_norm = -gamma * np.einsum('ij,ij->i', X, X)
        Y_norm = -gamma * np.einsum('i,i->', Y, Y)
        return ne.evaluate('v * exp(A + B + C)', {\
            'A'  :X_norm,\
            'B' : Y_norm,\
            'C' : 2.0 * gamma * np.dot(X, Y.T),\
            'g' : gamma,\
            'v' : var\
        })

class Error_Func():
    def LMS(self, expected, predicted):
        error = expected - predicted
        return self.learning_step * error

    def MCC(self, expected, predicted):
        error = expected - predicted
        mcc_weight = error * np.exp((-error**2) / (2 * self.sigma**2))
        return self.learning_step * mcc_weight

class Quantization():
    def calc_dist(self, new_input):
        diff = self.inputs - new_input
        dist = np.einsum("ij,ij->i", diff, diff)
        min_pos = np.argmin(dist)
        return min_pos, dist[min_pos]

    def Quantized(self, new_weight, new_input):
        min_pos, min_dist = self.calc_dist(new_input)
        if(min_dist < self.epsilon):
            self.weights[min_pos] += float(new_weight)
            self.merge = 1
        else:
            self.inputs.append(new_input)
            self.weights.append(float(new_weight))
            self.merge = 0

    def Non_Quantized(self, new_weight, new_input):
        self.weights.append(float(new_weight))
        self.inputs.append(new_input)

class KAF(Kernel, Error_Func, Quantization):
    def __init__(
        self,
        input,
        output,
        learning_step = 0.5,
        sigma = 1
    ):
        self.learning_step = learning_step
        self.sigma = sigma
        self.inputs = [input]
        self.weights = [float(learning_step * output)]
        self.pred = [0]

    def predict(self, new_input):
        gamma = 1/(2*self.sigma**2)
        kernel_res = self.kernel(self.inputs, new_input, gamma, 1)
        prediction = np.einsum("i,i->", self.weights, kernel_res)
        self.pred.append(prediction)
        return prediction

    @abstractmethod
    def update(self, new_input, expected):
        pass

class KLMS(KAF):
    def __init__(
        self,
        input,
        output,
        learning_step = 0.5,
        sigma = 1
    ):
        super().__init__(input, output, learning_step, sigma)

    def update(self, new_input, expected):
        prediction = self.predict(new_input)
        new_weight = self.LMS(expected, prediction)
        self.Non_Quantized(new_weight, new_input)

    def name(self):
        return "KLMS"

class QKLMS(KAF):
    def __init__(
        self,
        input,
        output,
        epsilon,
        learning_step = 0.5,
        sigma = 1
    ):
        super().__init__(input, output, learning_step, sigma)
        self.epsilon = epsilon

    def update(self, new_input, expected):
        prediction = self.predict(new_input)
        new_weight = self.LMS(expected, prediction)
        self.Quantized(new_weight, new_input)

    def name(self):
        return "QKLMS"

class KMCC(KAF):
    def __init__(
        self,
        input,
        output,
        learning_step = 0.5,
        sigma = 1
    ):
        super().__init__(input, output, learning_step, sigma)

    def update(self, new_input, expected):
        prediction = self.predict(new_input)
        new_weight = self.MCC(expected, prediction)
        self.Non_Quantized(new_weight, new_input)

    def name(self):
        return "KMCC"

class QKMCC(KAF):
    def __init__(
        self,
        input,
        output,
        epsilon,
        learning_step = 0.5,
        sigma = 1
    ):
        super().__init__(input, output, learning_step, sigma)
        self.epsilon = epsilon
        self.merge = 0

    def update(self, new_input, expected):
        prediction = self.predict(new_input)
        new_weight = self.MCC(expected, prediction)
        self.Quantized(new_weight, new_input)

    def name(self):
        return "QKMCC"

