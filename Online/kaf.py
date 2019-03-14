import numpy as np
import matplotlib.pyplot as plt
import models

from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

def read_data(file):
    file_read=open(file)
    read_buffer=[]
    for line in file_read.readlines():
        y=[float(value) for value in line.split()]
        read_buffer.append(y)
    return np.asarray(read_buffer)

def print_results(file, data):
    output_file=open(file, "w")
    for i in range(data.size):
        output_file.write(str(data[i])+"\n")

def plot(output, p, err):
    err_line, = plt.plot(savgol_filter(np.abs(err[:500]),5,2))
    actual_line, = plt.plot(savgol_filter(np.abs(output[:500]),5,2))
    pred_line, = plt.plot(savgol_filter(np.abs(p[:500]),5,2))
    plt.legend((err_line,actual_line,pred_line),("err","actual","pred"))
    plt.show()

input=read_data("../Data/raw_params.txt")
output=read_data("../Data/output.txt")

model=models.QKLMS(input.shape[1], 0.1, 0.2, 2.25)
p=np.zeros(output.size)

for i in range(output.size):
    model.update(input[i], output[i])

p=model.pred
output=output.reshape((-1,))
err=np.subtract(output, p)

print_results(model.name()+"/pred.txt", np.asarray(p))
print_results(model.name()+"/err.txt", err)
print_results(model.name()+"/weights.txt", np.asarray(model.weights))

print("MSE = " + str(mean_squared_error(output, p)))
plot(output, p, err)
