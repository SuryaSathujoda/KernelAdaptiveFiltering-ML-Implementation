import numpy as np
import matplotlib.pyplot as plt
import models
import clustermodels

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
    err_line, = plt.plot(savgol_filter(np.abs(err[:2000]),5,2))
    actual_line, = plt.plot(savgol_filter(np.abs(output[:2000]),5,2))
    pred_line, = plt.plot(savgol_filter(np.abs(p[:2000]),5,2))
    plt.legend((err_line,actual_line,pred_line),("err","actual","pred"))
    plt.show()

input=read_data("../Data/raw_params.txt")
output=read_data("../Data/output.txt")

list_of_models = []
# For sigma >0.7, kmcc work better than kmls
list_of_models.append(models.KLMS(input[0], output[0], 1.3, 0.67))
list_of_models.append(models.KMCC(input[0], output[0], 1.3, 0.67))
list_of_models.append(models.QKLMS(input[0], output[0], 0.0095, 1.3, 0.67))
list_of_models.append(models.QKMCC(input[0], output[0], 0.0095, 1.3, 0.67))
#list_of_models.append(clustermodels.NICE(input[0], output[0], models.QKLMS,
#                                         0.0005, 1.3, 1.5, 0.0001))
#list_of_models.append(clustermodels.NICE(input[0], output[0], models.QKMCC,
#                                         0.0005, 1.3, 1.5, 0.0001))

for model in list_of_models:
    p=np.zeros(output.size)

    for i in range(output.size-1):
        model.update(input[i+1], output[i+1])

    #print(len(model.cluster_size))

    p=model.pred
    output=output.reshape((-1,))
    err=np.subtract(output, p)

    print_results(model.name()+"/pred.txt", np.asarray(p))
    print_results(model.name()+"/err.txt", err)
    #print_results(model.name()+"/weights.txt", np.asarray(model.weights))

    print("MSE = " + str(mean_squared_error(output, p)))
    plot(output, p, err)

    mse=np.zeros(output.size)
    for i in range(output.size-1):
        mse[i] = mean_squared_error(output[:i+1], p[:i+1])

    axis = plt.gca()
    axis.set_ylim([0,0.1])
    plt.plot(mse[:-1], linewidth=0.5)
    plt.show()
