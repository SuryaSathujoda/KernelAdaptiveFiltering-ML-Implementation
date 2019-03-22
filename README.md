# Kernel Adaptive Filtering Toolbox

This toolbox provides implementation of multiple Kernel Adaptive Filters in Python

# Filters
The following are the filters currently available in this toolbox:

* Kernel Least Mean Square (KLMS)
* Quantized Kernel Least Mean Square (Q-KLMS)
* Kernel Maximum Correntropy (KMCC)
* Quantized Kernel Maximum Correntropy (Q-KMCC)

# Requirements
In order for this toolbox to run, the following are the requirements:

* Python 3.4+
* Numpy - `pip3 install numpy`
* Numexpr - `pip3 install numexpr`
* SciPy - `pip3 install scipy`
* Scikit-Learn - `pip3 install sklearn`
* Matplotlib - `pip3 install matplotlib`

# Demo
To apply the models in code, first import the `models.py` module.

`import models.py`

Then initialise any of the desired models by creating an instance of the model class.

```
klms = models.klms(first_input, first_output, sigma, learning_rate)
qklms = models.qklms(first_input, first_output, threshold_distance, sigma, learning_rate)
kmcc = models.kmcc(first_input, first_output, sigma, learning_rate)
qkmcc = models.qkmcc(first_input, first_output, threshold_distance, sigma, learning_rate)
```

As these are Online Learning Methods, as data is available, the respective models are updated using the updated function with the new input and output data.

`model.update(new_input, new_output)`

(Here `model` is the previously initialised `klms, qklms, kmcc, qkmcc`.)

The predictions of the model are available with the `pred` instance variable and the coefficients/weights of the model with the `weights` instance variable.

```
predictions = model.pred
weights = model.weights
```
These values are updated every time the `update` method is called. 
