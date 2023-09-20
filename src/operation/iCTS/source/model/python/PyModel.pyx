# distutils: language = c++
from catboost import CatBoostRegressor
from xgboost import XGBRegressor, Booster
from sklearn.linear_model import LinearRegression
import numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
import matplotlib.pyplot as plt
import joblib

# data transform
def dataTansform(const vector[vector[double]] & cpp_X, const vector[double] & cpp_y):
    p_X = np.array([cpp_X]).T.reshape(-1, cpp_X.size())
    p_y = np.array([cpp_y]).reshape(-1, 1)
    return p_X, p_y

# load model
cdef public pyLoadModel(const string& path):
    model = joblib.load(path.decode('UTF-8'))
    return model

# fit model
cdef public pyLinearModel(const vector[vector[double]] & cpp_X, const vector[double] & cpp_y):
    model = LinearRegression()
    p_X, p_y = dataTansform(cpp_X, cpp_y)
    model.fit(p_X, p_y)
    return model

cdef public pyCatBoostModel(const vector[vector[double]] & cpp_X, const vector[double] & cpp_y):
    model = CatBoostRegressor()
    p_X, p_y = dataTansform(cpp_X, cpp_y)
    model.fit(p_X, p_y, verbose=False)
    return model

cdef public pyXGBoostModel(const vector[vector[double]] & cpp_X, const vector[double] & cpp_y):
    model = XGBRegressor()
    p_X, p_y = dataTansform(cpp_X, cpp_y)
    model.fit(p_X, p_y)
    return model

cdef public double pyPredict(const vector[double] & cpp_X, model):
    p_X = np.array([cpp_X]).reshape((1, -1))
    return model.predict(p_X)

# plot
cdef public pyGenAX():
    fig, ax = plt.subplots()
    tp = (fig, ax)
    return tp

cdef public void pySaveFig(fig, const string & filename):
    plt.legend()
    fig.savefig(filename, dpi=300)

cdef public void pyPlot(ax, const vector[double] x, const vector[double] y, const string & name):
    if x.size() != y.size():
        raise ValueError("x and y must have same size")
    size = x.size()
    if size == 0:
        raise ValueError("x and y must have at least one element")
    
    label = name.decode('UTF-8') if name != "" else None
    if size == 1:
        ax.plot(x[0], y[0], 'o', label=label)
    elif size == 2:
        ax.plot([x[0], x[1]], [y[0], y[1]], 'o-', label=label)
    else:
        ax.plot(x, y, 'o-', label=label)