# I lost my old code, time to write it again. This time in python, hopefully better.
import scipy as sp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

def create_piecewise_approximator(f, df, m0 = 1, b0 = 0, iter = 8):
    m = [m0]; b = [b0]; c = [0]
    #get d vals:
    x0 = 0.01 # starting iteration point; set to past d
    m_prev = m0
    m_next = .5*m_prev
    b_prev = b0
    for i in range(iter):
        def df_zero(x):
            return df(x) - m_next
        d = fsolve(df_zero, x0)[0]
        print(d)
        left_side = f(d) - m_next*d - 2*b_prev
        def f_zero(x):
            return (2*m_prev - m_next)*x - f(x) - left_side
        c_next = fsolve(f_zero, x0)[0]
        if(c_next > x0 and c_next < d):
            c.append(c_next)
            m.append(m_next)
            b_next = (m_prev - m_next)*c_next + b_prev
            b.append(b_next)
            m_prev = m_next; b_prev = b_next
        m_next = .5*m_next
        x0 = d
    return m, b, c

class piecewise_approximator:
    
    def __init__(self, m, b, c, g = lambda x: -1*x, asmp = 1, df = None, name=None):
        self.g = g
        self.m = m
        self.b = b
        self.c = c
        self.asmp = asmp
        self.df = None
        self.name = name
    
    def f(self, x):
        x_abs = x
        if(x_abs < 0):
            x_abs *= -1
        index = -1
        for i in self.c:
            if x_abs > i:
                index += 1
        y = self.m[index]*x_abs + self.b[index]
        if(y > self.asmp):
            y = self.asmp
        if(x < 0):
            y = self.g(y)
        return y
    
    def default_df(self, x):
        x_abs = x
        if(x_abs < 0):
            x_abs *= -1
        index = -1
        for i in self.c:
            if x_abs > i:
                index += 1
        dy = self.m[index]
        if(dy > self.asmp):
            y = 0
        if(x < 0):
            y = -1*y # not strictly true, i should multiply by derivative g
        return y
    
    # or, the final c
    def get_asmp_point(self):
        return (self.asmp - self.b[-1])/self.m[-1]
    
    def get_f(self, tensorize = False):
        if(not tensorize):
            return self.f
        import sys
        import tensorflow as tf
        import numpy as np
        from tensorflow.python.framework import ops
        # modified from
        # https://medium.com/@chinesh4/custom-activation-function-in-tensorflow-for-deep-neural-networks-from-scratch-tutorial-b12e00652e24
        # by Chinesh Doshi
        # I think the majority of that text comes from some other stackoverflow
        # answer I saw at some other point but I couldn't find it.
        # edited to fit TensorFlow 2.5.0
        np_f = np.vectorize(self.f)
        np_f_32 = lambda x: np_f(x).astype(np.float32)
        
        df = self.df
        if(df == None):
            df = self.default_df
        np_df = np.vectorize(df)
        np_df_32 = lambda x: np_df(x).astype(np.float32)
        
        def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
            rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+6))
            tf.RegisterGradient(rnd_name)(grad)
            g = tf.compat.v1.get_default_graph()
            with g.gradient_override_map({self.name: rnd_name}):
                return tf.py_function(func, inp, Tout, name=name)
        
        def tf_df(x, name=None):
            with tf.name_scope(self.name) as name:
                y = tf.py_func(np_df_32,[x],[tf.float32],name=name,stateful=False)
                return y[0]
        
        def f_grad(op, grad):
            x = op.inputs[0]
            n_gr = tf_df(x)
            return grad*n_gr
        
        def tf_f(x, name=None):
            with tf.name_scope(self.name) as name:
                y = py_func(np_f_32,[x],[tf.float32],name=name,grad=f_grad)
                y[0].set_shape(x.get_shape())
                return y[0]
        
        return tf_f
        # an attempt at doing it with backend functions (didn't work):
        # import tensorflow.keras.backend as K
        # from tensorflow.keras.layers import Lambda
        # def K_f(x):
        #     x_abs = K.abs(x)
        #     x_0 = K.zeros_like(x)
        #     x_1 = K.zeros_like(x)
        #     for c, m, b in zip(self.c[1:], self.m, self.b):
        #         x_0 = K.switch(K.greater(x_abs, c), lambda: m, lambda: x_0)
        #         x_1 = K.switch(K.greater(x_abs, c), lambda: b, lambda: x_1)
        #     y = Lambda(lambda x: x_0*x_abs + x_1)([x_abs, x_0, x_1])
        #     y = K.switch(K.greater(y, self.asmp), lambda: self.asmp, lambda: y)
        #     y = K.switch(K.less(x, 0), lambda: self.g(y), lambda: y)
        #     return y
        # return K_f
    
def plot_piecewise_approximator(f_, f = None, x_max = 16, y_min = 0, y_max = 1, 
        plot_neg = False, savfig = False, savpath = None):
    x_min = 0
    if(plot_neg):
        x_min = -1*x_max
    num = 1000
    x = np.linspace(x_min, x_max, num)
    y = np.array([f_.f(x_) for x_ in x])
    plt.figure(figsize = (15, 4))
    plt.plot(x, y, color = 'k')
    if(f != None):
        y_true = np.array([f(x_) for x_ in x])
        plt.plot(x, y_true, color = 'r', alpha=0.5)
    if(savfig):    
        plt.savefig(savpath, dpi=800)
    plt.show()
    plt.close()

# works on LSTM models of any depth, with dense top without activation
# units is a list of all LSTM cell units. I could find a way to get this from
# the model but why bother. does not overwrite old model.
def replace_model_activations(model, units, activation, recurrent_activation):
    import tensorflow.keras as keras
    new_layer = keras.layers.LSTM(units[0],return_sequences=True,input_shape=[None,1],
                    activation=activation,recurrent_activation=recurrent_activation)
    layers = [new_layer]
    for i in range(1, len(model.layers) - 1):
        new_layer = keras.layers.LSTM(units[i],return_sequences=True,
                        activation=activation,recurrent_activation=recurrent_activation)
        layers.append(new_layer)
    layers.append(keras.layers.TimeDistributed(keras.layers.Dense(1)))
    
    pw_model = keras.models.Sequential(layers)
    for i in range(len(model.layers)):
        pw_model.layers[i].set_weights(model.layers[i].get_weights())
    return pw_model
    

if __name__ == '__main__':
    savfig=False
    #sigmoid function
    from math import exp
    f = lambda x: 1/(1 + exp(-1*x))
    df = lambda x: exp(-1*x)/(1+exp(-1*x))**2
    
    m, b, c = create_piecewise_approximator(f, df, m0=.25, b0=.5)
    f_ = piecewise_approximator(m, b, c, g = lambda x: 1 - x, asmp = 1)
    plot_piecewise_approximator(f_, f = None, y_max = 1.2, plot_neg = True, savfig=savfig, savpath="sig1.png")
    plot_piecewise_approximator(f_, f = f, y_max = 1.2, plot_neg = True, savfig=savfig, savpath="sig2.png")
    # H.Amin K. M . Curtis B.R. Hayes-Gill sigmoid
    m, b, c = [0.25, 0.125, 0.03125], [0.5, 0.625, 0.84375], [0,1,2.375]
    sig2_ = piecewise_approximator(m, b, c, g = lambda x: 1 - x, asmp = 1)
    plot_piecewise_approximator(sig2_, f = f, y_max = 1.2, plot_neg = True, savfig=savfig, savpath="Aminsig1.png")
    #arctan function
    from math import atan, pi
    f = atan
    df = lambda x: 1/(1 + x**2)
    
    m, b, c = create_piecewise_approximator(f, df, m0=1, b0=0, iter = 12)
    f_ = piecewise_approximator(m, b, c, g = lambda x: -1*x, asmp = pi/2)
    plot_piecewise_approximator(f_, f = None, y_min = -1*(pi/2 + .2),y_max = pi/2 + .2, plot_neg = True, savfig=savfig, savpath="atan1.png")
    plot_piecewise_approximator(f_, f = f, y_min = -1*(pi/2 + .2),y_max = pi/2 + .2, plot_neg = True, savfig=savfig, savpath="atan2.png")
    #tanh function
    from numpy import tanh, cosh
    f = tanh
    df = lambda x: 1/cosh(2*x)
    
    m, b, c = create_piecewise_approximator(f, df, m0=1, b0=0, iter = 8)
    print(m) # notice that 0.25 is skipped in the slopes!
    f_ = piecewise_approximator(m, b, c, g = lambda x: -1*x, asmp = 1)
    plot_piecewise_approximator(f_, f = None, x_max = 8, y_min = -1.2 ,y_max = 1.2, plot_neg = True, savfig=savfig, savpath="tanh1.png")
    plot_piecewise_approximator(f_, f = f, x_max = 8, y_min = -1.2 ,y_max = 1.2, plot_neg = True, savfig=savfig, savpath="tanh2.png")
    
    #swish function = sigmoid(x)*x
    f = lambda x: 1/(1 + exp(-1*x))
    df = lambda x: exp(-1*x)/(1+exp(-1*x))**2
    
    m, b, c = create_piecewise_approximator(f, df, m0=.25, b0=.5)
    f_ = piecewise_approximator(m, b, c, g = lambda x: 1 - x, asmp = 1)
    
    class swish:
        def __init__(self, sigmoid):
            self.sigmoid = sigmoid
        def f(self, x):
            return self.sigmoid.f(x)*x
    swish_ = swish(f_)
    plot_piecewise_approximator(swish_, f = None, x_max = 4, y_min = -0.5 ,y_max = 4, plot_neg = True, savfig=True, savpath="swsh1.png")
    plot_piecewise_approximator(swish_, f = lambda x: f(x)*x, x_max = 4, y_min = -0.5 ,y_max = 4, plot_neg = True, savfig=True, savpath="swsh2.png")