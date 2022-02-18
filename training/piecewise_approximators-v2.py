""" I've finally developed enough mathematics to find the global minimum
of the activation functions. I"ll write out the ideas better somewhere
else, but basically you can find the point (x, y) value where there will
be the maximum difference between the true f(x) and f^~(x) approximation. From
there you can extrapolate on both sides to define the full f^~(x). 

In this code I want to create these approximators, compare them to my v1 
approximators, and to the Amin et al. approximations. """

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
    
class piecewise_approximator:
    
    def method_1(self, m0, b0, iterations):
        m = [m0]; b = [b0]; c = [0]
        #get d vals:
        x0 = 0.01 # starting iteration point; set to past d
        m_prev = m0
        m_next = .5*m_prev
        b_prev = b0
        for i in range(iterations):
            def df_zero(x):
                return self.df(x) - m_next
            d = fsolve(df_zero, x0)[0]
            # print(d)
            left_side = self.func(d) - m_next*d - 2*b_prev
            def f_zero(x):
                return (2*m_prev - m_next)*x - self.func(x) - left_side
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
    
    # this is a mess. i'm sorry.
    def method_2(self, m0 = 1, line_segs = 8):
        m = [m0*2**(-i) for i in range(line_segs)]
        d = [fsolve(lambda x: self.df(x) - m_, 0.01)[0] for m_ in m]
        b = [lambda x, d_=_, m_ = __: (self.func(d_) + self.func(x) - m_*(x + d_))/2 for _, __ in zip(d, m)] # list of functions
        g = [lambda x, d_=_, m_=__, b_=___: (self.func(x) + self.func(d_) - (m_*d_ + b_(x))) for _,__,___ in zip(d, m, b)]
        # G = lambda x: min([g_(x) for g_ in g])
        c = [0]
        for i in range(len(g) - 1):
            c.append((self.func(d[i])-self.func(d[i+1])-m[i]*d[i]+m[i+1]*d[i+1])/(m[i+1] - m[i]))
        yc = [g_(c_) for g_, c_ in zip(g, c)]
        deltas = [yc_ - self.func(c_) for yc_, c_ in zip(yc, c)]
        delta_index = deltas.index(max(deltas))
        cs = [0]*line_segs # list of constants (returned)
        bs = [0]*line_segs
        cs[delta_index] = c[delta_index]
        bs[delta_index] = b[delta_index](cs[delta_index])
        for i in range(delta_index - 1, 0, -1): # what to do if an m is skipped?
            c_next = fsolve(lambda x: g[i](x) - (m[i+1]*x + bs[i+1]), cs[i+1])[0]
            cs[i] = c_next
            bs[i] = b[i](c_next)
        bs[0] = b[0](cs[1])
        for i in range(delta_index + 1, line_segs, 1):
            c_next = fsolve(lambda x: g[i](x) - (m[i-1]*x + bs[i-1]), cs[i-1])[0]
            cs[i] = c_next
            bs[i] = b[i](c_next)
        # for i in range(line_segs - 1):
        #     d_ = d[i]; m_ = m[i]; c_0 = c[i]; c_1 = c[i+1]
        #     b_0 = (self.func(d_) + self.func(c_0) - m_*(c_0 + d_))/2
        #     b_1 = (self.func(d_) + self.func(c_1) - m_*(c_1 + d_))/2
        #     bs.append(min(b_0, b_1))
        # # assume the final line segment must point to the left
        # bs.append((f(d[-1]) + self.func(c[-1]) - m[-1]*(c[-1] + d[-1]))/2)
        return m, bs, cs
    
    def generate_d(self):
        d = [fsolve(lambda x: self.df(x) - m_, 0.01)[0] for m_ in self.m]
        return d
    
    def __init__(self, f, df, g=lambda x: -1*x, asmp=1, name=None, \
                 iterations=8, method=2, m0=1, b0=0):
        self.func = f
        self.df = df
        self.g = g
        self.asmp = asmp
        self.name = name
        if(method == 1):
            self.m, self.b, self.c = self.method_1(m0, b0, iterations)
        else:
            self.m, self.b, self.c = self.method_2(m0, iterations)
        self.d = self.generate_d() # not always necessary, but i'll do it
    
    def f(self, x):
        x_abs = x
        if(x_abs < 0):
            x_abs *= -1
        index = -1
        for i in self.c:
            if x_abs >= i:
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
    
    # for all possible, use points = c + d + [get_asmp_point]
    # next make it also return location of max point(s)
    def get_max_dif(self):
        max_dif = 0
        max_point = 0
        for point in self.c + self.d + [self.get_asmp_point()]:
            if(abs(self.func(point) - self.f(point)) > max_dif):
                max_dif = abs(self.func(point) - self.f(point))
                max_point = point
        return max_dif, max_point
    
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
    
    
    def test_continuity(self, epsilon = 0.0001):
        rtrn = True
        for i in range(1, len(self.c) - 1):
            if(abs(self.m[i-1]*self.c[i]+self.b[i-1] - (self.m[i]*self.c[i]+self.b[i])) > epsilon):
                rtrn = False
        return rtrn
    
    def get_values(self, latex=True, print_it = False):
        if(not latex):
            if(print_it):
                print('{}\n{}\n{}'.format(self.m, self.c, self.b))
            return '{}\n{}\n{}'.format(self.m, self.c, self.b)
        rtrn = '\hline\n m & c & b \\\ \n \hline \n'
        for m_, c_, b_ in zip(self.m, self.c, self.b):
            rtrn += ('{} & {} & {} \\\ \n \hline \n'.format(m_, c_, b_))
        if(print_it):
            print(rtrn)
        return rtrn
        
        
    
    def plot(self, x_max = 16, y_min = 0, y_max = 1, plot_true = False,\
            plot_neg = False, savfig = False, savpath = None):
        x_min = 0
        if(plot_neg):
            x_min = -1*x_max
        num = 1000
        x = np.linspace(x_min, x_max, num)
        y = np.array([self.f(x_) for x_ in x])
        plt.figure(figsize = (15, 4))
        plt.plot(x, y, color = 'k')
        if(plot_true):
            y_true = np.array([self.func(x_) for x_ in x])
            plt.plot(x, y_true, color = 'r', alpha=0.5)
        if(savfig):    
            plt.savefig(savpath, dpi=800)
        plt.show()
        plt.close()
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
    savfig=True
    #sigmoid function
    from math import exp
    f = lambda x: 1/(1 + exp(-1*x))
    df = lambda x: exp(-1*x)/(1+exp(-1*x))**2
    
    f_1 = piecewise_approximator(f, df, g = lambda x: 1-x, m0=.25, b0=.5, method=1)
    f_2 = piecewise_approximator(f, df, g = lambda x: 1-x, m0=.25, method=2)
    f_1.plot(y_max = 1.2, plot_neg = True, plot_true = True, savfig=savfig, savpath="./plots/sig1.png")
    f_2.plot(y_max = 1.2, plot_neg = True, plot_true = True, savfig=savfig, savpath="./plots/sig2.png")
    print("sigmoid, method 1 max dif: " + str(f_1.get_max_dif()))
    print("sigmoid, method 2 max dif: " + str(f_2.get_max_dif()))
    # H.Amin K. M . Curtis B.R. Hayes-Gill sigmoid
    m, b, c = [0.25, 0.125, 0.03125], [0.5, 0.625, 0.84375], [0,1,2.375]
    # I'll just override the instance variables of f_2
    f_2.m = m; f_2.b = b; f_2.c = c
    f_2.plot(y_max = 1.2, plot_neg = True, plot_true = True, savfig=savfig, savpath="./plots/sig3.png")
    print("sigmoid, Amin et al. max dif: " + str(f_2.get_max_dif()))
    #arctan function
    from math import atan, pi
    f = atan
    df = lambda x: 1/(1 + x**2)
    f_1 = piecewise_approximator(f, df, g = lambda x: -1*x, m0=1, b0=0, asmp=pi/2, method=1, iterations=12)
    f_2 = piecewise_approximator(f, df, g = lambda x: -1*x, m0=1, asmp=pi/2, method=2, iterations=12)
    f_1.plot(y_min = -1*(pi/2 + .2),y_max = pi/2 + .2, plot_neg = True, savfig=savfig, savpath="./plots/atan1.png")
    f_2.plot(y_min = -1*(pi/2 + .2),y_max = pi/2 + .2, plot_neg = True, savfig=savfig, savpath="./plots/atan2.png")
    print("arctan, method 1 max dif: " + str(f_1.get_max_dif()))
    print("arctan, method 2 max dif: " + str(f_2.get_max_dif()))
    #tanh function
    from numpy import tanh, cosh
    f = tanh
    df = lambda x: 1/cosh(x)**2
    
    f_1 = piecewise_approximator(f, df, g = lambda x: -1*x, m0=1, b0=0, method=1)
    f_2 = piecewise_approximator(f, df, g = lambda x: -1*x, m0=1, method=2)
    f_1.plot(x_max = 8, y_min = -1.2 ,y_max = 1.2, plot_neg = True, savfig=savfig, savpath="./plots/tanh1.png")
    f_2.plot(x_max = 8, y_min = -1.2 ,y_max = 1.2, plot_neg = True, savfig=savfig, savpath="./plots/tanh2.png")
    print("tanh, method 1 max dif: " + str(f_1.get_max_dif()))
    print("tanh, method 2 max dif: " + str(f_2.get_max_dif()))
    
    #swish function = sigmoid(x)*x
    # f = lambda x: 1/(1 + exp(-1*x))
    # df = lambda x: exp(-1*x)/(1+exp(-1*x))**2
    
    # m, b, c = create_piecewise_approximator(f, df, m0=.25, b0=.5)
    # f_ = piecewise_approximator(m, b, c, g = lambda x: 1 - x, asmp = 1)
    
    # class swish:
    #     def __init__(self, sigmoid):
    #         self.sigmoid = sigmoid
    #     def f(self, x):
    #         return self.sigmoid.f(x)*x
    # swish_ = swish(f_)
    # plot_piecewise_approximator(swish_, f = None, x_max = 4, y_min = -0.5 ,y_max = 4, plot_neg = True, savfig=True, savpath="swsh1.png")
    # plot_piecewise_approximator(swish_, f = lambda x: f(x)*x, x_max = 4, y_min = -0.5 ,y_max = 4, plot_neg = True, savfig=True, savpath="swsh2.png")