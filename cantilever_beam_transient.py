from main import PINN
import pandas as pd
import tensorflow as tf
import numpy as np
import types

if __name__ == "__main__":
    solution = np.array([0.0, 0.0843201354146, 0.000311157549731, 0.00122306111734, 0.00270409253426, 0.00472263386473,
                         0.00724706612527, 0.0102457720786, 0.0136871328577, 0.0175395291299, 0.0217713452876,
                         0.0263509619981, 0.0312467608601, 0.0364271253347, 0.0418604314327, 0.0475150682032,
                         0.0533594116569, 0.0593618489802, 0.0654907599092, 0.0717145204544, 0.0780015215278],
                        dtype=np.float64)
    pinn = PINN(  # constants declared here so that they are not constructed each time in functions
        constants={'rho': tf.constant(value=2700., shape=(1, 1), dtype=tf.float64),
                   'E': tf.constant(value=70. * 10 ** 9, shape=(1, 1), dtype=tf.float64),
                   'I': tf.constant(value=np.pi * 0.1 ** 4 / 4., shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(value=2.0, shape=(101, 1), dtype=tf.float64),
                   '0': tf.constant(value=0.0, shape=(101, 1), dtype=tf.float64),
                   'solution': tf.expand_dims(tf.convert_to_tensor(solution, dtype=tf.float64), axis=0)},
        data={'collocation': {'x': np.random.uniform(low=0.0, high=2.0, size=21),  # datasets for each training method
                              't': np.random.uniform(low=0.0, high=3.32, size=101)},
              'boundary': {'t': np.random.uniform(low=0.0, high=3.32, size=101)},
              'test': {'x': np.linspace(start=0., stop=2.0, num=21, dtype=np.float64),
                       't': np.array([3.32], dtype=np.float64)}},
        layers=[10, 15, 25, 15, 10, 1],  # neuron per layers
        activation_function='tanh')


    def collocation(self, model, data):
        value = (self.v['c']['E'] / self.v['c']['I'] * self.d(wrt={'x': 4}, model=model, data=data) +
                 self.v['c']['rho'] * self.d(wrt={'t': 2}, model=model, data=data))
        return value


    def boundary(self, model, data):
        y_0_t = model(tf.stack([self.v['c']['0'], data], axis=1))
        y_L_0 = self.d(wrt={'t': 1},
                       model=model,
                       data=tf.stack([self.v['c']['L'], data], axis=1)) - 0.1
        dy_dx_0_t = self.d(wrt={'x': 1},
                           model=model,
                           data=tf.stack([self.v['c']['0'], data], axis=1))
        d2y_dx2_L_t = self.d(wrt={'x': 2},
                             model=model,
                             data=tf.stack([self.v['c']['L'], data], axis=1))
        d3y_dx3_L_t = self.d(wrt={'x': 4},
                             model=model,
                             data=tf.stack([self.v['c']['L'], data], axis=1))
        return y_0_t, dy_dx_0_t, d2y_dx2_L_t, d3y_dx3_L_t, y_L_0


    def equation(self, model, data):
        return self.v['c']['solution'] - model(data)


    def loss_function(self, model, train_data):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'])))
        loss += tf.reduce_sum([tf.reduce_mean(tf.square(bc)) for bc in self.boundary(model, train_data['boundary'])])
        return loss


    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function = types.MethodType(loss_function, pinn)
    pinn.train_network(epochs=5000,
                       # batches={'collocation': 100,
                       #          'boundary': 100},
                       error=10 ** -4,
                       plot_x='x')
