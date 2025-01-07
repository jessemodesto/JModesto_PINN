from main import PINN
import tensorflow as tf
import numpy as np
import types
import csv
import time

if __name__ == "__main__":
    start = time.time()
    pinn = PINN(
        constants={'I': tf.constant(value=np.pi * 0.01 ** 4 / 4., shape=(1, 1), dtype=tf.float64),
                   # 'I': tf.constant(value=1.0, shape=(1, 1), dtype=tf.float64),
                   'E': tf.constant(value=7.1 * 10 ** 10, shape=(1, 1), dtype=tf.float64),
                   # 'E': tf.constant(value=1.0, shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'w': tf.constant(1000.0, shape=(1, 1), dtype=tf.float64),
                   # 'w': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   '0': tf.constant(0.0, shape=(1, 1), dtype=tf.float64)},
        data={'collocation': {'x': np.random.uniform(low=0., high=1.0, size=1000)},
              'test': {'x': np.linspace(start=0.0, stop=1.0, num=101, dtype=np.float64)}},
        layers=[15, 30, 60, 30, 15, 1],
        activation_function='tanh',
        guess=0)

    def collocation(self, model, data):
        value = self.d(wrt={'x': 4}, model=model, data=data) + \
                self.v['c']['w'] / self.v['c']['E'] / self.v['c']['I']
        return value

    def boundary(self, model):
        y_0 = model(self.v['c']['0'])
        y_L = model(self.v['c']['L'])
        d2y_dx2_0 = self.d(wrt={'x': 2}, model=model, data=self.v['c']['0'])
        d2y_dx2_L = self.d(wrt={'x': 2}, model=model, data=self.v['c']['L'])
        return y_0, y_L, d2y_dx2_0, d2y_dx2_L

    def equation(self, data):
        return (self.v['c']['w'] / self.v['c']['E'] / self.v['c']['I'] / 24.0 *
                (-data ** 4 + 2.0 * self.v['c']['L'] * data ** 3 - data * self.v['c']['L'] ** 3))

    def loss_function(self, model, train_data):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'])))
        loss += tf.reduce_sum(tf.square(self.boundary(model)))
        return loss

    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function = types.MethodType(loss_function, pinn)
    pinn.train_network(epochs=1000000,
                       test_error=10 ** -2,
                       plot_x='x')
