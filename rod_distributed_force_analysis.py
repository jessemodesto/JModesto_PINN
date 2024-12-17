import itertools
import time
from main import PINN
import tensorflow as tf
import numpy as np
import types
import csv

collocation_points = [40, 50, 60, 100, 150, 200, 300, 400, 500, 1000]
layer_analysis = [i * [10] for i in range(1, 16)]
neuron_analysis = [[i] for i in range(5, 151, 5)]
batch_size = [4, 8, 12, 16, 20, 24, 28, 32]
activation_function_analysis = ['tanh', 'sigmoid', 'softplus']

neuron_set = itertools.product(collocation_points, neuron_analysis, batch_size, activation_function_analysis)
layer_set = itertools.product(collocation_points, neuron_analysis, batch_size, activation_function_analysis)

with open('neuron_analysis.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(
        [['collocation_pts', 'layers', 'batch_size', 'activation_func', 'time', 'epochs', 'test_error', 'loss']])

for collocations, neuron, batch, activation in neuron_set:
    start = time.time()
    pinn = PINN(
        constants={'c': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'A': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'E': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'zero': tf.constant(0.0, shape=(1, 1), dtype=tf.float64),
                   'length': tf.constant(1.0, shape=(1, 1), dtype=tf.float64)},
        data={'collocation': {'x': np.random.uniform(low=0., high=1.0, size=collocations)},
              'boundary': {'x': np.array([0], dtype=np.float64)},
              'test': {'x': np.linspace(start=0.0, stop=1.0, num=11, dtype=np.float64)}},
        layers=neuron + [1],  # neuron per layers
        activation_function=activation)


    def collocation(self, model, data):
        value = self.v['c']['A'] * self.v['c']['E'] * self.d(wrt={'x': 2}, model=model, data=data) + self.v['c'][
            'c'] * data
        return value


    def boundary(self, model, _):
        fix = model(self.v['c']['zero'])
        end = self.d(wrt={'x': 1}, model=model, data=self.v['c']['L'])
        return fix, end


    def equation(self, model, data):
        return (self.v['c']['c'] / self.v['c']['A'] / self.v['c']['E'] / 6 *
                (-data ** 3 + (3 * data * self.v['c']['L'] ** 2))) - model(data)


    def loss_function_batch(self, model, train_data, index):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'][index])))
        loss += tf.reduce_sum([tf.reduce_mean(tf.square(bc)) for bc in self.boundary(model, train_data['boundary'][0])])
        return loss


    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function_batch = types.MethodType(loss_function_batch, pinn)
    epoch, error, loss = pinn.train_network(epochs=100000,
                                            batches={'collocation': batch,
                                                     'boundary': 1},
                                            test_error=0.01,
                                            plot_x='x')
    elapsed = time.time() - start
    with open('neuron_analysis.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[collocations, neuron, batch, activation, elapsed, epoch, error, loss]])
    print()
