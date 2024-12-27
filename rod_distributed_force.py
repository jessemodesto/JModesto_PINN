from main import PINN
import tensorflow as tf
import numpy as np
import types
import csv

if __name__ == "__main__":
    pinn = PINN(
        constants={'c': tf.constant(1000.0, shape=(1, 1), dtype=tf.float64),
                   'A': tf.constant(value=np.pi * 10 ** -4., shape=(1, 1), dtype=tf.float64),
                   'E': tf.constant(value=7.1 * 10 ** 10, shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'zero': tf.constant(0.0, shape=(1, 1), dtype=tf.float64),
                   'length': tf.constant(1.0, shape=(1, 1), dtype=tf.float64)},
        data={'collocation': {'x': np.random.uniform(low=0., high=1.0, size=501)},
              'boundary': {'x': np.array([0], dtype=np.float64)},
              'test': {'x': np.linspace(start=0.0, stop=1.0, num=101, dtype=np.float64)}},
        layers=[5, 10, 15, 10, 5, 1],  # neuron per layers
        activation_function='tanh')  # activation function


    def collocation(self, model, data):
        value = self.v['c']['A'] * self.v['c']['E'] * self.d(wrt={'x': 2}, model=model, data=data) + self.v['c']['c'] * data

        return value


    def boundary(self, model, _):
        fix = model(self.v['c']['zero'])
        end = self.d(wrt={'x': 1}, model=model, data=self.v['c']['L'])
        return fix, end


    def equation(self, data):
        return (self.v['c']['c'] / self.v['c']['A'] / self.v['c']['E'] / 6 *
                (-data ** 3 + (3 * data * self.v['c']['L'] ** 2)))


    def loss_function_batch(self, model, train_data, index):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'][index])))
        with open('collocation.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[loss.numpy()]])
        bcs = [tf.square(bc) for bc in self.boundary(model, train_data['boundary'][0])]
        loss += tf.reduce_sum(bcs)
        with open('bcs.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[bc.numpy()[0][0] for bc in bcs]])
        return loss


    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function_batch = types.MethodType(loss_function_batch, pinn)
    print(pinn.train_network(epochs=10000,
                             batches={'collocation': 32,
                                      'boundary': 1},
                             test_error=10 ** -7,
                             plot_x='x'))
