from main import PINN
import tensorflow as tf
import numpy as np
import types

if __name__ == "__main__":
    pinn = PINN(
        constants={'c': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'A': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'E': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'zero': tf.constant(0.0, shape=(1, 1), dtype=tf.float64),
                   'length': tf.constant(1.0, shape=(1, 1), dtype=tf.float64)},
        data={'collocation': {'x': np.random.uniform(low=0., high=1.0, size=500)},
              'boundary': {'x': np.array([0], dtype=np.float64)},
              'test': {'x': np.linspace(start=0.0, stop=1.0, num=11, dtype=np.float64)}},
        layers=[15, 25, 15, 1],  # neuron per layers
        activation_function='tanh')  # activation function

    def collocation(self, model, data):
        value = self.v['c']['A'] * self.v['c']['E'] * self.d(wrt={'x': 2}, model=model, data=data) + self.v['c']['c'] * data
        return value

    def boundary(self, model, _):
        fix = model(self.v['c']['zero'])
        end = (self.d(wrt={'x': 1},
                      model=model,
                      data=self.v['c']['L']) - self.v['c']['c'] / 2 / self.v['c']['A'] / self.v['c']['E'] * self.v['c']['L'] ** 2)
        return fix, end

    def equation(self, model, data):
        return (self.v['c']['c'] / self.v['c']['A'] / self.v['c']['E'] *
                (-data ** 3 / 6 + (data * self.v['c']['L'] ** 2))) - model(data)

    # TODO: verify this equation is correct
    def loss_function_batch(self, model, train_data, index):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'][index])))
        loss += tf.reduce_sum([tf.reduce_mean(tf.square(bc)) for bc in
                               self.boundary(model, train_data['boundary'][0])])
        return loss

    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function_batch = types.MethodType(loss_function_batch, pinn)
    pinn.train_network(epochs=5000,
                       batches={'collocation': 16,
                                'boundary': 1},
                       test_error=10 ** -4,
                       plot_x='x')
