from main import PINN
import tensorflow as tf
import numpy as np
import types

if __name__ == "__main__":
    pinn = PINN(
        constants={'I': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'E': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'w': tf.constant(1.0, shape=(1, 1), dtype=tf.float64),
                   'zero': tf.constant(0.0, shape=(1, 1), dtype=tf.float64),
                   'length': tf.constant(1.0, shape=(1, 1), dtype=tf.float64)},
        data={'collocation': {'x': np.random.uniform(low=0., high=1.0, size=1000)},
              'boundary': {'x': np.array([0], dtype=np.float64)},
              'test': {'x': np.linspace(start=0.0, stop=1.0, num=11, dtype=np.float64)}},
        layers=[15, 25, 15, 1],  # neuron per layers
        activation_function='tanh')  # activation function


    def collocation(self, model, data):
        value = (self.v['c']['w'] / self.v['c']['E'] / self.v['c']['I'] +
                 self.d(wrt={'x': 4},
                        model=model,
                        data=data))
        return value


    def boundary(self, model, _):
        y_0 = model(self.v['c']['zero'])
        dy_dx_0 = self.d(wrt={'x': 1},
                         model=model,
                         data=self.v['c']['zero'])
        d2y_dx2_L = self.d(wrt={'x': 2},
                           model=model,
                           data=self.v['c']['L'])
        d3y_dx3_L = self.d(wrt={'x': 3},
                           model=model,
                           data=self.v['c']['L'])
        return y_0, dy_dx_0, d2y_dx2_L, d3y_dx3_L


    def equation(self, model, data):
        return (self.v['c']['w'] / self.v['c']['E'] / self.v['c']['I'] / 24.0 *
                (-data ** 4 + 4.0 * self.v['c']['L'] * data ** 3 - 6.0 * data ** 2 * self.v['c']['L'] ** 2)) - model(data)


    def loss_function_batch(self, model, train_data, index):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'][index])))
        loss += tf.reduce_sum([tf.reduce_mean(tf.square(bc)) for bc in self.boundary(model, train_data['boundary'][0])])
        return loss


    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function_batch = types.MethodType(loss_function_batch, pinn)
    pinn.train_network(epochs=10000,
                       batches={'collocation': 16,
                                'boundary': 1},
                       plot_x='x',
                       test_error=10 ** -7)
