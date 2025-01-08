from main import PINN
import tensorflow as tf
import numpy as np
import types

if __name__ == "__main__":
    alpha = 1.0
    pinn = PINN(
        constants={'0': tf.constant(value=0.0, shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(value=1.0, shape=(1, 1), dtype=tf.float64),
                   'alpha': tf.constant(value=alpha, shape=(1, 1), dtype=tf.float64)},
        data={'collocation': {'x': np.random.uniform(low=0., high=1.0, size=50),
                              't': np.random.uniform(low=0., high=1.0, size=40)},
              'initial': {'x': np.linspace(start=0., stop=1.0, num=51, dtype=np.float64)},
              'boundary': {'t': np.linspace(start=0., stop=2.0, num=41, dtype=np.float64)},
              'test': {'x': np.linspace(start=0., stop=1.0, num=11, dtype=np.float64),
                       't': np.array([0.1], dtype=np.float64)}},
        layers=[15, 30, 60, 30, 15, 1],
        activation_function='tanh',
        guess=0)

    def collocation(self, model, data):
        value = self.d(wrt={'t': 1}, model=model, data=data) - \
                self.v['c']['alpha'] * self.d(wrt={'x': 2}, model=model, data=data)
        return value

    def boundary(self, model, boundary, initial):
        shape_boundary = boundary.shape[0]
        shape_initial = initial.shape[0]
        T_0_t = tf.reduce_mean(model(tf.stack([tf.reshape(tf.repeat(self.v['c']['0'], shape_boundary), (shape_boundary, 1)), boundary], axis=1)))
        T_L_t = tf.reduce_mean(model(tf.stack([tf.reshape(tf.repeat(self.v['c']['L'], shape_boundary), (shape_boundary, 1)), boundary], axis=1)))
        T_x_0 = tf.reduce_mean(model(tf.stack([initial, tf.reshape(tf.repeat(self.v['c']['0'], shape_initial), (shape_initial, 1))], axis=1))) - 10
        return T_0_t, T_L_t, T_x_0

    def equation(self, data):
        return 40/np.pi * (tf.math.exp(-(np.pi) ** 2 * self.v['c']['alpha'] * tf.gather(data, 1, axis=1)) * tf.math.sin(np.pi * tf.gather(data, 0, axis=1)) +
                           1 / 3 * tf.math.exp(-(3 * np.pi) ** 2 * self.v['c']['alpha'] * tf.gather(data, 1, axis=1)) * tf.math.sin(3 * np.pi * tf.gather(data, 0, axis=1)) +
                           1 / 5 * tf.math.exp(-(5 * np.pi) ** 2 * self.v['c']['alpha'] * tf.gather(data, 1, axis=1)) * tf.math.sin(5 * np.pi * tf.gather(data, 0, axis=1)))

    def loss_function(self, model, train_data):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'])))
        loss += tf.reduce_sum(tf.square(self.boundary(model, train_data['boundary'], train_data['initial'], )))
        return loss

    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function = types.MethodType(loss_function, pinn)
    pinn.train_network(epochs=1000000,
                       test_error=10 ** -4,
                       plot_x='x')
