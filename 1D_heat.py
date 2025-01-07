from main import PINN
import tensorflow as tf
import numpy as np
import types

if __name__ == "__main__":
    T_0 = 20
    T_L = 10
    pinn = PINN(
        constants={'0': tf.constant(value=0.0, shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(value=1.0, shape=(1, 1), dtype=tf.float64)},
        data={'collocation': {'x': np.random.uniform(low=0., high=1.0, size=1000)},
              'test': {'x': np.linspace(start=0., stop=1.0, num=101, dtype=np.float64)}},
        layers=[15, 30, 60, 30, 15, 1],
        activation_function='tanh',
        guess=20)

    def collocation(self, model, data):
        value = self.d(wrt={'x': 2}, model=model, data=data)
        return value

    def boundary(self, model):
        T_1 = model(self.v['c']['0']) - T_0
        T_2 = model(self.v['c']['L']) - T_L
        return T_1, T_2

    def equation(self, data):
        return (T_L - T_0) / self.v['c']['L'] * data + T_0

    def loss_function(self, model, train_data):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'])))
        loss += tf.reduce_sum(tf.square(self.boundary(model)))
        return loss

    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function = types.MethodType(loss_function, pinn)
    pinn.train_network(epochs=1000000,
                       test_error=10 ** -4,
                       plot_x='x')
