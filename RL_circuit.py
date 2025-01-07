from main import PINN
import tensorflow as tf
import numpy as np
import types
import csv
import time

if __name__ == "__main__":
    start = time.time()
    pinn = PINN(
        constants={'L': tf.constant(value=1.0, shape=(1, 1), dtype=tf.float64),
                   'R': tf.constant(value=1.0, shape=(1, 1), dtype=tf.float64),
                   'V': tf.constant(value=1.0, shape=(1, 1), dtype=tf.float64),
                   '0': tf.constant(value=0.0, shape=(1, 1), dtype=tf.float64)},
        data={'collocation': {'t': np.random.uniform(low=0.0, high=6.0, size=1000)},
              'test': {'t': np.linspace(start=0., stop=6.0, num=101)}},
        layers=[15, 30, 60, 30, 15, 1],
        activation_function='tanh', #sigmoid, relu
        guess=0)

    def collocation(self, model, data):
        value = self.v['c']['R'] * model(data) + \
                self.v['c']['L'] * self.d(wrt={'t': 1}, model=model, data=data) - \
                self.v['c']['V']
        return value

    def boundary(self, model):
        i_0 = model(self.v['c']['0'])
        return i_0

    def equation(self, data):
        return self.v['c']['V'] / self.v['c']['R'] * (1 - tf.math.exp(-self.v['c']['R'] * data / self.v['c']['L']))

    def loss_function(self, model, train_data):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'])))
        with open('collocation.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[loss.numpy()]])
        bcs = tf.square(self.boundary(model))
        with open('bcs.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[bcs.numpy()[0][0]]])
        loss += bcs
        return loss

    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function = types.MethodType(loss_function, pinn)
    print(pinn.train_network(epochs=100000,
                             test_error=10 ** -2,
                             plot_x='t'))
    print(time.time() - start)
