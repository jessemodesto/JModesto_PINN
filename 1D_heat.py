from main import PINN
import tensorflow as tf
import numpy as np
import types
import csv
import time

if __name__ == "__main__":
    start = time.time()
    T1 = 350.0
    T2 = 300.0
    pinn = PINN(  # constants declared here so that they are not constructed each time in functions
        constants={'alpha': tf.constant(value=20., shape=(1, 1), dtype=tf.float64),
                   'Q_t': tf.constant(value=10., shape=(1, 1), dtype=tf.float64),
                   'c_1': tf.constant(value=70.344995319758030, shape=(1, 1), dtype=tf.float64),
                   'c_2': tf.constant(value=2.796550046802420 * 10 ** 2, shape=(1, 1), dtype=tf.float64),
                   '0': tf.constant(value=0.0, shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(value=1.5, shape=(1, 1), dtype=tf.float64)},
        data={'collocation': {'x': np.random.uniform(low=0., high=1.5, size=1001)},
              'test': {'x': np.linspace(start=0., stop=1.5, num=151, dtype=np.float64)}},
        layers=[5, 10, 15, 10, 5, 1],  # neuron per layers
        activation_function='tanh')  # activation function

    def collocation(self, model, data):
        value = self.v['c']['alpha'] * self.d(wrt={'x': 2}, model=model, data=data) + \
                self.v['c']['Q_t'] * model(data)
        return value

    def boundary(self, model):
        T_0 = model(self.v['c']['0']) - T1
        T_L = model(self.v['c']['L']) - T2
        return T_0, T_L

    def equation(self, data):
        return - (self.v['c']['Q_t'] / self.v['c']['alpha']) * tf.math.square(data) + (T2 - T1 + (self.v['c']['Q_t'] / self.v['c']['alpha'])/2 * self.v['c']['L'] ** 2) / self.v['c']['L'] * data + T1

    def loss_function(self, model, train_data):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'])))
        with open('collocation.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[loss.numpy()]])
        bcs = tf.square(self.boundary(model))
        with open('bcs.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[bc.numpy()[0][0] for bc in bcs]])
        loss += tf.reduce_sum(bcs)
        return loss

    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function = types.MethodType(loss_function, pinn)
    print(pinn.train_network(epochs=1000000,
                             test_error=10 ** -7,
                             plot_x='x'))
    print(time.time() - start)
