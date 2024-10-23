from main import PINN
import tensorflow as tf
import numpy as np
import types

if __name__ == "__main__":
    pinn = PINN(  # constants declared here so that they are not constructed each time in functions
        constants={'L': tf.constant(value=1.0, shape=(1, 1), dtype=tf.float64),
                   'R': tf.constant(value=1.0, shape=(1, 1), dtype=tf.float64),
                   'V_0': tf.constant(value=1.0, shape=(1, 1), dtype=tf.float64),
                   '0': tf.constant(value=0.0, shape=(1, 1), dtype=tf.float64)},
        data={'collocation': {'t': np.random.uniform(low=0.0, high=6.0, size=1001)},
              'test': {'t': np.linspace(start=0., stop=6.0, num=13)}},
        layers=[10, 15, 25, 15, 10, 1],  # neuron per layers
        activation_function='tanh')
    
    def collocation(self, model, data):
        value = self.v['c']['R'] * model(data) + self.v['c']['L'] * self.d(wrt={'t': 1}, model=model, data=data) - self.v['c']['V_0']
        return value

    def boundary(self, model, _):
        i_0 = model(self.v['c']['0'])
        #L_0 = self.v['c']['V_0'] - self.v['c']['L'] * self.d(wrt={'t': 1}, model=model, data=self.v['c']['0'])
        return i_0,# L_0

    def equation(self, model, data):
        return (self.v['c']['V_0'] / self.v['c']['R'] * (1 - tf.math.exp(-self.v['c']['R'] * data / self.v['c']['L'])) -
                model(data))

    def loss_function(self, model, train_data):
        _ = None
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'])))
        loss += tf.reduce_sum([tf.reduce_mean(tf.square(bc)) for bc in self.boundary(model, _)])
        return loss


    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function = types.MethodType(loss_function, pinn)
    pinn.train_network(epochs=5000,
                       error=10 ** -4,
                       plot_x='t')