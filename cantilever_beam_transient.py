from main import PINN
import tensorflow as tf
import numpy as np
import types

if __name__ == "__main__":
    solution = np.array(
        [0.0, 1.99328638928e-07, 7.82537085797e-07, 1.7293456267e-06, 3.01947488879e-06,
         4.63264723294e-06, 6.54858649796e-06, 8.74702072906e-06, 1.12076813821e-05, 1.39103076435e-05,
         1.68346450664e-05, 1.99604473892e-05, 2.32674847211e-05, 2.67355353571e-05, 3.03443976009e-05,
         3.40738806699e-05, 3.79038210667e-05, 4.18140771217e-05, 4.5784519898e-05, 4.97950641147e-05,
         5.38256463187e-05],
        dtype=np.float64)
    pinn = PINN(  # constants declared here so that they are not constructed each time in functions
        constants={'rho': tf.constant(value=2770.0, shape=(1, 1), dtype=tf.float64),
                   'E': tf.constant(value=7.1 * 10 ** 10, shape=(1, 1), dtype=tf.float64),
                   'I': tf.constant(value=np.pi * 0.01 ** 4 / 4., shape=(1, 1), dtype=tf.float64),
                   'A': tf.constant(value=np.pi * 10 ** -4., shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(value=2.0, shape=(1, 1), dtype=tf.float64),
                   '0': tf.constant(value=0.0, shape=(1, 1), dtype=tf.float64),
                   'solution': tf.expand_dims(tf.convert_to_tensor(solution, dtype=tf.float64), axis=1)},
        data={'collocation': {'x': np.random.uniform(low=0.0, high=2.0, size=201),  # datasets for each training method
                              't': np.array([1.00288032714], dtype=np.float64)},
              'boundary': {'t': np.array([1.00288032714], dtype=np.float64)},
              'test': {'x': np.linspace(start=0., stop=2.0, num=21, dtype=np.float64),
                       't': np.array([1.00288032714], dtype=np.float64)},
              'hybrid': {'x': np.linspace(start=0., stop=2.0, num=21, dtype=np.float64),
                         't': np.array([1.00288032714], dtype=np.float64)}},
        layers=[10, 15, 15, 10, 1],  # neuron per layers
        activation_function='tanh')


    def collocation(self, model, data):
        value = (self.v['c']['E'] * self.v['c']['I'] * self.d(wrt={'x': 4}, model=model, data=data) +
                 self.v['c']['rho'] * self.v['c']['A'] * self.d(wrt={'t': 2}, model=model, data=data))
        return value


    def boundary(self, model, data):
        # dy_dt_0_t = self.d(wrt={'t': 1},
        #                    model=model,
        #                    data=tf.stack([self.v['c']['0'], data], axis=1))
        # d2y_dt2_L_t = self.d(wrt={'t': 2},
        #                      model=model,
        #                      data=tf.stack([self.v['c']['0'], data], axis=1))
        y_0_t = model(tf.concat([self.v['c']['0'], data], axis=1))
        dy_dx_0_t = self.d(wrt={'x': 1},
                           model=model,
                           data=tf.concat([self.v['c']['0'], data], axis=1))
        d2y_dx2_L_t = self.d(wrt={'x': 2},
                             model=model,
                             data=tf.concat([self.v['c']['L'], data], axis=1))
        d3y_dx3_L_t = (self.d(wrt={'x': 3},
                              model=model,
                              data=tf.concat([self.v['c']['L'], data], axis=1)) * self.v['c']['E'] * self.v['c']['I'] -
                       0.1 * tf.math.sin(2.0 * np.pi * 10.0 * data))
        return y_0_t, dy_dx_0_t, d2y_dx2_L_t, d3y_dx3_L_t  # dy_dt_0_t, d2y_dt2_L_t


    def hybrid(self, model, data):
        return tf.reduce_sum(tf.square(model(data) - self.v['c']['solution']))


    def equation(self, model, data):
        return self.v['c']['solution'] - model(data)


    def loss_function(self, model, train_data):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'])))
        loss += tf.reduce_sum([tf.reduce_mean(tf.square(bc)) for bc in self.boundary(model, train_data['boundary'])])
        return loss


    def loss_function_batch(self, model, train_data, index):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'][index])))
        loss += tf.reduce_sum([tf.square(bc) for bc in self.boundary(model, train_data['boundary'][0])])
        loss += self.hybrid(model, train_data['hybrid'][0])
        return loss


    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.hybrid = types.MethodType(hybrid, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function = types.MethodType(loss_function, pinn)
    pinn.loss_function_batch = types.MethodType(loss_function_batch, pinn)
    pinn.train_network(epochs=10000,
                       batches={'collocation': 16,
                                'boundary': 1,
                                'hybrid': 21},
                       plot_x='x',
                       test_error=10 ** -7)
    input()
