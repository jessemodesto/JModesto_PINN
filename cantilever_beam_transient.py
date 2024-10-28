from main import PINN
import tensorflow as tf
import numpy as np
import types

if __name__ == "__main__":
    solution = np.array(
        [0.0, 1.58114971782e-06, 6.21499566478e-06, 1.37408696901e-05, 2.3998103643e-05,
         3.68260298274e-05, 5.20639805472e-05, 6.95512862876e-05, 8.91272720764e-05, 0.00011063128477,
         0.000133902649395, 0.000158780690981, 0.000185104741831, 0.000212714148802, 0.000241448229644,
         0.000271146302111, 0.000301647756714, 0.000332791852998, 0.000364417966921, 0.000396365387132,
         0.000428473518696],
        dtype=np.float64)
    pinn = PINN(  # constants declared here so that they are not constructed each time in functions
        constants={'rho': tf.constant(value=2770.0, shape=(1, 1), dtype=tf.float64),
                   'E': tf.constant(value=7.1 * 10 ** 10, shape=(1, 1), dtype=tf.float64),
                   'I': tf.constant(value=np.pi * 0.01 ** 4 / 4., shape=(1, 1), dtype=tf.float64),
                   'A': tf.constant(value=np.pi * 10 ** -4., shape=(1, 1), dtype=tf.float64),
                   'L': tf.constant(value=2.0, shape=(1, 1), dtype=tf.float64),
                   '0': tf.constant(value=0.0, shape=(1, 1), dtype=tf.float64),
                   'solution': tf.expand_dims(tf.convert_to_tensor(solution, dtype=tf.float64), axis=0)},
        data={'collocation': {'x': np.random.uniform(low=0.0, high=2.0, size=201),  # datasets for each training method
                              't': np.array([1.00890625], dtype=np.float64)},
              'boundary': {'t': np.array([1.00890625], dtype=np.float64)},
              'test': {'x': np.linspace(start=0., stop=2.0, num=21, dtype=np.float64),
                       't': np.array([1.00890625], dtype=np.float64)}},
        layers=[10, 15, 15, 10, 1],  # neuron per layers
        activation_function='tanh')


    def collocation(self, model, data):
        value = (self.v['c']['E'] * self.v['c']['I'] * self.d(wrt={'x': 4}, model=model, data=data) +
                 self.v['c']['rho'] * self.v['c']['A'] * self.d(wrt={'t': 2}, model=model, data=data))
        return value


    def boundary(self, model, data):
        dy_dt_0_t = self.d(wrt={'t': 1},
                           model=model,
                           data=tf.stack([self.v['c']['0'], data], axis=1))
        d2y_dt2_L_t = self.d(wrt={'t': 2},
                             model=model,
                             data=tf.stack([self.v['c']['0'], data], axis=1))
        y_0_t = model(tf.stack([self.v['c']['0'], data], axis=1))
        dy_dx_0_t = self.d(wrt={'x': 1},
                           model=model,
                           data=tf.stack([self.v['c']['0'], data], axis=1))
        d2y_dx2_L_t = self.d(wrt={'x': 2},
                             model=model,
                             data=tf.stack([self.v['c']['L'], data], axis=1))
        d3y_dx3_L_t = (self.d(wrt={'x': 3},
                              model=model,
                              data=tf.stack([self.v['c']['L'], data], axis=1)) * self.v['c']['E'] * self.v['c']['I'] -
                       0.1 * tf.math.sin(2.0 * np.pi * 10.0 * data))
        return y_0_t, dy_dx_0_t, d2y_dx2_L_t, d3y_dx3_L_t, dy_dt_0_t, d2y_dt2_L_t


    def equation(self, model, data):
        return self.v['c']['solution'] - model(data)


    def loss_function(self, model, train_data):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'])))
        loss += tf.reduce_sum([tf.reduce_mean(tf.square(bc)) for bc in self.boundary(model, train_data['boundary'])])
        return loss


    def loss_function_batch(self, model, train_data, index):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'][index])))
        loss += tf.reduce_sum([tf.reduce_mean(tf.square(bc)) for bc in self.boundary(model, train_data['boundary'][0])])
        return loss


    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function = types.MethodType(loss_function, pinn)
    pinn.loss_function_batch = types.MethodType(loss_function_batch, pinn)
    pinn.train_network(epochs=10000,
                       batches={'collocation': 16,
                                'boundary': 1},
                       plot_x='x',
                       test_error=10 ** -7)
