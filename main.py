import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import HeNormal
import numpy as np
import sys
import time


class Numerical:
    def __init__(self, total_number_elements, gauss_points, length,
                 modulus_elasticity, cross_section_area, c):
        # 1 Geometry #
        # Total number of elements
        n_el = total_number_elements
        # Number of nodes per element
        # Number of Gauss points
        n_ne = n_G = gauss_points
        # Total number of nodes
        n = n_ne + (n_el - 1) * (n_ne - 1)
        # Nodal coordinates array
        x = np.linspace(0, length, num=n)
        # Nodal connectivity table
        T_n = np.zeros([n_el, n_ne], dtype=int)
        for e in range(0, n_el):
            for i in range(0, n_ne):
                T_n[e, i] = int(e * (n_ne - 1) + i)

        # 2 Computation of the global stiffness matrix #
        # Initialize matrices
        B = np.zeros([1, n_ne, n_el, n_G])
        K_e = np.zeros([n_ne, n_ne, n_el])
        K = np.zeros([n, n])
        # Define gauss points values
        if n_G == 1:
            w_k = np.array([2.])
            zeta_k = np.array([0.])
        elif n_G == 2:
            w_k = np.array([1., 1.])
            zeta_k = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        elif n_G == 3:
            w_k = np.array([8 / 9, 5 / 9, 5 / 9])
            zeta_k = np.array([0, np.sqrt(3 / 5), -np.sqrt(3 / 5)])

        # 3 Computation of global force vector
        N = np.zeros([1, n_ne, n_el, n_G])
        x_G = np.zeros([n_G, n_el])
        f_e = np.zeros([n_ne, n_el])
        f = np.zeros([n, 1])

        for e in range(0, n_el):
            # 2 a) Get element's size
            l_e = x[T_n[e, n_ne - 1]] - x[T_n[e, 0]]
            # 3 a) Get nodal coordinates and element’s size
            x_e = x[T_n[e, :]]
            for k in range(0, n_G):
                if n_G == 2:
                    J_e = -0.5 * x[T_n[e, 0]] + 0.5 * x[T_n[e, 1]]
                elif n_G == 3:
                    J_e = ((zeta_k[k] - 0.5) * x[T_n[e, 0]] +
                           (-2 * zeta_k[k]) * x[T_n[e, 1]] +
                           (zeta_k[k] + 0.5) * x[T_n[e, 2]])
                # 2 b) Compute B matrix evaluated at 𝜉k
                # 3 b) Compute N matrix evaluated at 𝜉k
                for i in range(0, n_ne):
                    if n_G == 2:
                        B[0, i, e, k] = (1 / J_e) * np.array([-0.5,
                                                              0.5])[i]
                        N[0, i, e, k] = np.array([0.5 * (1 - zeta_k[k]),
                                                  0.5 * (1 + zeta_k[k])])[i]
                    elif n_G == 3:
                        B[0, i, e, k] = (1 / J_e) * np.array([zeta_k[k] - 0.5,
                                                              -2 * zeta_k[k],
                                                              zeta_k[k] + 0.5])[i]
                        N[0, i, e, k] = np.array([0.5 * zeta_k[k] * (zeta_k[k] - 1),
                                                  1 - ((zeta_k[k]) ** 2),
                                                  0.5 * zeta_k[k] * (zeta_k[k] + 1)])[i]
                # 2 c) Compute the element stiffness matrix evaluated at 𝜉k
                K_e[:, :, e] = (K_e[:, :, e] + w_k[k] * J_e * modulus_elasticity * cross_section_area *
                                np.transpose(B[:, :, e, k]) * B[:, :, e, k])
                # 3 c) Compute Gauss point coordinate and element force vector evaluated at 𝜉𝑘
                x_G[k, e] = N[:, :, e, k].dot(x_e)
                # 3 d) Evaluate body force at Gauss point
                b_k = c * x_G[k, e]
                # 3 e) Compute element force vector evaluated at 𝜉𝑘
                f_e[:, e][:, None] = (f_e[:, e][:, None] + w_k[k] * J_e *
                                      np.transpose(N[:, :, e, k]) * b_k)
            # 3 d) Assembly to global force matrix
            # 2 d) Assembly to global stiffness matrix
            for i in range(0, n_ne):
                f[T_n[e, i], 0] = f[T_n[e, i], 0] + f_e[i, e]
                for j in range(0, n_ne):
                    K[T_n[e, i], T_n[e, j]] = K[T_n[e, i], T_n[e, j]] + K_e[i, j, e]

        # 4 Global system of equations
        u = np.zeros([n])
        # 4 c) System resolution
        u[1:n][:, None] = np.matmul(np.linalg.inv(K[1:n, 1:n]),
                                    (f[1:n, 0] - K[1:n, 0] * u[0])[:, None])
        f_r = K[0, 0] * u[0] + K[0, 1:n] - f[0, 0]

        # 5 Computation of stress
        sigma_G = np.zeros([n_G, n_el])

        for e in range(0, n_el):
            # 5 a) Obtain element nodes displacement
            u_e = u[T_n[e, :]]
            # 5 b) Obtain stress at Gauss point
            for k in range(0, n_G):
                sigma_G[k, e] = modulus_elasticity * np.matmul(B[:, :, e, k], u_e)

        f = interpolate.interp1d(x_G.reshape(-1), sigma_G[:].reshape(-1), fill_value="extrapolate")
        sigma = f(x)
        pass

        # self.solution = np.array([self.analytic_solution(x) for x in x])
        # norm_value = 0
        # top = 0
        # bottom = 0
        # for e in range(0, n_el):
        #     for k in range(0, n_G - 1):
        #         l_k = x[T_n[e, k + 1]] - x[T_n[e, k]]
        #         top = top + (l_k * ((self.solution[T_n[e, k]] - u[T_n[e, k]]) ** 2)) ** 0.5
        #         bottom = bottom + (l_k * (self.solution[T_n[e, k]] ** 2)) ** 0.5
        # norm_value = top / bottom


class NeuralNetwork:
    def __init__(self,
                 input_layer_size, number_hidden_layers, number_neurons_per_layer, activation_function, optimizer,
                 cross_section_area, youngs_modulus, constant, length):
        self.A = cross_section_area
        self.E = youngs_modulus
        self.c = constant
        self.L = length
        self.model = Sequential()
        self.add_layers(input_layer_size=input_layer_size,
                        number_hidden_layers=number_hidden_layers,
                        number_neurons_per_layer=number_neurons_per_layer,
                        activation_function=activation_function)
        self.compile_model(optimizer=optimizer)
        self.training = np.array([])
        self.output = np.array([])
        self.track = 0

    def add_layers(self, input_layer_size, number_hidden_layers, number_neurons_per_layer, activation_function):
        if len(number_neurons_per_layer) != number_hidden_layers:
            print('Number of hidden layers and length of neurons per layer not the same!')
            sys.exit(1)
        self.model.add(Input(shape=(input_layer_size,)))
        if number_hidden_layers > 0:
            for i in range(number_hidden_layers):
                self.model.add(Dense(units=number_neurons_per_layer[i],
                                     activation=activation_function,
                                     kernel_initializer=HeNormal()))
        self.model.add(Dense(units=input_layer_size, activation=None, kernel_initializer=HeNormal()))
        return

    def compile_model(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=self.custom_mse, metrics=None)
        return

    def custom_mse(self, y_true_t, y_pred_t):
        if self.track > self.training.shape[0]:
            self.track = 0

        def compute_mse_for_sample(y_true, y_pred):
            input_layer = self.training[self.track, :][tf.newaxis, :]
            with tf.GradientTape() as g:
                g.watch(input_layer)
                with tf.GradientTape(persistent=True) as gg:
                    gg.watch(input_layer)
                    u_pred = self.model(input_layer)
                du_dx = gg.gradient(u_pred, input_layer)
                del gg
            d2u_dx2 = g.gradient(du_dx, input_layer)
            del g
            mse_sample = (
                    tf.reduce_mean(tf.square(d2u_dx2[0, 1:-1] + self.c / self.A / self.E * input_layer[0, 1:-1])) +
                    tf.square(u_pred[0, 0]) +
                    tf.square(du_dx[0, -1])
            )
            self.track += 1
            return mse_sample

        mse = tf.map_fn(lambda true_pred: compute_mse_for_sample(true_pred[0], true_pred[1]),
                        (y_true_t, y_pred_t),
                        dtype=tf.float32)
        mse = tf.reduce_mean(mse)
        return mse

    # def custom_metric(self, x, u):
    #     def compute_metric_for_sample(x_i, u_i):
    #         x_i = x_i[tf.newaxis, :]
    #         u_i = u_i[tf.newaxis, :]
    #         u_pred = self.model(x_i)
    #
    #         return u_pred - u_i
    #
    #     metric = tf.map_fn(lambda x_u: compute_metric_for_sample(x_u[0], x_u[1]), (x, u), dtype=tf.float32)
    #     return tf.reduce_mean(metric)

    def training_init(self, number_train_points, number_train_sets):
        if number_train_points < 2:
            print('Not enough training samples (2)!')
            sys.exit(1)
        self.training = np.zeros([number_train_sets, number_train_points])
        self.output = np.zeros([number_train_sets, number_train_points])
        for i in range(number_train_sets):
            vals = self.training_data(number_train_points=number_train_points)
            self.training[i, :] = vals
        self.training = tf.convert_to_tensor(self.training, dtype=tf.float32)
        self.output = tf.convert_to_tensor(self.output, dtype=tf.float32)
        return

    def training_data(self, number_train_points):
        x_train = self.L * np.random.rand(number_train_points - 2)  # Random collocation points in (0, L)
        x_train = np.sort(x_train, axis=0)  # Sort to maintain order for boundary conditions
        x_train = np.append(np.array([0]), x_train, axis=0)  # First point (boundary condition)
        x_train = np.append(x_train, np.array([self.L]), axis=0)  # Last point (boundary condition)
        return x_train

    def train_network(self, batch_size=32, epochs=100):
        start = time.time()
        self.model.fit(self.training, self.output, epochs=epochs, batch_size=batch_size, verbose=1)
        return time.time() - start

    def analytic_solution(self, x):
        return self.c / 6.0 / self.A / self.E * (-(x ** 3) + 3 * (self.L ** 3) * x)


if __name__ == "__main__":
    nn = NeuralNetwork(15,
                       3,
                       [30, 60, 30],
                       'sigmoid',  # relu always return 0 for second derivative
                       'adam',
                       1.0,
                       1.0,
                       1.0,
                       1.0)
    nn.training_init(number_train_points=15, number_train_sets=10000)
    print(nn.train_network(batch_size=1, epochs=1))

    x_new = np.linspace(0, 1.0, 15)
    x_new = np.reshape(x_new, (1, 15))
    u_pred = nn.model.predict(x_new)

    u_real = [nn.analytic_solution(x) for x in x_new]
