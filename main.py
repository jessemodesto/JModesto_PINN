import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import HeNormal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("QtAgg")


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
    def __init__(self, number_neurons_per_layer, number_train_sets, activation_function,
                 cross_section_area, youngs_modulus, constant, length):
        self.A = cross_section_area
        self.E = youngs_modulus
        self.c = constant
        self.L = length
        self.number_neurons_per_layer = number_neurons_per_layer
        self.model = Sequential()
        self.add_layers(activation_function=activation_function)
        self.training = np.array([])
        self.output = np.array([])
        self.training_init(number_train_sets)
        self.initial_weights = None

    def add_layers(self, activation_function):
        self.model.add(Input(shape=(self.number_neurons_per_layer[0],), dtype='float64'))
        if len(self.number_neurons_per_layer) > 2:
            for i in range(1, len(self.number_neurons_per_layer) - 1):
                self.model.add(Dense(units=self.number_neurons_per_layer[i],
                                     activation=activation_function,
                                     kernel_initializer=HeNormal(),
                                     dtype='float64'))
        self.model.add(Dense(units=self.number_neurons_per_layer[-1],
                             activation=None,
                             kernel_initializer=HeNormal(),
                             dtype='float64'))
        return

    def training_init(self, number_train_sets):
        self.training = np.zeros([number_train_sets, self.number_neurons_per_layer[0]], dtype=np.float64)
        self.output = np.zeros([number_train_sets, self.number_neurons_per_layer[0]], dtype=np.float64)
        for i in range(number_train_sets):
            vals = self.training_data()
            self.training[i, :] = vals
        return

    def training_data(self):
        x_train = self.L * np.random.rand(self.number_neurons_per_layer[0]).astype(np.float64)
        x_train = np.sort(x_train, axis=0)
        return x_train

    def train_loss(self, x):
        with tf.GradientTape() as t2:  # Tape for second derivative
            t2.watch(x)
            with tf.GradientTape() as t1:  # Tape for first derivative
                t1.watch(x)
                u = self.model(x)  # Forward pass to get u(x)
            du_dx = t1.gradient(u, x)
        d2u_dx2 = t2.gradient(du_dx, x)
        return du_dx, d2u_dx2

    def train_network(self, optimizer=None, batch_size=1, epochs=1):
        self.training = tf.data.Dataset.from_tensor_slices(self.training)
        self.training = self.training.shuffle(buffer_size=1024).batch(batch_size)
        number_of_batches = len(self.training)
        self.initial_weights = tf.keras.backend.get_value(self.model.trainable_variables)

        x_test = np.linspace(0, 1, 10).astype(np.float64)
        x_test = np.sort(x_test, axis=0)

        figure, ax = plt.subplots(figsize=(10, 8))
        line1, = ax.plot(x_test, [self.analytic_solution(x) for x in x_test])
        line2, = ax.plot(x_test, [0 for _ in x_test])
        ax.set_ylim(-2, 2)
        plt.ion()
        plt.show()

        L = tf.constant(self.L, shape=(1, 1), dtype=tf.float64)
        zero = tf.constant(0., shape=(1, 1), dtype=tf.float64)
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}")
            step = 0
            for train_batch in self.training:
                print(f"Step: {step + 1}/{number_of_batches}")
                with tf.GradientTape() as tape:
                    _, d2u_dx2 = self.train_loss(train_batch)
                    du_dx, _ = self.train_loss(L)
                    loss = tf.reduce_mean(tf.square(self.A * self.E * d2u_dx2 + self.c * self.model(train_batch))) + \
                           tf.square(self.model(zero)) + \
                           tf.square(du_dx)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                print(f"Loss: {loss.numpy()[0][0]}")

                line2.set_ydata([self.model(x.reshape(1, 1)).numpy()[0] for x in x_test])
                figure.canvas.draw()
                figure.canvas.flush_events()
                step += 1
        return

    def analytic_solution(self, x):
        return self.c / 6.0 / self.A / self.E * (-(x ** 3) + (3 * (self.L ** 2) * x))


if __name__ == "__main__":
    nn = NeuralNetwork(number_neurons_per_layer=[1, 30, 60, 30, 1],
                       number_train_sets=1000,
                       activation_function='tanh',
                       cross_section_area=1.0,
                       youngs_modulus=1.0,
                       constant=1.0,
                       length=1.0)
    nn.train_network(optimizer=tf.keras.optimizers.Adam(),
                     batch_size=16,
                     epochs=100)
    print('Breakpoint here: Try different test data points')
    print('more')
