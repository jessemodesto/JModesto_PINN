import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import HeNormal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")  # Comment out if PyQt is not installed


class NeuralNetwork:
    def __init__(self, dependents_test_domains, constants, boundary_conditions,
                 number_neurons_per_layer, activation_function,
                 number_train_sets):
        self.constants = constants  # constants to be used later
        self.constants_to_float64()  # convert constants to float64
        self.model = Sequential()  # define Sequential model so that layers can be added
        input_layer_size = len(dependents_test_domains)  # extract amount of dependent variables
        self.add_layers(input_layer_size=input_layer_size,  # add layers to the model
                        number_neurons_per_layer=number_neurons_per_layer,
                        activation_function=activation_function)
        self.training, self.output = self.training_init(input_layer_size=input_layer_size,  # set test data (float64)
                                                        dependents_test_domains=dependents_test_domains,
                                                        number_train_sets=number_train_sets)
        self.boundary_conditions = self.read_boundary_conditions(boundary_conditions, dependents_test_domains)
        self.initial_weights = None

    def constants_to_float64(self):
        for key in self.constants:
            self.constants[key] = np.float64(self.constants[key])
        return

    def add_layers(self, input_layer_size, number_neurons_per_layer, activation_function):
        self.model.add(Input(shape=(input_layer_size,), dtype='float64'))  # add input layer to model
        if len(number_neurons_per_layer) > 1:  # if at least one hidden layer
            for i in range(0, len(number_neurons_per_layer) - 2):
                self.model.add(Dense(units=number_neurons_per_layer[i],  # add hidden layers to model
                                     activation=activation_function,
                                     kernel_initializer=HeNormal(),
                                     dtype='float64'))
        self.model.add(Dense(units=number_neurons_per_layer[-1],  # add output layer to model
                             activation=None,
                             kernel_initializer=HeNormal(),
                             dtype='float64'))
        return

    def training_init(self, input_layer_size, dependents_test_domains, number_train_sets):
        training = np.zeros([number_train_sets, input_layer_size], dtype=np.float64)  # create input to train
        output = np.zeros([number_train_sets, input_layer_size], dtype=np.float64)  # create output of model
        for dependent_variable, j in zip(dependents_test_domains, range(input_layer_size)):
            lower_bound = dependents_test_domains[dependent_variable][0]
            lower_bound = self.constants[lower_bound] if isinstance(lower_bound, str) else np.float64(lower_bound)
            upper_bound = dependents_test_domains[dependent_variable][1]
            upper_bound = self.constants[upper_bound] if isinstance(upper_bound, str) else np.float64(upper_bound)
            for i in range(number_train_sets):
                training[i, j] = np.random.uniform(lower_bound, upper_bound)
        return training, output

    def train_network(self, optimizer=tf.keras.optimizers.Adam(), batch_size=1, epochs=1, plot=False):
        self.training = tf.data.Dataset.from_tensor_slices(self.training)
        self.training = self.training.shuffle(buffer_size=1024).batch(batch_size)
        number_of_batches = len(self.training)
        self.initial_weights = tf.keras.backend.get_value(self.model.trainable_variables)

        plot = DynamicPlot()
        x_test = np.sort(np.linspace(0, 1, 10).astype(np.float64), axis=0)
        y_test = np.array([self.analytic_solution(x) for x in x_test], dtype=np.float64)
        plot.plot(((x_test, y_test),
                   (x_test, np.zeros(len(x_test)))))
        pbar = tf.keras.utils.Progbar(target=number_of_batches)

        L = tf.constant(self.L, shape=(1, 1), dtype=tf.float64)
        zero = tf.constant(0., shape=(1, 1), dtype=tf.float64)
        for epoch in range(epochs):
            tf.print(f"Epoch: {epoch + 1}/{epochs}")
            step = 1
            for train_batch in self.training:
                with tf.GradientTape() as tape:
                    _, d2u_dx2 = self.train_loss(train_batch)
                    du_dx, _ = self.train_loss(L)
                    loss = tf.reduce_mean(tf.square(self.A * self.E * d2u_dx2 + self.c * train_batch)) + \
                           tf.square(self.model(zero)) + \
                           tf.square(du_dx)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                test_prediction = np.array([self.model(x.reshape(1, 1)).numpy()[0] for x in x_test], dtype=np.float64)
                test_loss = np.mean(np.square(test_prediction - y_test), dtype=np.float64)
                pbar.update(step, values=[('loss', loss), ('test loss', test_loss)])
                plot.update(y=test_prediction, plot_number=1)
                step += 1
        return

    def read_boundary_conditions(self, boundary_conditions, dependents_test_domains):
        to_evaluate = ()
        for i in boundary_conditions:
            if i['function_type'] == 'derivative':
                input_layer = i['input_values']
                value = None
                for j in input_layer:
                    if j not in dependents_test_domains.keys():
                        print('BC dependents mismatch!')
                        sys.exit(1)
                    value = input_layer[j]
                    if isinstance(value, (int, float, complex)):
                        value = tf.constant(value, shape=(1, 1), dtype=tf.float64)
                    elif isinstance(value, str):
                        value = tf.constant(boundary_conditions[value], shape=(1, 1), dtype=tf.float64)
                to_evaluate = to_evaluate + (Derivative(amount=i['amount'], input_layer=value), )
            elif i['function_type'] == 'model':
                pass
        return to_evaluate


class Derivative:
    def __init__(self, amount, input_layer):
        self.model = None
        self.amount = amount
        self.input_layer = input_layer

    def derivative(self, _):
        with tf.GradientTape() as tape:
            tape.watch(self.input_layer)
            if self.amount == 1:
                value = self.model(self.input_layer)
            else:
                self.amount -= 1
                value = self.derivative(None)
            output = tape.gradient(value, self.input_layer)
        return output


class DynamicPlot:
    def __init__(self):
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.data = {}
        plt.ion()
        plt.show()

    def plot(self, pairs):
        for i in range(len(pairs)):
            self.data[i] = self.ax.plot(pairs[i][0], pairs[i][1])
        return

    def update(self, x=None, y=None, plot_number=None):
        if x is not None:
            self.data[plot_number][0].set_xdata(x)
        if y is not None:
            self.data[plot_number][0].set_ydata(y)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        return


if __name__ == "__main__":
    nn = NeuralNetwork(dependents_test_domains={'x': (0.0, 'L')},
                       constants={'c': 1.0,
                                  'A': 1.0,
                                  'E': 1.0,
                                  'L': 1.0},
                       boundary_conditions=({'function_type': 'derivative',
                                             'amount': 1,
                                             'input_values': {'x': 0}
                                             },
                                            {'function_type': 'model',
                                             'input_values': {'x': 'L'}
                                             }),
                       number_neurons_per_layer=[30, 60, 30, 1],
                       number_train_sets=1000,
                       activation_function='tanh')
    nn.train_network(optimizer=tf.keras.optimizers.Adam(),
                     batch_size=16,
                     epochs=400)
    print('Breakpoint here: Try different test data points')
