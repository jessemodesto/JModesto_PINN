import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import types
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
matplotlib.use("QtAgg")  # Comment out if PyQt is not installed


def derivative(amount: int, model, input_layer):
    with tf.GradientTape() as tape:
        tape.watch(input_layer)
        if amount == 1:
            value = model(input_layer)
        else:
            amount -= 1
            value = derivative(amount=amount,
                               model=model,
                               input_layer=input_layer)
        output = tape.gradient(value, input_layer)
    return output


class GoverningEquation:
    def __init__(self):
        self.parameters = {'constant': {},
                           'dependent': {}}

    def read_constant(self, constant):
        for constant_variable in constant:
            self.parameters['constant'][constant_variable] = np.float64(constant[constant_variable])
        return

    def read_dependent_domain(self, dependent):
        for dependent_variable in dependent:
            lower_bound = dependent[dependent_variable][0]
            lower_bound = self.parameters['constant'][lower_bound] \
                if isinstance(lower_bound, str) else np.float64(lower_bound)
            upper_bound = dependent[dependent_variable][1]
            upper_bound = self.parameters['constant'][upper_bound] \
                if isinstance(upper_bound, str) else np.float64(upper_bound)
            self.parameters['dependent'][dependent_variable] = (lower_bound, upper_bound)
        return


class NeuralNetwork:
    def __init__(self):
        self.governing = None
        self.model = None
        self.training = None
        self.output = None

    def set_governing_equation(self, governing_equation):
        self.governing = governing_equation
        return

    def set_layers(self, number_neurons_per_layer, activation_function):
        self.model = Sequential()
        input_layer_size = len(self.governing.parameters['dependent'])
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

    def set_training(self, number_train_sets):
        input_layer_size = len(self.governing.parameters['dependent'])
        self.training = np.zeros([number_train_sets, input_layer_size], dtype=np.float64)  # create input to train
        self.output = np.zeros([number_train_sets, input_layer_size], dtype=np.float64)  # create output of model
        for i, dependent_variable in enumerate(self.governing.parameters['dependent']):
            for j in range(number_train_sets):
                self.training[j, i] = np.random.uniform(self.governing.parameters['dependent'][dependent_variable][0],
                                                        self.governing.parameters['dependent'][dependent_variable][1])
        return

    def train_network(self, batch_size: int, epochs: int, optimizer=Adam(), plot: dict = False, x_axis_plot: str = False):
        self.training = tf.data.Dataset.from_tensor_slices(self.training)
        self.training = self.training.shuffle(buffer_size=1024).batch(batch_size)
        number_of_batches = len(self.training)
        progress_bar = Progbar(target=number_of_batches)

        boundary = False
        if 'boundary' in self.governing.__dict__:
            boundary = True
        collocation = False
        if 'collocation' in self.governing.__dict__:
            collocation = True

        if plot is not False and x_axis_plot is not False:
            dynamic_plot = DynamicPlot()
            values = np.array([v for v in zip(*plot.values())], dtype=np.float64)
            values_dict = [dict(zip(plot, v)) for v in zip(*plot.values())]
            output = np.array([self.governing.equation(val) for val in values_dict], dtype=np.float64)
            dynamic_plot.plot(
                (
                    (plot[x_axis_plot], output),
                    (plot[x_axis_plot], np.zeros(len(output)))
                ))

        for epoch in range(epochs):
            tf.print(f"Epoch: {epoch + 1}/{epochs}")
            step = 0
            for train_batch in self.training:
                step += 1
                # TODO: implement hybrid approach
                with tf.GradientTape() as tape:
                    loss = None
                    if loss is None and collocation:
                        loss = tf.reduce_mean(tf.square(self.governing.collocation(self.model, train_batch)))
                    if loss is None and boundary:
                        loss = tf.reduce_sum([tf.square(bound) for bound in self.governing.boundary(self.model, train_batch)])
                    elif boundary:
                        loss = loss + tf.reduce_sum([tf.square(bound) for bound in self.governing.boundary(self.model, train_batch)])
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                if plot is not False and x_axis_plot is not False:
                    test_prediction = np.array(self.model(values), dtype=np.float64).reshape(-1)
                    dynamic_plot.update(y=test_prediction, plot_number=1)
                    test_loss = np.mean(np.square(test_prediction - output), dtype=np.float64)
                    progress_bar.update(step, values=[('loss', loss), ('test loss', test_loss)])
                else:
                    progress_bar.update(step, values=[('loss', loss)])
        return


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
    def rod_1():
        rod_example_1 = GoverningEquation()
        rod_example_1.read_constant(constant={'c': 1.0,
                                              'A': 1.0,
                                              'E': 1.0,
                                              'L': 1.0})
        rod_example_1.read_dependent_domain(dependent={'x': (0.0, 'L')})
        nn = NeuralNetwork()
        nn.set_governing_equation(governing_equation=rod_example_1)
        nn.set_layers(number_neurons_per_layer=[10, 15, 10, 1],
                      activation_function='tanh')
        nn.set_training(number_train_sets=1000)

        def collocation(self, model, input_layer):
            value = (self.parameters['constant']['A'] *
                     self.parameters['constant']['E'] *
                     derivative(amount=2, model=model, input_layer=input_layer) +
                     self.parameters['constant']['c'] * input_layer)
            return value

        def boundary(self, model, input_layer):
            zero = tf.constant(0., shape=(1, 1), dtype=tf.float64)
            length = tf.constant(self.parameters['constant']['L'], shape=(1, 1), dtype=tf.float64)
            fix = model(zero)
            end = derivative(amount=1, model=model, input_layer=length)
            return fix, end

        def equation(self, input_dictionary):
            return self.parameters['constant']['c'] / 6.0 / self.parameters['constant']['A'] / self.parameters['constant']['E'] * (-(input_dictionary['x'] ** 3) + (3 * (self.parameters['constant']['L'] ** 2) * input_dictionary['x']))

        rod_example_1.collocation = types.MethodType(collocation, rod_example_1)
        rod_example_1.boundary = types.MethodType(boundary, rod_example_1)
        rod_example_1.equation = types.MethodType(equation, rod_example_1)
        nn.train_network(batch_size=16,
                         epochs=500,
                         plot={'x': np.linspace(0, 1, 10).astype(np.float64)},
                         x_axis_plot='x')


    def rod_2():
        rod_example_2 = GoverningEquation()
        rod_example_2.read_constant(constant={'c': 1.0,
                                              'A': 1.0,
                                              'E': 1.0,
                                              'L': 1.0})
        rod_example_2.read_dependent_domain(dependent={'x': (0.0, 'L')})
        nn = NeuralNetwork()
        nn.set_governing_equation(governing_equation=rod_example_2)
        nn.set_layers(number_neurons_per_layer=[10, 15, 10, 1],
                      activation_function='tanh')
        nn.set_training(number_train_sets=1000)

        def collocation(self, model, input_layer):
            value = (self.parameters['constant']['A'] *
                     self.parameters['constant']['E'] *
                     derivative(amount=2, model=model, input_layer=input_layer) +
                     self.parameters['constant']['c'] * input_layer)
            return value

        def boundary(self, model, input_layer):
            zero = tf.constant(0., shape=(1, 1), dtype=tf.float64)
            length = tf.constant(self.parameters['constant']['L'], shape=(1, 1), dtype=tf.float64)
            fix = model(zero)
            end = derivative(amount=1, model=model, input_layer=length) - self.parameters['constant']['c'] / 2 / self.parameters['constant']['A'] / self.parameters['constant']['E'] * self.parameters['constant']['L'] ** 2
            return fix, end

        def equation(self, input_dictionary):
            return self.parameters['constant']['c'] / self.parameters['constant']['A'] / self.parameters['constant']['E'] * (-(input_dictionary['x'] ** 3) / 6 + (self.parameters['constant']['L'] ** 2) * input_dictionary['x'])

        rod_example_2.collocation = types.MethodType(collocation, rod_example_2)
        rod_example_2.boundary = types.MethodType(boundary, rod_example_2)
        rod_example_2.equation = types.MethodType(equation, rod_example_2)
        nn.train_network(batch_size=16,
                         epochs=500,
                         plot={'x': np.linspace(0, 1, 10).astype(np.float64)},
                         x_axis_plot='x')

    def heat_1d():
        heat_1d_example = GoverningEquation()
        heat_1d_example.read_constant(constant={'alpha': 0.001})
        heat_1d_example.read_dependent_domain(dependent={'x': (0.0, 1.0),
                                                         't': (0.0, 1.0)})
        nn = NeuralNetwork()
        nn.set_governing_equation(governing_equation=heat_1d_example)
        nn.set_layers(number_neurons_per_layer=[10, 15, 10, 1],
                      activation_function='tanh')
        nn.set_training(number_train_sets=1000)

        def collocation(self, model, input_layer):
            value = derivative(amount=1, model=model, input_layer=input_layer)[1] - self.parameters['constant']['alpha'] * derivative(amount=2, model=model, input_layer=input_layer)[0]
            return value

        def equation(self, input_dictionary):
            return np.sin(np.pi * input_dictionary['x']) * np.exp(-self.parameters['constant']['alpha'] * input_dictionary['t'])

        heat_1d_example.collocation = types.MethodType(collocation, heat_1d_example)
        heat_1d_example.equation = types.MethodType(equation, heat_1d_example)
        nn.train_network(batch_size=16,
                         epochs=500,
                         plot={'x': np.linspace(0, 1, 10).astype(np.float64),
                               't': np.linspace(0.9, 0.9, 10).astype(np.float64)},
                         x_axis_plot='x')
    heat_1d()