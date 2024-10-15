import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
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
import csv
import itertools

matplotlib.use("QtAgg")  # Comment out if PyQt is not installed


class GoverningEquation:
    def __init__(self):
        self.parameters = {'constant': {},
                           'dependent': {}}
        self.index = {}

    def read_constant(self, constant):
        for constant_variable in constant:
            if isinstance(constant[constant_variable], float) or isinstance(constant[constant_variable], int):
                self.parameters['constant'][constant_variable] = np.float64(constant[constant_variable])
            else:
                self.parameters['constant'][constant_variable] = constant[constant_variable]
        return

    def read_dependent_domain(self, dependent):
        for i, dependent_variable in enumerate(dependent):
            lower_bound = dependent[dependent_variable][0]
            lower_bound = self.parameters['constant'][lower_bound] \
                if isinstance(lower_bound, str) else np.float64(lower_bound)
            upper_bound = dependent[dependent_variable][1]
            upper_bound = self.parameters['constant'][upper_bound] \
                if isinstance(upper_bound, str) else np.float64(upper_bound)
            self.parameters['dependent'][dependent_variable] = (lower_bound, upper_bound)
            self.index[dependent_variable] = i
        return

    # TODO: implement partial derivatives with respect to multiple variables ex: d/(dxdy)
    def derivative(self, differential: dict, model, input_layer):
        max_order = sum(differential.values())
        if max_order == 1:
            with tf.GradientTape(persistent=False) as tape:
                tape.watch(input_layer)
                value = model(input_layer)
            value = tape.gradient(value, input_layer)
        else:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(input_layer)
                value = model(input_layer)
                for i in range(max_order):
                    value = tape.gradient(value, input_layer)
        if len(self.index) == 1:
            return value
        else:
            return value[(slice(None), self.index[next(iter(differential))])]


class NeuralNetwork:
    def __init__(self, governing):
        self.governing = governing
        self.model = None
        self.training = None
        self.output = None

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

    def set_testing(self, number_train_sets):
        input_layer_size = len(self.governing.parameters['dependent'])
        testing = np.zeros([number_train_sets, input_layer_size], dtype=np.float64)  # create input to test
        for i, dependent_variable in enumerate(self.governing.parameters['dependent']):
            for j in range(number_train_sets):
                testing[j, i] = np.random.uniform(self.governing.parameters['dependent'][dependent_variable][0],
                                                  self.governing.parameters['dependent'][dependent_variable][1])
        return testing

    def train_network(self, batch_size: int, epochs: int, optimizer=Adam(),
                      test_parameters: dict = None):
        self.training = tf.data.Dataset.from_tensor_slices(self.training)
        self.training = self.training.shuffle(buffer_size=1024).batch(batch_size)
        number_of_batches = len(self.training)
        progress_bar = Progbar(target=number_of_batches)
        loss_function = self.loss_function()

        if test_parameters is not None:
            test_error = False
            epoch_error = False
            testing = np.sort(self.set_testing(test_parameters['samples']), axis=0)
            if 'epoch_error' in test_parameters.keys():
                epoch_previous = 0
            if 'plot_x' in test_parameters.keys():
                testing_output = self.governing.equation(lambda x: 0, testing)
                dynamic_plot = DynamicPlot()
                if len(testing.shape) > 1:
                    dynamic_plot.plot(pairs=((testing[:, self.governing.index[test_parameters['plot_x']]], testing_output),
                                             (testing[:, self.governing.index[test_parameters['plot_x']]], np.zeros(len(testing_output)))))
                else:
                    dynamic_plot.plot(pairs=((testing, testing_output),
                                             (testing, np.zeros(len(testing_output)))))

        for epoch in range(1, epochs + 1):
            tf.print(f"Epoch: {epoch}/{epochs}")
            step = 0
            for train_batch in self.training:
                step += 1
                with tf.GradientTape() as tape:
                    loss = loss_function(self.model, train_batch)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                progress_bar.update(step, values=[('loss', loss), ])

            if test_parameters is not None:
                if 'test_error' in test_parameters.keys():
                    testing_difference = np.mean(np.abs(self.governing.equation(self.model, testing)))
                    tf.print(f"test_error: {testing_difference}")
                    if testing_difference < test_parameters['test_error']:
                        test_error = True
                if 'epoch_error' in test_parameters.keys():
                    # TODO: find out why np array needs to be converted to tensor
                    epoch_value = loss_function(self.model, tf.convert_to_tensor(testing))
                    epoch_difference = np.mean(np.abs(epoch_value - epoch_previous))
                    tf.print(f"epoch_difference: {epoch_difference}")
                    if epoch_difference < test_parameters['epoch_error']:
                        epoch_error = True
                    else:
                        epoch_previous = epoch_value
                if 'plot_x' in test_parameters.keys():
                    testing_output = np.array(self.model(tf.convert_to_tensor(testing)), dtype=np.float64).reshape(-1)
                    dynamic_plot.update(y=testing_output, plot_number=1)
                if any((test_error, epoch_error)):
                    loss = loss_function(self.model, tf.convert_to_tensor(testing))
                    loss = loss.numpy()
                    return epoch, loss
        return

    def loss_function(self, collocation: bool = True, boundary: bool = True, hybrid: bool = False):
        result = []
        if 'collocation' in self.governing.__dict__ and collocation:
            result.append(lambda model, train_batch:
                          tf.reduce_mean(tf.square(self.governing.collocation(model, train_batch))))
        if 'boundary' in self.governing.__dict__ and boundary:
            result.append(lambda model, train_batch:
                          tf.reduce_sum([tf.square(bc) for bc in self.governing.boundary(model, train_batch)]))
        if 'equation' in self.governing.__dict__ and hybrid:
            result.append(lambda model, train_batch:
                          tf.reduce_mean(tf.square(self.governing.equation(model, train_batch))))
        return lambda model, train_batch: sum(point(model, train_batch) for point in result)


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
    def rod_1(layers: list, training_sets: int, batch_size: int):
        rod_example_1 = GoverningEquation()
        rod_example_1.read_constant(constant={'c': 1.0,
                                              'A': 1.0,
                                              'E': 1.0,
                                              'L': 1.0,
                                              'zero': tf.constant(0., shape=(1, 1), dtype=tf.float64),
                                              'length': tf.constant(1.0, shape=(1, 1), dtype=tf.float64)})
        rod_example_1.read_dependent_domain(dependent={'x': (0.0, 'L')})
        nn = NeuralNetwork(governing=rod_example_1)
        nn.set_layers(number_neurons_per_layer=layers,
                      activation_function='tanh')
        nn.set_training(number_train_sets=training_sets)

        def collocation(self, model, input_layer):
            value = (self.parameters['constant']['A'] *
                     self.parameters['constant']['E'] *
                     self.derivative(differential={'x': 2},
                                     model=model,
                                     input_layer=input_layer) +
                     self.parameters['constant']['c'] * input_layer)
            return value

        def boundary(self, model, input_layer):
            fix = model(self.parameters['constant']['zero'])
            end = self.derivative(differential={'x': 1},
                                  model=model,
                                  input_layer=self.parameters['constant']['length'])
            return fix, end

        def equation(self, model, input_layer):
            return (self.parameters['constant']['c'] /
                    self.parameters['constant']['A'] /
                    self.parameters['constant']['E'] / 6 *
                    (-input_layer ** 3 +
                     (3 * input_layer * self.parameters['constant']['L'] ** 2))) - model(input_layer)

        rod_example_1.collocation = types.MethodType(collocation, rod_example_1)
        rod_example_1.boundary = types.MethodType(boundary, rod_example_1)
        rod_example_1.equation = types.MethodType(equation, rod_example_1)
        start_time = time.time()
        epoch, test_error = nn.train_network(batch_size=batch_size,
                                  epochs=50000,
                                  test_parameters={'samples': 100,
                                                   'test_error': 10 ** -4})
        return time.time()-start_time, epoch, test_error

    def rod_2():
        rod_example_2 = GoverningEquation()
        rod_example_2.read_constant(constant={'c': 1.0,
                                              'A': 1.0,
                                              'E': 1.0,
                                              'L': 1.0,
                                              'zero': tf.constant(0., shape=(1, 1), dtype=tf.float64),
                                              'length': tf.constant(1.0, shape=(1, 1), dtype=tf.float64)})
        rod_example_2.read_dependent_domain(dependent={'x': (0.0, 'L')})
        nn = NeuralNetwork(governing=rod_example_2)
        nn.set_layers(number_neurons_per_layer=[10, 15, 10, 1],
                      activation_function='tanh')
        nn.set_training(number_train_sets=1000)

        def collocation(self, model, input_layer):
            value = (self.parameters['constant']['A'] *
                     self.parameters['constant']['E'] *
                     self.derivative(differential={'x': 2},
                                     model=model,
                                     input_layer=input_layer) +
                     self.parameters['constant']['c'] * input_layer)
            return value

        def boundary(self, model, input_layer):
            fix = model(self.parameters['constant']['zero'])
            end = (self.derivative(differential={'x': 1},
                                   model=model,
                                   input_layer=self.parameters['constant']['length']) -
                   self.parameters['constant']['c'] / 2 / self.parameters['constant']['A'] / self.parameters['constant']['E'] * self.parameters['constant']['L'] ** 2)
            return fix, end

        def equation(self, model, input_layer):
            return (self.parameters['constant']['c'] /
                    self.parameters['constant']['A'] /
                    self.parameters['constant']['E'] *
                    (-input_layer ** 3 / 6 +
                     (input_layer * self.parameters['constant']['L'] ** 2))) - model(input_layer)

        rod_example_2.collocation = types.MethodType(collocation, rod_example_2)
        rod_example_2.boundary = types.MethodType(boundary, rod_example_2)
        rod_example_2.equation = types.MethodType(equation, rod_example_2)
        nn.train_network(batch_size=16,
                         epochs=500,
                         test_parameters={'samples': 100,
                                          'test_error': 10 ** -4,
                                          'epoch_error': 10 ** -4})

    def beam_1():
        beam_example_1 = GoverningEquation()
        beam_example_1.read_constant(constant={'E': 1.0,
                                               'I': 1.0,
                                               'L': 1.0,
                                               'w': 1.0,
                                               'zero': tf.constant(0., shape=(1, 1), dtype=tf.float64),
                                               'length': tf.constant(1.0, shape=(1, 1), dtype=tf.float64)})
        beam_example_1.read_dependent_domain(dependent={'x': (0.0, 'L')})
        nn = NeuralNetwork(governing=beam_example_1)
        nn.set_layers(number_neurons_per_layer=[40, 40, 1],
                      activation_function='tanh')
        nn.set_training(number_train_sets=1000)

        def collocation(self, model, input_layer):
            value = (self.parameters['constant']['w'] / self.parameters['constant']['E'] / self.parameters['constant']['I'] +
                     self.derivative(differential={'x': 4},
                                     model=model,
                                     input_layer=input_layer))
            return value

        def boundary(self, model, input_layer):
            y_0 = model(self.parameters['constant']['zero'])
            y_L = model(self.parameters['constant']['length'])
            d2y_dx2_0 = self.derivative(differential={'x': 2},
                                        model=model,
                                        input_layer=self.parameters['constant']['zero'])
            d2y_dx2_L = self.derivative(differential={'x': 2},
                                        model=model,
                                        input_layer=self.parameters['constant']['length'])
            return y_0, y_L, d2y_dx2_0, d2y_dx2_L

        def equation(self, model, input_layer):
            return (self.parameters['constant']['w'] /
                    self.parameters['constant']['E'] /
                    self.parameters['constant']['I'] / 24.0 *
                    (-input_layer ** 4 + 2.0 * self.parameters['constant']['L'] * input_layer ** 3 -
                     input_layer * self.parameters['constant']['L'] ** 3)) - model(input_layer)

        beam_example_1.collocation = types.MethodType(collocation, beam_example_1)
        beam_example_1.boundary = types.MethodType(boundary, beam_example_1)
        beam_example_1.equation = types.MethodType(equation, beam_example_1)
        nn.train_network(batch_size=16,
                         epochs=500,
                         test_parameters={'samples': 100,
                                          'test_error': 10 ** -4,
                                          'epoch_error': 10 ** -4})

    def beam_2():
        beam_example_2 = GoverningEquation()
        beam_example_2.read_constant(constant={'E': 1.0,
                                               'I': 1.0,
                                               'L': 1.0,
                                               'w': 1.0,
                                               'zero': tf.constant(0., shape=(1, 1), dtype=tf.float64),
                                               'length': tf.constant(1.0, shape=(1, 1), dtype=tf.float64)})
        beam_example_2.read_dependent_domain(dependent={'x': (0.0, 'L')})
        nn = NeuralNetwork(governing=beam_example_2)
        nn.set_layers(number_neurons_per_layer=[40, 40, 1],
                      activation_function='tanh')
        nn.set_training(number_train_sets=1000)

        def collocation(self, model, input_layer):
            value = (self.parameters['constant']['w'] / self.parameters['constant']['E'] / self.parameters['constant']['I'] +
                     self.derivative(differential={'x': 4},
                                     model=model,
                                     input_layer=input_layer))
            return value

        def boundary(self, model, input_layer):
            y_0 = model(self.parameters['constant']['zero'])
            dy_dx_0 = self.derivative(differential={'x': 1},
                                      model=model,
                                      input_layer=self.parameters['constant']['zero'])
            d2y_dx2_L = self.derivative(differential={'x': 2},
                                        model=model,
                                        input_layer=self.parameters['constant']['length'])
            d3y_dx3_L = self.derivative(differential={'x': 3},
                                        model=model,
                                        input_layer=self.parameters['constant']['length'])
            return y_0, dy_dx_0, d2y_dx2_L, d3y_dx3_L

        def equation(self, model, input_layer):
            return (self.parameters['constant']['w'] /
                    self.parameters['constant']['E'] /
                    self.parameters['constant']['I'] / 24.0 *
                    (-input_layer ** 4 + 4.0 * self.parameters['constant']['L'] * input_layer ** 3 -
                     6.0 * self.parameters['constant']['L'] ** 2 * input_layer ** 2)) - model(input_layer)

        beam_example_2.collocation = types.MethodType(collocation, beam_example_2)
        beam_example_2.boundary = types.MethodType(boundary, beam_example_2)
        beam_example_2.equation = types.MethodType(equation, beam_example_2)
        nn.train_network(batch_size=16,
                         epochs=500,
                         test_parameters={'samples': 100,
                                          'test_error': 10 ** -4,
                                          'epoch_error': 10 ** -4})

    def beam_3():
        beam_example_3 = GoverningEquation()
        beam_example_3.read_constant(constant={'E': 1.0,
                                               'I': 1.0,
                                               'L': 1.0,
                                               'w': 1.0,
                                               'zero': tf.constant(0., shape=(1, 1), dtype=tf.float64),
                                               'length': tf.constant(1.0, shape=(1, 1), dtype=tf.float64)})
        beam_example_3.read_dependent_domain(dependent={'x': (0.0, 'L')})
        nn = NeuralNetwork(governing=beam_example_3)
        nn.set_layers(number_neurons_per_layer=[40, 40, 1],
                      activation_function='tanh')
        nn.set_training(number_train_sets=1000)

        def collocation(self, model, input_layer):
            value = (self.parameters['constant']['w'] / self.parameters['constant']['E'] / self.parameters['constant']['I'] +
                     self.derivative(differential={'x': 4},
                                     model=model,
                                     input_layer=input_layer))
            return value

        def boundary(self, model, input_layer):
            y_0 = model(self.parameters['constant']['zero'])
            y_L = model(self.parameters['constant']['length'])
            dy_dx_0 = self.derivative(differential={'x': 1},
                                      model=model,
                                      input_layer=self.parameters['constant']['zero'])
            d2y_dx2_L = self.derivative(differential={'x': 2},
                                        model=model,
                                        input_layer=self.parameters['constant']['length'])
            return y_0, y_L, dy_dx_0, d2y_dx2_L

        def equation(self, model, input_layer):
            return (self.parameters['constant']['w'] /
                    self.parameters['constant']['E'] /
                    self.parameters['constant']['I'] / 24.0 *
                    (-input_layer ** 4 +
                     5.0 / 2.0 * self.parameters['constant']['L'] * input_layer ** 3 -
                     3.0 / 2.0 * input_layer ** 2)) - model(input_layer)

        beam_example_3.collocation = types.MethodType(collocation, beam_example_3)
        beam_example_3.boundary = types.MethodType(boundary, beam_example_3)
        beam_example_3.equation = types.MethodType(equation, beam_example_3)
        nn.train_network(batch_size=16,
                         epochs=500,
                         test_parameters={'samples': 100,
                                          'test_error': 10 ** -4,
                                          'epoch_error': 10 ** -4})

    def heat_1d_stationary():
        heat_1d_example_1 = GoverningEquation()
        heat_1d_example_1.read_constant(constant={'alpha': 20,
                                                  'Qt': 10,
                                                  'c_1': 70.344995319758030,
                                                  'c_2': 2.796550046802420 * 10 ** 2,
                                                  'zero': tf.constant(0., shape=(1, 1), dtype=tf.float64),
                                                  'length': tf.constant(1.5, shape=(1, 1), dtype=tf.float64)})

        heat_1d_example_1.read_dependent_domain(dependent={'x': (0.0, 1.5)})
        nn = NeuralNetwork(governing=heat_1d_example_1)
        nn.set_layers(number_neurons_per_layer=[50, 50, 1],
                      activation_function='tanh')
        nn.set_training(number_train_sets=1000)

        def collocation(self, model, input_layer):
            value = (self.parameters['constant']['alpha'] * self.derivative(differential={'x': 2}, model=model, input_layer=input_layer) +
                     self.parameters['constant']['Qt'] * model(input_layer))
            return value

        def boundary(self, model, input_layer):
            T_0 = model(self.parameters['constant']['zero']) - 350
            T_L = model(self.parameters['constant']['length']) - 300
            return T_0, T_L

        def equation(self, model, input_layer):
            return (self.parameters['constant']['c_1'] *
                    np.exp(np.sqrt(self.parameters['constant']['Qt'] / self.parameters['constant']['alpha']) *
                           input_layer) +
                    self.parameters['constant']['c_2'] *
                    np.exp(-np.sqrt(self.parameters['constant']['Qt'] / self.parameters['constant']['alpha']) *
                           input_layer)) - model(input_layer)

        heat_1d_example_1.collocation = types.MethodType(collocation, heat_1d_example_1)
        heat_1d_example_1.boundary = types.MethodType(boundary, heat_1d_example_1)
        heat_1d_example_1.equation = types.MethodType(equation, heat_1d_example_1)
        nn.train_network(batch_size=16,
                         epochs=500,
                         test_parameters={'samples': 100,
                                          'test_error': 10 ** -4,
                                          'epoch_error': 10 ** -4})

    def heat_1d_transient():
        heat_1d_example = GoverningEquation()
        heat_1d_example.read_constant(constant={'alpha': 20,
                                                'Qt': 10,
                                                'c_1': 70.344995319758030,
                                                'c_2': 2.796550046802420 * 10 ** 2,
                                                'zero': 0.0,
                                                'L': 1.5})

        heat_1d_example.read_dependent_domain(dependent={'x': (0.0, 1.5),
                                                         't': (0.0, 1.0)})
        nn = NeuralNetwork(governing=heat_1d_example)
        nn.set_layers(number_neurons_per_layer=[15, 30, 15, 1],
                      activation_function='tanh')
        nn.set_training(number_train_sets=1000)

        def collocation(self, model, input_layer):
            value = (self.derivative(differential={'t': 1}, model=model, input_layer=input_layer) -
                     self.parameters['constant']['alpha'] * self.derivative(differential={'x': 2}, model=model, input_layer=input_layer) +
                     self.parameters['constant']['Qt'] * model(input_layer))
            return value

        def boundary(self, model, input_layer):
            batch_size = tf.shape(input_layer)[0]
            T_0_t = model(tf.stack([tf.fill(batch_size, self.parameters['constant']['zero']), input_layer[(slice(None), self.index['t'])]], axis=1)) - 350
            T_L_t = model(tf.stack([tf.fill(batch_size, self.parameters['constant']['L']), input_layer[(slice(None), self.index['t'])]], axis=1)) - 300
            return T_0_t, T_L_t

        def equation(self, model, input_layer):
            return (self.parameters['constant']['c_1'] *
                    np.exp(np.sqrt(self.parameters['constant']['Qt'] / self.parameters['constant']['alpha']) * input_layer[(slice(None), self.index['x'])]) +
                    self.parameters['constant']['c_2'] *
                    np.exp(-np.sqrt(self.parameters['constant']['Qt'] / self.parameters['constant']['alpha']) * input_layer[(slice(None), self.index['x'])]) +
                    self.parameters['constant']['Qt'] *
                    np.exp(-((self.parameters['constant']['alpha'] * (np.pi / 2) ** 2) + self.parameters['constant']['Qt']) * input_layer[(slice(None), self.index['t'])]) *
                    np.sin(np.pi / 2 * input_layer[(slice(None), self.index['x'])])) - model(input_layer)

        heat_1d_example.collocation = types.MethodType(collocation, heat_1d_example)
        heat_1d_example.boundary = types.MethodType(boundary, heat_1d_example)
        heat_1d_example.equation = types.MethodType(equation, heat_1d_example)
        nn.train_network(batch_size=16,
                         epochs=500,
                         test_parameters={'samples': 1000,
                                          'test_error': 10 ** -4,
                                          'epoch_error': 10 ** -4,
                                          'plot_x': 'x'})

    def beam_2_transient():
        beam_2_transient_example = GoverningEquation()
        beam_2_transient_example.read_constant(constant={'E': 1.0,
                                                         'I': 1.0,
                                                         'L': 1.0,
                                                         'rho': 1.0,
                                                         'zero': 0.0})

        def collocation(self, model, input_layer):
            value = (self.parameters['constant']['E'] /
                     self.parameters['constant']['I'] *
                     self.derivative(differential={'x': 4}, model=model, input_layer=input_layer) +
                     self.parameters['constant']['rho'] *
                     self.derivative(differential={'t': 2}, model=model, input_layer=input_layer))
            return value

        def boundary(self, model, input_layer):
            batch_size = tf.shape(input_layer)[0]
            y_0_t = model(tf.stack([tf.fill(batch_size, self.parameters['constant']['zero']), input_layer[(slice(None), self.index['t'])]], axis=1))
            dy_dx_0_t = self.derivative(differential={'x': 1},
                                        model=model,
                                        input_layer=self.parameters['constant']['zero'])
            d2y_dx2_L_t = self.derivative(differential={'x': 2},
                                          model=model,
                                          input_layer=self.parameters['constant']['length'])
            d3y_dx3_L_t = self.derivative(differential={'x': 3},
                                          model=model,
                                          input_layer=self.parameters['constant']['length'])
            return y_0_t, dy_dx_0_t, d2y_dx2_L_t, d3y_dx3_L_t

    def elliptical():
        elliptical_example = GoverningEquation()
        elliptical_example.read_constant(constant={'zero': tf.constant(0., shape=(1, 1), dtype=tf.float64),
                                                   'one': tf.constant(1., shape=(1, 1), dtype=tf.float64),
                                                   'kappa': 1.4})
        elliptical_example.read_dependent_domain(dependent={'x': (0.0, 1.0)})
        nn = NeuralNetwork(governing=elliptical_example)
        nn.set_layers(number_neurons_per_layer=[10, 15, 10, 1],
                      activation_function='tanh')
        nn.set_training(number_train_sets=1000)

        def collocation(self, model, input_layer):
            value = (self.derivative(differential={'x': 2},
                                     model=model,
                                     input_layer=input_layer) +
                     tf.math.sin(2.0 * np.pi *
                                 input_layer *
                                 self.parameters['constant']['kappa']))
            return value

        def boundary(self, model, input_layer):
            #batch_size = tf.shape(input_layer)[0]
            start = model(self.parameters['constant']['zero'])
            end = model(self.parameters['constant']['one'])
            #end = model(tf.stack([tf.fill(batch_size, self.parameters['constant']['one']), tf.fill(batch_size, self.parameters['constant']['kappa'])], axis=1))
            return start, end

        def equation(self, model, input_layer):
            return (1 / ((2.0 * np.pi * self.parameters['constant']['kappa']) ** 2) *
                    (tf.math.sin(2 * np.pi * self.parameters['constant']['kappa'] * input_layer) -
                     tf.math.sin(2 * np.pi * self.parameters['constant']['kappa']) * input_layer)
                    ) - model(input_layer)
        elliptical_example.collocation = types.MethodType(collocation, elliptical_example)
        elliptical_example.boundary = types.MethodType(boundary, elliptical_example)
        elliptical_example.equation = types.MethodType(equation, elliptical_example)
        nn.train_network(batch_size=16,
                         epochs=500,
                         test_parameters={'samples': 100,
                                          'test_error': 10 ** -4,
                                          'plot_x': 'x'})


    elliptical()