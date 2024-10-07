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

    def train_network(self, batch_size: int, epochs: int, optimizer=Adam(), plot: dict = False, x_axis: str = False):
        self.training = tf.data.Dataset.from_tensor_slices(self.training)
        self.training = self.training.shuffle(buffer_size=1024).batch(batch_size)
        number_of_batches = len(self.training)
        progress_bar = Progbar(target=number_of_batches)
        loss_function = self.loss_function()

        if plot is not False and x_axis is not False:
            dynamic_plot = DynamicPlot()
            values = np.array([v for v in zip(*plot.values())], dtype=np.float64)
            values_dict = [dict(zip(plot, v)) for v in zip(*plot.values())]
            output = np.array([self.governing.equation(val) for val in values_dict], dtype=np.float64)
            dynamic_plot.plot(
                (
                    (plot[x_axis], output),
                    (plot[x_axis], np.zeros(len(output)))
                ))

        for epoch in range(epochs):
            tf.print(f"Epoch: {epoch + 1}/{epochs}")
            step = 0
            for train_batch in self.training:
                step += 1
                with tf.GradientTape() as tape:
                    loss = loss_function(self.model, train_batch)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                if plot is not False and x_axis is not False:
                    test_prediction = np.array(self.model(values), dtype=np.float64).reshape(-1)
                    dynamic_plot.update(y=test_prediction, plot_number=1)
                    test_loss = np.mean(np.square(test_prediction - output), dtype=np.float64)
                    progress_bar.update(step, values=[('loss', loss), ('test loss', test_loss)])
                else:
                    progress_bar.update(step, values=[('loss', loss)])
        return

    def loss_function(self):
        result = []
        if 'collocation' in self.governing.__dict__:
            result.append(lambda model, train_batch: tf.reduce_mean(tf.square(self.governing.collocation(model, train_batch))))
        if 'boundary' in self.governing.__dict__:
            result.append(lambda model, train_batch: tf.reduce_sum([tf.square(bound) for bound in self.governing.boundary(model, train_batch)]))
        if 'hybrid' in self.governing.__dict__:
            pass
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
    def rod_1():
        rod_example_1 = GoverningEquation()
        rod_example_1.read_constant(constant={'c': 1.0,
                                              'A': 1.0,
                                              'E': 1.0,
                                              'L': 1.0,
                                              'zero': tf.constant(0., shape=(1, 1), dtype=tf.float64),
                                              'length': tf.constant(1.0, shape=(1, 1), dtype=tf.float64)})
        rod_example_1.read_dependent_domain(dependent={'x': (0.0, 'L')})
        nn = NeuralNetwork(governing=rod_example_1)
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
            end = self.derivative(differential={'x': 1},
                                  model=model,
                                  input_layer=self.parameters['constant']['length'])
            return fix, end

        def equation(self, input_dictionary):
            return self.parameters['constant']['c'] / 6.0 / self.parameters['constant']['A'] / self.parameters['constant']['E'] * (-(input_dictionary['x'] ** 3) + (3 * (self.parameters['constant']['L'] ** 2) * input_dictionary['x']))

        rod_example_1.collocation = types.MethodType(collocation, rod_example_1)
        rod_example_1.boundary = types.MethodType(boundary, rod_example_1)
        rod_example_1.equation = types.MethodType(equation, rod_example_1)
        nn.train_network(batch_size=16, epochs=500, plot={'x': np.linspace(0, 1, 10).astype(np.float64)}, x_axis='x')

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

        def equation(self, input_dictionary):
            return self.parameters['constant']['c'] / self.parameters['constant']['A'] / self.parameters['constant']['E'] * (-(input_dictionary['x'] ** 3) / 6 + (self.parameters['constant']['L'] ** 2) * input_dictionary['x'])

        rod_example_2.collocation = types.MethodType(collocation, rod_example_2)
        rod_example_2.boundary = types.MethodType(boundary, rod_example_2)
        rod_example_2.equation = types.MethodType(equation, rod_example_2)
        nn.train_network(batch_size=16, epochs=500, plot={'x': np.linspace(0, 1, 10).astype(np.float64)}, x_axis='x')

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

        def equation(self, input_dictionary):
            return self.parameters['constant']['w'] / self.parameters['constant']['E'] / self.parameters['constant']['I'] / 24.0 * (-input_dictionary['x'] ** 4 + 2.0 * self.parameters['constant']['L'] * input_dictionary['x'] ** 3 - self.parameters['constant']['L'] ** 3 * input_dictionary['x'])

        beam_example_1.collocation = types.MethodType(collocation, beam_example_1)
        beam_example_1.boundary = types.MethodType(boundary, beam_example_1)
        beam_example_1.equation = types.MethodType(equation, beam_example_1)
        nn.train_network(batch_size=16, epochs=500, plot={'x': np.linspace(0, 1, 10).astype(np.float64)}, x_axis='x')

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

        def equation(self, input_dictionary):
            return self.parameters['constant']['w'] / self.parameters['constant']['E'] / self.parameters['constant']['I'] / 24.0 * (-input_dictionary['x'] ** 4 + 4.0 * self.parameters['constant']['L'] * input_dictionary['x'] ** 3 - 6.0 * self.parameters['constant']['L'] ** 2 * input_dictionary['x'] ** 2)

        beam_example_2.collocation = types.MethodType(collocation, beam_example_2)
        beam_example_2.boundary = types.MethodType(boundary, beam_example_2)
        beam_example_2.equation = types.MethodType(equation, beam_example_2)
        nn.train_network(batch_size=16, epochs=500, plot={'x': np.linspace(0, 1, 10).astype(np.float64)}, x_axis='x')

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

        def equation(self, input_dictionary):
            return self.parameters['constant']['w'] / self.parameters['constant']['E'] / self.parameters['constant']['I'] / 24.0 * (-input_dictionary['x'] ** 4 + 5.0 / 2.0 * self.parameters['constant']['L'] * input_dictionary['x'] ** 3 - 3.0 / 2.0 * input_dictionary['x'] ** 2)

        beam_example_3.collocation = types.MethodType(collocation, beam_example_3)
        beam_example_3.boundary = types.MethodType(boundary, beam_example_3)
        beam_example_3.equation = types.MethodType(equation, beam_example_3)
        nn.train_network(batch_size=16, epochs=500, plot={'x': np.linspace(0, 1, 10).astype(np.float64)}, x_axis='x')

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

        def equation(self, input_dictionary):
            return (self.parameters['constant']['c_1'] * np.exp(np.sqrt(self.parameters['constant']['Qt'] / self.parameters['constant']['alpha']) * input_dictionary['x']) +
                    self.parameters['constant']['c_2'] * np.exp(-np.sqrt(self.parameters['constant']['Qt'] / self.parameters['constant']['alpha']) * input_dictionary['x']))

        heat_1d_example_1.collocation = types.MethodType(collocation, heat_1d_example_1)
        heat_1d_example_1.boundary = types.MethodType(boundary, heat_1d_example_1)
        heat_1d_example_1.equation = types.MethodType(equation, heat_1d_example_1)
        nn.train_network(batch_size=32,
                         epochs=1000,
                         optimizer=Adam(),
                         plot={'x': np.linspace(0.0, 1.5, 10).astype(np.float64)},
                         x_axis='x')

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

        def equation(self, input_dictionary):
            return (self.parameters['constant']['c_1'] * np.exp(np.sqrt(self.parameters['constant']['Qt'] / self.parameters['constant']['alpha']) * input_dictionary['x']) +
                    self.parameters['constant']['c_2'] * np.exp(-np.sqrt(self.parameters['constant']['Qt'] / self.parameters['constant']['alpha']) * input_dictionary['x']) +
                    self.parameters['constant']['Qt'] * np.exp(-((self.parameters['constant']['alpha'] * (np.pi / 2) ** 2) + self.parameters['constant']['Qt']) * input_dictionary['t']) *
                    np.sin(np.pi / 2 * input_dictionary['x']))

        heat_1d_example.collocation = types.MethodType(collocation, heat_1d_example)
        heat_1d_example.boundary = types.MethodType(boundary, heat_1d_example)
        heat_1d_example.equation = types.MethodType(equation, heat_1d_example)
        nn.train_network(batch_size=32,
                         epochs=1000,
                         optimizer=Adam(),
                         plot={'x': np.linspace(0.0, 1.5, 10).astype(np.float64),
                               't': np.linspace(0.0, 1.0, 10).astype(np.float64)},
                         x_axis='x')

    rod_1()
