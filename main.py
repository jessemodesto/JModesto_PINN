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


class PINN:
    def __init__(self, constants: dict, data: dict, layers: list, activation_function: str):
        # variables
        self.v = {'c': {},  # constants
                  't': {}}  # trainables
        self.i = {}  # index of dependents
        self.data = {}
        self.model = None  # neural network model
        self.loss_function = None  # loss function of the neural network
        self.loss_function_batch = None  # custom loss function if batching
        self.read_constant(constants=constants)
        self.set_training(data=data)
        self.set_layers(layers=layers, activation_function=activation_function)

    def read_constant(self, constants: dict):
        for constant_variable in constants:
            self.v['c'][constant_variable] = constants[constant_variable]
        return

    def set_training(self, data: dict):
        index = None
        for data_type in data:
            if isinstance(data[data_type], tuple):
                if data_type not in self.data.keys():
                    self.data[data_type] = []
                for data_set in data[data_type]:
                    length = len(data_set)
                    training = np.array(np.meshgrid(*tuple(data_set.values())), dtype=np.float64).T.reshape(-1, length)
                    self.data[data_type].append(tf.constant(training, dtype=tf.float64))
            else:
                if index is None:
                    for index, dependent_variable in enumerate(data[data_type]):
                        self.i[dependent_variable] = index
                length = len(data[data_type])
                training = np.array(np.meshgrid(*tuple(data[data_type].values())), dtype=np.float64).T.reshape(-1, length)
                self.data[data_type] = tf.constant(training, dtype=tf.float64)
        return

    def set_layers(self, layers: list, activation_function: str):
        self.model = Sequential()
        input_layer_size = (self.data['test']).shape[1]
        self.model.add(Input(shape=(input_layer_size,),
                             dtype='float64'))
        for i in range(0, len(layers) - 1):  # if at least one hidden layer
            self.model.add(Dense(units=layers[i],  # add hidden layers to model
                                 activation=activation_function,
                                 kernel_initializer=HeNormal(),
                                 dtype='float64'))
        self.model.add(Dense(units=layers[-1],  # add output layer to model
                             activation=None,
                             kernel_initializer=HeNormal(),
                             dtype='float64'))

    def d(self, wrt: dict, model, data):
        max_order = sum(wrt.values())
        if max_order == 1:
            with tf.GradientTape(persistent=False) as tape:
                tape.watch(data)
                value = model(data)
            value = tape.gradient(value, data)
        else:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(data)
                value = model(data)
                for i in range(max_order):
                    value = tape.gradient(value, data)
        if len(self.i) == 1:
            return value
        else:
            return tf.expand_dims(value[:, self.i[next(iter(wrt))]], axis=1)

    def train_network(self, epochs: int, batches: dict = None, optimizer=None, plot_x: str = None, test_error: float = None):
        if optimizer is None:
            optimizer = Adam()
        if plot_x is not None or test_error is not None:
            testing_output = tf.reshape(self.equation(lambda x: 0, self.data['test']), [-1])
            error = 0
            if plot_x is not None:
                dynamic_plot = DynamicPlot()
                dynamic_plot.plot(pairs=(
                    (self.data['test'][:, self.i[plot_x]], testing_output),
                    (self.data['test'][:, self.i[plot_x]], tf.zeros([testing_output.shape[0]], dtype=tf.float64))))
        if batches is None:  # full size batching
            progress_bar = Progbar(target=epochs)
            for epoch in range(1, epochs + 1):
                with tf.GradientTape() as tape:
                    loss = self.loss_function(self.model, self.data)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                if plot_x is not None or test_error is not None:
                    testing_output = self.model(self.data['test'])[:, 0]
                    if plot_x is not None:
                        dynamic_plot.update(y=testing_output, plot_number=1)
                    if test_error is not None:
                        error = tf.math.reduce_mean(tf.math.abs(testing_output - self.model(self.data['test'])))
                        progress_bar.update(epoch, values=[('loss', loss), ('test error', error)])
                        if error < test_error:
                            if plot_x is not None:
                                dynamic_plot.save_figure()
                            return epoch,
                else:
                    progress_bar.update(epoch, values=[('loss', loss), ])
        else:  # proportional batching
            number_of_batches = 0
            for batch_set in batches:
                self.data[batch_set] = tf.data.Dataset.from_tensor_slices(self.data[batch_set])
                self.data[batch_set] = self.data[batch_set].shuffle(buffer_size=1024).batch(batches[batch_set])
                number_of_batches = max(number_of_batches, len(self.data[batch_set]))
            progress_bar = Progbar(target=number_of_batches)
            for epoch in range(1, epochs + 1):
                tf.print(f"Epoch: {epoch}/{epochs}")
                train_batch = {batch_set: [*self.data[batch_set]] for batch_set in batches}
                for index in range(1, number_of_batches + 1):
                    with tf.GradientTape() as tape:
                        loss = self.loss_function_batch(self.model, train_batch, index-1)
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    if index != number_of_batches:
                        if test_error is not None:
                            progress_bar.update(index, values=[('loss', loss), ('test error', error)])
                        else:
                            progress_bar.update(index, values=[('loss', loss), ])
                if plot_x is not None or test_error is not None:
                    testing_output = tf.reshape(self.model(self.data['test']), [-1])
                    if plot_x is not None:
                        dynamic_plot.update(y=testing_output, plot_number=1)
                    if test_error is not None:
                        error = tf.math.reduce_mean(tf.math.abs(testing_output - self.model(self.data['test'])))
                        progress_bar.update(index, values=[('loss', loss), ('test error', error)])
                        if error < test_error:
                            if plot_x is not None:
                                dynamic_plot.save_figure()
                            return epoch, error.numpy(), loss.numpy()
                else:
                    progress_bar.update(index, values=[('loss', loss), ])
        return


class DynamicPlot:
    def __init__(self):
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.data = {}
        plt.ion()

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
        plt.show()
        return

    def save_figure(self):
        plt.savefig('foo.png')
        return


if __name__ == "__main__":
    # sample PINN using 1-D transient heat equation
    pinn = PINN(  # constants declared here so that they are not constructed each time in functions
        constants={'alpha': tf.constant(value=20., shape=(1, 1), dtype=tf.float64),
                   'Q_t': tf.constant(value=10., shape=(1, 1), dtype=tf.float64),
                   'c_1': tf.constant(value=70.344995319758030, shape=(1, 1), dtype=tf.float64),
                   'c_2': tf.constant(value=2.796550046802420 * 10 ** 2, shape=(1, 1), dtype=tf.float64),
                   '0': tf.constant(value=0.0, shape=(100, 1), dtype=tf.float64),
                   'L': tf.constant(value=1.5, shape=(100, 1), dtype=tf.float64)},
        data={'collocation': {'x': np.random.uniform(low=0., high=1.5, size=100),  # datasets for each training method
                              't': np.random.uniform(low=1.0, high=1.0, size=1)},
              'boundary': {'t': np.random.uniform(low=0.0, high=1.0, size=100)},
              'test': {'x': np.linspace(start=0., stop=1.5, num=16, dtype=np.float64),
                       't': np.array([1.0], dtype=np.float64)}},
        layers=[15, 25, 15, 1],  # neuron per layers
        activation_function='tanh')  # activation function

    def collocation(self, model, data):
        value = (self.d(wrt={'t': 1}, model=model, data=data) -
                 self.v['c']['alpha'] * self.d(wrt={'x': 2}, model=model, data=data) +
                 self.v['c']['Q_t'] * model(data))
        return value

    def boundary(self, model, data):
        T_0_t = model(tf.stack([self.v['c']['0'], data], axis=1)) - 350.0
        T_L_t = model(tf.stack([self.v['c']['L'], data], axis=1)) - 300.0
        return T_0_t, T_L_t

    def equation(self, model, data):
        return (self.v['c']['c_1'] * tf.math.exp(tf.math.sqrt(self.v['c']['Q_t'] / self.v['c']['alpha']) * data[:, self.i['x']]) +
                self.v['c']['c_2'] * tf.math.exp(-tf.math.sqrt(self.v['c']['Q_t'] / self.v['c']['alpha']) * data[:, self.i['x']]) +
                self.v['c']['Q_t'] * tf.math.exp(-(self.v['c']['alpha'] * (np.pi / 2) ** 2 + self.v['c']['Q_t']) * data[:, self.i['t']]) *
                tf.math.sin(np.pi / 2 * data[:, self.i['x']])) - model(data)

    def loss_function(self, model, train_data):
        loss = tf.reduce_mean(tf.square(self.collocation(model, train_data['collocation'])))
        loss += tf.reduce_sum([tf.reduce_mean(tf.square(bc)) for bc in self.boundary(model, train_data['boundary'])])
        return loss

    pinn.collocation = types.MethodType(collocation, pinn)
    pinn.boundary = types.MethodType(boundary, pinn)
    pinn.equation = types.MethodType(equation, pinn)
    pinn.loss_function = types.MethodType(loss_function, pinn)
    pinn.train_network(epochs=5000,
                       batches={'collocation': 100,
                                'boundary': 100},
                       test_error=10 ** -4,
                       plot_x='x')
