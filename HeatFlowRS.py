import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate training data
alpha = 0.01  # Thermal diffusivity
np.random.seed(42)
num_samples = 1000
x_train = np.random.uniform(low=0, high=1, size=(num_samples, 1))
t_train = np.random.uniform(low=0, high=1, size=(num_samples, 1))
u_exact = np.sin(np.pi * x_train) * np.exp(-alpha * t_train)

# Convert to TensorFlow tensors
x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)
t_train_tf = tf.convert_to_tensor(t_train, dtype=tf.float32)
u_exact_tf = tf.convert_to_tensor(u_exact, dtype=tf.float32)

# Combine x and t for training input
input_train = tf.concat([x_train_tf, t_train_tf], axis=1)

# Generate test data for prediction
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
t_test = np.linspace(0, 1, 100).reshape(-1, 1)

# Convert to TensorFlow tensors
x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
t_test_tf = tf.convert_to_tensor(t_test, dtype=tf.float32)

input_test = tf.concat([x_test_tf, t_test_tf], axis=1)

u_exact_test = np.sin(np.pi * x_test) * np.exp(-alpha * t_test)
u_exact_test_tf = tf.convert_to_tensor(u_exact_test, dtype=tf.float32)

# Define the physics-informed neural network (PINN) model
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(50, activation='tanh', input_dim=2)
        self.dense2 = tf.keras.layers.Dense(50, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        concat_input = tf.concat([x, t], axis=1)
        hidden1 = self.dense1(concat_input)
        hidden2 = self.dense2(hidden1)
        output = self.output_layer(hidden2)
        return output

# Define the loss function (physics-informed loss)
def physics_loss(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        u_pred = model(tf.concat([x, t], axis=1))
        u_x = tape.gradient(u_pred, x)
        u_xx = tape.gradient(u_x, x)
        u_t = tape.gradient(u_pred, t)
        del tape

    alpha = 0.01  # Thermal diffusivity

    # Define the heat equation PDE
    pde_loss = u_t - alpha * u_xx

    return tf.reduce_mean(tf.square(pde_loss))


# Create and compile the PINN model
model = PINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        physics_loss_value = physics_loss(model, x_train_tf, t_train_tf)
        data_loss_value = tf.reduce_mean(tf.square(model(input_train) - u_exact_tf))
        total_loss = physics_loss_value + data_loss_value

    gradients = tape.gradient(total_loss, model.trainable_variables)
    #gradients = tape.gradient(physics_loss_value, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Total Loss: {total_loss.numpy()}, Physics Loss: {physics_loss_value.numpy()}, Data Loss: {data_loss_value.numpy()}")
        #print(f"Epoch {epoch}/{num_epochs}, Physics Loss: {physics_loss_value.numpy()}")
u_pred=model(input_test)


# Final eval
x_eval = tf.convert_to_tensor(np.reshape(np.linspace(0,1,11),[11,1]), dtype=tf.float32)
t_eval = tf.convert_to_tensor(np.zeros([11,1]), dtype=tf.float32)
u_pred = model(tf.concat([x_eval, t_eval], axis=1))
u_ex = np.sin(np.pi * np.reshape(np.linspace(0,1,11),[11,1])) * np.exp(-alpha * np.zeros([11,1]))

plt.plot(x_eval.numpy(),u_pred.numpy())
plt.plot(x_eval.numpy(),u_ex)
plt.show()

x_eval = tf.convert_to_tensor(np.reshape(np.linspace(0,1,11),[11,1]), dtype=tf.float32)
t_eval = tf.convert_to_tensor(np.ones([11,1]), dtype=tf.float32)
u_pred = model(tf.concat([x_eval, t_eval], axis=1))
u_ex = np.sin(np.pi * np.reshape(np.linspace(0,1,11),[11,1])) * np.exp(-alpha * np.ones([11,1]))

plt.plot(x_eval.numpy(),u_pred.numpy())
plt.plot(x_eval.numpy(),u_ex)
plt.show()

print('end')

