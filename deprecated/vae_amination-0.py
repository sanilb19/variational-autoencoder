import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras import layers, models

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape the data
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

# Define the encoder network
latent_dim = 2  # Latent space dimension for visualization

encoder_inputs = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation='relu')(encoder_inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)

# Latent space parameters
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Sampling layer (reparameterization trick)
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Define the decoder network
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
decoder_outputs = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

# Build the VAE model
vae = models.Model(encoder_inputs, decoder_outputs)

# Define the VAE loss (reconstruction loss + KL divergence)
def vae_loss(y_true, y_pred):
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(y_true, y_pred), axis=(1, 2)))
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))
    return reconstruction_loss + kl_loss

# Compile the model
vae.compile(optimizer='adam', loss=vae_loss)

# Define encoder model to extract latent space (mu points)
encoder_model = models.Model(vae.input, z_mean)

# Callback to visualize latent space after each epoch
class LatentSpaceCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Extract the mu points from the encoder after each epoch
        mu_points = encoder_model.predict(train_images, batch_size=128)
        if epoch % 10 == 0:  # Save and plot every 10 epochs
            self.plot_latent_space(mu_points, train_labels)

    def plot_latent_space(self, mu_points, labels):
        plt.figure(figsize=(8, 6))
        plt.scatter(mu_points[:, 0], mu_points[:, 1], c=labels, cmap='tab10')
        plt.colorbar()
        plt.title(f"Latent Space at Epoch {len(mu_points)}")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.show()

# Train the model
latent_space_callback = LatentSpaceCallback()
vae.fit(train_images, train_images, epochs=50, batch_size=128, validation_data=(test_images, test_images), callbacks=[latent_space_callback])

# Visualization of the latent space after training
mu_points = encoder_model.predict(test_images, batch_size=128)

# Scatter plot of the latent space colored by digit labels
plt.figure(figsize=(8, 6))
plt.scatter(mu_points[:, 0], mu_points[:, 1], c=test_labels, cmap='tab10')
plt.colorbar()
plt.title("Latent Space after Training")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()

