import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape the data
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

# Define a custom VAE class
class VAE(tf.keras.Model):
    def __init__(self, latent_dim=2, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Define encoder layers
        self.encoder_conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.encoder_pool1 = layers.MaxPooling2D((2, 2))
        self.encoder_conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.encoder_pool2 = layers.MaxPooling2D((2, 2))
        self.encoder_flatten = layers.Flatten()
        self.encoder_dense = layers.Dense(64, activation='relu')
        
        # Latent space parameters
        self.z_mean_layer = layers.Dense(latent_dim)
        self.z_log_var_layer = layers.Dense(latent_dim)
        
        # Define decoder layers
        self.decoder_dense = layers.Dense(7 * 7 * 64, activation='relu')
        self.decoder_reshape = layers.Reshape((7, 7, 64))
        self.decoder_conv_t1 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')
        self.decoder_conv_t2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')
        self.decoder_output = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')
        
    def encode(self, x):
        x = self.encoder_conv1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_flatten(x)
        x = self.encoder_dense(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def decode(self, z):
        x = self.decoder_dense(z)
        x = self.decoder_reshape(x)
        x = self.decoder_conv_t1(x)
        x = self.decoder_conv_t2(x)
        return self.decoder_output(x)
    
    def call(self, inputs, training=None):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        
        # Add KL divergence loss as a model loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        )
        self.add_loss(kl_loss)
        
        return reconstructed

    def get_latent_encoding(self, inputs):
        z_mean, _ = self.encode(inputs)
        return z_mean

# Create VAE model
vae = VAE(latent_dim=2)

# Define the reconstruction loss function (pixel-wise binary crossentropy)
def reconstruction_loss(y_true, y_pred):
    return tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(y_true, y_pred), 
            axis=(1, 2)
        )
    ) * 28 * 28  # Scale by image dimensions

# Compile the model
vae.compile(optimizer='adam', loss=reconstruction_loss)

# Callback to visualize latent space after each epoch
class LatentSpaceVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, vae_model, images, labels, save_path=None):
        super().__init__()
        self.vae_model = vae_model  # Changed from self.model to self.vae_model
        self.images = images
        self.labels = labels
        self.save_path = save_path
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Save and plot every 5 epochs
            # Use a smaller subset for visualization during training to improve speed
            sample_size = min(5000, len(self.images))
            sample_idx = np.random.choice(len(self.images), sample_size, replace=False)
            sample_images = self.images[sample_idx]
            sample_labels = self.labels[sample_idx]
            
            # Get latent space representations
            mu_points = self.vae_model.get_latent_encoding(sample_images)
            self.plot_latent_space(mu_points, sample_labels, epoch)
    
    def plot_latent_space(self, mu_points, labels, epoch):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(mu_points[:, 0], mu_points[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f"Latent Space at Epoch {epoch}")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        
        # Save plot to file
        plt.savefig(f"vae_animation-1/outputs/latent_space_epoch_{epoch}.png")
        plt.close()

# Create the callback with a different name for the model parameter
latent_space_callback = LatentSpaceVisualizer(vae, train_images, train_labels)

# Train the model (using a smaller batch size for M3 Mac)
history = vae.fit(
    train_images, 
    train_images,
    epochs=30,  # Reduced epochs for faster training
    batch_size=64,  # Smaller batch size for M3 Mac
    validation_data=(test_images, test_images),
    callbacks=[latent_space_callback]
)

# Get latent space representations for visualization
mu_points = vae.get_latent_encoding(test_images)

# Scatter plot of the latent space colored by digit labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(mu_points[:, 0], mu_points[:, 1], c=test_labels, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title("Latent Space after Training")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.tight_layout()
plt.savefig("vae_animation-1/outputs/final_latent_space.png")
plt.show()

# Generate digits from the latent space
n = 15  # Grid size
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Create a grid of values from -3 to 3 for both latent dimensions
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)[::-1]

# Decode for each grid point
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = vae.decode(z_sample)
        digit = x_decoded[0].numpy().reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = n * digit_size + start_range
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.imshow(figure, cmap='Greys_r')
plt.title("Digits generated from the latent space")
plt.savefig("vae_animation-1/outputs/generated_digits.png")
plt.show()
