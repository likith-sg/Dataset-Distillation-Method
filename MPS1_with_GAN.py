import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import random
from tensorflow.keras.callbacks import ModelCheckpoint

# Loading MNIST Dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train[..., np.newaxis] / 255.0
X_test = X_test[..., np.newaxis] / 255.0

# Selecting 1000 training examples
X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=1000, random_state=42)

# Define CNN model
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define GAN architecture
class GAN:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizers.Adam(),
                                   metrics=['accuracy'])
        self.combined = self.build_combined()

    def build_generator(self):
        model = models.Sequential()
        model.add(layers.Dense(256, input_dim=100, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(np.prod(self.img_shape), activation='sigmoid'))
        model.add(layers.Reshape(self.img_shape))
        return model

    def build_discriminator(self):
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=self.img_shape))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def build_combined(self):
        self.generator.trainable = False
        z = layers.Input(shape=(100,))
        img = self.generator(z)
        valid = self.discriminator(img)
        combined = models.Model(z, valid)
        combined.compile(loss='binary_crossentropy', optimizer=optimizers.Adam())
        return combined

    def train(self, X_train, epochs, batch_size=128):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.combined.train_on_batch(noise, valid)

    def generate_synthetic_data(self, num_examples):
        noise = np.random.normal(0, 1, (num_examples, 100))
        gen_imgs = self.generator.predict(noise)
        return gen_imgs

# RL Agent for dataset distillation
class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Function to create synthetic dataset with GAN
def create_synthetic_dataset_with_gan(X_train, y_train, num_examples, gan):
    X_synthetic = np.zeros((num_examples, 28, 28, 1))
    y_synthetic = np.zeros((num_examples,), dtype=int)

    # Generate synthetic data using GAN
    synthetic_data = gan.generate_synthetic_data(num_examples)

    # Replace some examples with GAN-generated data
    indices = np.random.choice(len(X_train), num_examples, replace=False)
    X_synthetic[:num_examples//2] = synthetic_data[:num_examples//2]
    y_synthetic[:num_examples//2] = np.random.randint(0, 10, size=num_examples//2)

    return X_synthetic, y_synthetic

# Create RL agent
state_size = X_train.shape[1] * X_train.shape[2]
action_size = 10  # Number of labels
agent = RLAgent(state_size, action_size)

# Function to create synthetic dataset with RL optimization
def create_synthetic_dataset_with_rl(X_train, y_train, num_examples, agent):
    X_synthetic = np.zeros((num_examples, 28, 28, 1))
    y_synthetic = np.zeros((num_examples,), dtype=int)

    indices = np.random.choice(len(X_train), num_examples, replace=False)
    X_synthetic = X_train[indices]
    y_synthetic = y_train[indices]

    for epoch in range(100):
        synthetic_model = create_cnn_model()
        synthetic_model.fit(X_synthetic, y_synthetic, epochs=5, batch_size=32, verbose=0)
        y_synthetic_pred = np.argmax(synthetic_model.predict(X_synthetic), axis=1)

        for i in range(num_examples):
            state = X_synthetic[i].flatten().reshape(1, -1)
            action = y_synthetic_pred[i]
            reward = 1 if y_synthetic[i] == action else -1
            next_state = state
            done = True
            agent.remember(state, action, reward, next_state, done)

        agent.replay(32)

    return X_synthetic, y_synthetic

# Initialize GAN and train
img_shape = X_train_subset.shape[1:]
gan = GAN(img_shape)
gan.train(X_train_subset, epochs=100, batch_size=32)

# Create synthetic dataset with GAN
X_synthetic_gan, y_synthetic_gan = create_synthetic_dataset_with_gan(X_train_subset, y_train_subset, 437, gan)

# Create synthetic dataset with RL optimization
X_synthetic_rl, y_synthetic_rl = create_synthetic_dataset_with_rl(X_train_subset, y_train_subset, 437, agent)

# Combine synthetic datasets from GAN and RL
X_synthetic_combined = np.concatenate([X_synthetic_gan, X_synthetic_rl])
y_synthetic_combined = np.concatenate([y_synthetic_gan, y_synthetic_rl])


# Define a checkpoint callback to save the best model
checkpoint_path = "best_model.keras.h5"
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max',
                                      verbose=1)

# Train your synthetic model on the combined dataset
synthetic_model = create_cnn_model()
synthetic_model.fit(X_synthetic_combined, y_synthetic_combined,
                    epochs=5, batch_size=32,
                    validation_split=0.2,
                    callbacks=[checkpoint_callback])

# Load the best model
synthetic_model.load_weights(checkpoint_path)

# Evaluate on the test set
test_loss, test_acc = synthetic_model.evaluate(X_test, y_test)
print(f"Test accuracy on synthetic dataset: {test_acc}")
