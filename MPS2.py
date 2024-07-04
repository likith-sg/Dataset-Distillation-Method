import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import random

# Function to select and load the dataset
def load_dataset(dataset_name):
    if dataset_name == "mnist":
        return tf.keras.datasets.mnist.load_data()
    elif dataset_name == "fashionmnist":
        return tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_name == "cifar10":
        return tf.keras.datasets.cifar10.load_data()
    elif dataset_name == "cifar100":
        return tf.keras.datasets.cifar100.load_data()
    else:
        raise ValueError("Dataset not supported.")

# Defining the CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# A NN with hidden layers having 24 neurons each, using epsilon for exploring and exploiting, training in batches
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
        model.add(layers.Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])  # Reshape state if necessary
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])  # Reshape state if necessary
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])  # Reshape next_state if necessary
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Function to create synthetic dataset
def create_synthetic_dataset(X_train, y_train, num_examples_per_class, num_epochs):
    num_classes = len(np.unique(y_train))
    num_examples = num_examples_per_class * num_classes
    X_synthetic = np.zeros((num_examples, *X_train.shape[1:]))
    y_synthetic = np.zeros((num_examples,), dtype=int)

    for class_label in range(num_classes):
        class_indices = np.where(y_train == class_label)[0]
        selected_indices = np.random.choice(class_indices, num_examples_per_class, replace=False)

        start_idx = class_label * num_examples_per_class
        end_idx = start_idx + num_examples_per_class
        X_synthetic[start_idx:end_idx] = X_train[selected_indices]

        y_synthetic[start_idx:end_idx] = y_train[selected_indices].flatten()

    agent = RLAgent(state_size, num_classes)  # Initialize RL agent here

    # Data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    for epoch in range(100):
        synthetic_model = create_cnn_model(X_synthetic.shape[1:], num_classes)
        synthetic_model.fit(datagen.flow(X_synthetic, y_synthetic, batch_size=32), epochs=10, verbose=0)
        y_synthetic_pred = np.argmax(synthetic_model.predict(X_synthetic), axis=1)

        for i in range(num_examples):
            state = X_synthetic[i]
            action = y_synthetic_pred[i]
            reward = 1 if y_synthetic[i] == action else -1
            next_state = state
            done = True
            agent.remember(state, action, reward, next_state, done)

        agent.replay(32)

        train_loss, train_acc = synthetic_model.evaluate(X_synthetic, y_synthetic, verbose=0)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}")

        # Evaluate on original test set
        test_loss, test_acc = synthetic_model.evaluate(X_test, y_test, verbose=0)
        print(f"Epoch {epoch + 1}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")

    return X_synthetic, y_synthetic

# Main program to run the code
if __name__ == "__main__":
    # Define number of epochs for each dataset
    epochs_dict = {
        "mnist": 10,
        "fashionmnist": 10,
        "cifar10": 25,
        "cifar100": 25
    }

    # Ask user for input
    dataset_name = input("Enter dataset name (mnist, fashionmnist, cifar10, cifar100): ").lower()
    num_images_per_class = int(input("Enter number of images per class (1, 10, 50): "))

    # Set the number of epochs based on the dataset
    num_epochs = epochs_dict.get(dataset_name, 10)  # Default to 10 if dataset is not in the dictionary

    # Load the selected dataset
    (X_train, y_train), (X_test, y_test) = load_dataset(dataset_name)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Ensure correct data shapes
    if len(X_train.shape) == 3:  # For datasets like MNIST
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    # Select subset of training examples
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=1000, random_state=42)

    # Create RL agent and synthetic dataset
    state_size = np.prod(X_train.shape[1:])  # Flatten the state shape
    action_size = len(np.unique(y_train))
    X_synthetic, y_synthetic = create_synthetic_dataset(X_train_subset, y_train_subset, num_images_per_class, num_epochs)

    # Evaluate synthetic dataset performance
    synthetic_model = create_cnn_model(X_synthetic.shape[1:], action_size)
    synthetic_model.fit(X_synthetic, y_synthetic, epochs=num_epochs, batch_size=32, validation_split=0.2)

    # Evaluate on original test set
    test_loss, test_acc = synthetic_model.evaluate(X_test, y_test)
    print(f"Final Test accuracy on synthetic dataset: {test_acc:.4f}")
