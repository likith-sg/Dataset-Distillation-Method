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

# Defining various CNN models
def create_cnn_model(input_shape, num_classes, model_name):
    if model_name == "vgg11bn":
        return create_vgg11bn(input_shape, num_classes)
    elif model_name == "alexnet":
        return create_alexnet(input_shape, num_classes)
    elif model_name == "lenet":
        return create_lenet(input_shape, num_classes)
    elif model_name == "resnet":
        return create_resnet(input_shape, num_classes)
    elif model_name == "densenet":
        return create_densenet(input_shape, num_classes)
    elif model_name == "inceptionv3":
        return create_inceptionv3(input_shape, num_classes)
    else:
        raise ValueError("Model not supported.")

def create_vgg11bn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_alexnet(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(256, (5, 5), activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(384, (3, 3), activation='relu'),
        layers.Conv2D(384, (3, 3), activation='relu'),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lenet(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=input_shape),
        layers.AveragePooling2D(),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_resnet(input_shape, num_classes):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=input_shape)
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_densenet(input_shape, num_classes):
    base_model = tf.keras.applications.DenseNet121(include_top=False, weights=None, input_shape=input_shape)
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_inceptionv3(input_shape, num_classes):
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# RL Agent for reinforcement learning
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
        state = np.reshape(state, [1, self.state_size])  
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])  
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])  
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Function to create synthetic dataset using RL
def create_synthetic_dataset(X_train, y_train, num_examples_per_class, num_epochs, model_name):
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

    agent = RLAgent(np.prod(X_train.shape[1:]), num_classes)  # Initialize RL Agent

    # Data augmentation with ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}: X_synthetic shape = {X_synthetic.shape}")

        # Assuming grayscale images
        if len(X_synthetic.shape) == 3:
            X_synthetic = np.expand_dims(X_synthetic, axis=-1)  

        model = create_cnn_model(X_synthetic.shape[1:], num_classes, model_name)

        model.fit(datagen.flow(X_synthetic, y_synthetic, batch_size=32), epochs=5, validation_split=0.2, callbacks=[EarlyStopping(patience=2)])

        agent.remember(X_train, y_train, X_synthetic, y_synthetic, batch_size=32)

    return X_synthetic, y_synthetic

if __name__ == "__main__":
    epochs_dict = {
        "mnist": 10,
        "fashionmnist": 10,
        "cifar10": 25,
        "cifar100": 25
    }

    # Ask user for input
    dataset_name = input("Enter dataset name (mnist, fashionmnist, cifar10, cifar100): ").lower()
    cnn_model_name = input("Enter CNN model name (vgg11bn, alexnet, lenet, resnet, densenet, inceptionv3): ").lower()
    num_images_per_class = int(input("Enter number of images per class (1, 10, 50): "))

    # Set the number of epochs based on the dataset
    num_epochs = epochs_dict.get(dataset_name, 10)  

    # Load the selected dataset
    (X_train, y_train), (X_test, y_test) = load_dataset(dataset_name)

    # Normalize pixel values to be between 0 and 1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Ensure correct data shapes
    if len(X_train.shape) == 3:  
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    # Select subset of training examples
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=1000, random_state=42)

    # Create RL agent and synthetic dataset
    state_size = np.prod(X_train.shape[1:]) 
    action_size = len(np.unique(y_train))
    X_synthetic, y_synthetic = create_synthetic_dataset(X_train_subset, y_train_subset, num_images_per_class, num_epochs, cnn_model_name)

    # Evaluate synthetic dataset performance
    synthetic_model = create_cnn_model(X_synthetic.shape[1:], action_size, cnn_model_name)
    synthetic_model.fit(X_synthetic, y_synthetic, epochs=num_epochs, batch_size=32, validation_split=0.2)

    # Evaluate on original test set
    test_loss, test_acc = synthetic_model.evaluate(X_test, y_test)
    print(f"Final Test accuracy on synthetic dataset: {test_acc:.4f}")
