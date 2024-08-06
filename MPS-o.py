import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import InceptionV3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from stable_baselines3 import DQN, DDPG, TD3, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import optuna

# Load the dataset
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

# Defining CNN models
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

# RL Agent creation
class RLAgent:
    def __init__(self, state_size, action_size, algorithm, study=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.algorithm = algorithm

        if study is None:
            self.study = optuna.create_study(direction='maximize')
        else:
            self.study = study

        # Optimize RL hyperparameters
        self.optimize_hyperparameters()

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

    def optimize_hyperparameters(self):
        def objective(trial):
            self.epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.95, 0.999)
            self.learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

            # Initialize RL models
            if self.algorithm == "dqn":
                self.model = DQN('MlpPolicy', verbose=1, tensorboard_log="./dqn/", learning_rate=self.learning_rate)
            elif self.algorithm == "ddpg":
                self.model = DDPG('MlpPolicy', verbose=1, tensorboard_log="./ddpg/", learning_rate=self.learning_rate)
            elif self.algorithm == "td3":
                self.model = TD3('MlpPolicy', verbose=1, tensorboard_log="./td3/", learning_rate=self.learning_rate)
            elif self.algorithm == "ppo":
                self.model = PPO('MlpPolicy', verbose=1, tensorboard_log="./ppo/", learning_rate=self.learning_rate)
            else:
                raise ValueError("RL algorithm not supported.")

            # Train RL model for fixed iterations
            self.model.learn(total_timesteps=1000)

            # Evaluate RL model
            mean_reward, _ = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10)

            return mean_reward

        self.study.optimize(objective, n_trials=10)

        # Select best hyperparameters
        self.epsilon_decay = self.study.best_params['epsilon_decay']
        self.learning_rate = self.study.best_params['learning_rate']

        # Initialize RL model with best hyperparameters
        if self.algorithm == "dqn":
            self.model = DQN('MlpPolicy', verbose=1, tensorboard_log="./dqn/", learning_rate=self.learning_rate)
        elif self.algorithm == "ddpg":
            self.model = DDPG('MlpPolicy', verbose=1, tensorboard_log="./ddpg/", learning_rate=self.learning_rate)
        elif self.algorithm == "td3":
            self.model = TD3('MlpPolicy', verbose=1, tensorboard_log="./td3/", learning_rate=self.learning_rate)
        elif self.algorithm == "ppo":
            self.model = PPO('MlpPolicy', verbose=1, tensorboard_log="./ppo/", learning_rate=self.learning_rate)
        else:
            raise ValueError("RL algorithm not supported.")

    def get_study(self):
        return self.study

# Function to create synthetic dataset using RL agent
def create_synthetic_dataset(X_train, y_train, num_images_per_class, num_epochs, cnn_model_name, rl_algorithm):
    rl_agent = RLAgent(X_train.shape[1:], len(np.unique(y_train)), rl_algorithm)

    # Data augmentation with ImageDataGenerator
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(X_train)

    # Train RL agent to generate synthetic dataset
    for epoch in range(num_epochs):
        total_reward = 0
        for i in range(len(X_train)):
            state = X_train[i]
            action = rl_agent.act(state)
            next_state = datagen.random_transform(state)
            reward = calculate_reward(state, next_state, y_train[i])
            rl_agent.remember(state, action, reward, next_state, False)
            total_reward += reward
            X_train[i] = next_state

        rl_agent.replay(batch_size=32)
        avg_reward = total_reward / len(X_train)
        print(f"Epoch {epoch + 1}/{num_epochs} - Synthetic dataset generation complete. Avg Reward: {avg_reward:.2f}")

    return X_train, y_train

# Reward function design
def calculate_reward(state, next_state, label, model, original_data):
    state = state / 255.0
    next_state = next_state / 255.0
    similarity = np.mean(np.abs(state - next_state))  
    similarity_reward = 1.0 / (1.0 + similarity) 
    synthetic_data = np.expand_dims(next_state, axis=0) 
    synthetic_label = np.argmax(label)
    predictions = model.predict(synthetic_data)
    predicted_label = np.argmax(predictions[0])
    classification_reward = 1.0 if predicted_label == synthetic_label else 0.0

    # Final reward
    total_reward = 0.5 * similarity_reward + 0.5 * classification_reward

    return total_reward

# Main 
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
    rl_algorithm = input("Enter RL algorithm (dqn, ddpg, td3, ppo): ").lower()
    num_images_per_class = int(input("Enter number of images per class (1, 10, 50): "))

    # Minimum number of epochs
    num_epochs = epochs_dict.get(dataset_name, 10)

    # Load the selected dataset
    (X_train, y_train), (_, _) = load_dataset(dataset_name)

    # Normalize pixel values to be between 0 and 1
    X_train = X_train.astype('float32') / 255.0

    # Ensure correct data shapes for grayscale and RGB images
    if len(X_train.shape) < 4:
        X_train = np.expand_dims(X_train, axis=-1)

    # Select subset of training examples
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=num_images_per_class * len(np.unique(y_train)))

    # Create CNN model based on user input
    cnn_model = create_cnn_model(X_train.shape[1:], len(np.unique(y_train)), cnn_model_name)

    # Hyperparameter tuning for CNN model
    cnn_model.fit(X_train_subset, y_train_subset, epochs=num_epochs, batch_size=32, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

    # Evaluate CNN model on original test set
    _, acc = cnn_model.evaluate(X_train_subset, y_train_subset, verbose=0)
    print(f"Accuracy of CNN model on original dataset: {acc * 100:.2f}%")

    # Create synthetic dataset using RL
    X_synthetic, y_synthetic = create_synthetic_dataset(X_train_subset, y_train_subset, num_images_per_class, num_epochs, cnn_model_name, rl_algorithm)

    # Evaluate synthetic dataset performance using original test set
    _, acc = cnn_model.evaluate(X_synthetic, y_synthetic, verbose=0)
    print(f"Accuracy of CNN model on synthetic dataset: {acc * 100:.2f}%")
