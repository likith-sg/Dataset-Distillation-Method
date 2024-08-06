import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import random
from PIL import Image
import os
from stable_baselines3 import DQN, DDPG, TD3, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

dataset_specs = {
    'mnist': {'input_shape': (28, 28, 1), 'num_classes': 10},
    'fashionmnist': {'input_shape': (28, 28, 1), 'num_classes': 10},
    'cifar10': {'input_shape': (32, 32, 3), 'num_classes': 10},
    'cifar100': {'input_shape': (32, 32, 3), 'num_classes': 100}
}

model_input_shape = {
    "vgg11bn": (224, 224, 3),
    "alexnet": (227, 227, 3),
    "lenet": (32, 32, 1),
    "resnet": (224, 224, 3),
    "densenet": (224, 224, 3),
    "inceptionv3": (299, 299, 3)
}

def load_dataset(dataset_name):
    datasets = {
        "mnist": tf.keras.datasets.mnist,
        "fashionmnist": tf.keras.datasets.fashion_mnist,
        "cifar10": tf.keras.datasets.cifar10,
        "cifar100": tf.keras.datasets.cifar100
    }
    if dataset_name in datasets:
        return datasets[dataset_name].load_data()
    else:
        raise ValueError("Dataset not supported.")
        
def resize_images(X_train, target_shape):
    resized_images = []
    for image in X_train:
        img = Image.fromarray((image * 255).astype(np.uint8))
        img = img.resize(target_shape[:2], Image.ANTIALIAS)
        resized_images.append(np.array(img, dtype=np.float32) / 255.0)
    return np.array(resized_images, dtype=np.float32)

def resize_images_in_batches(image_paths, target_shape, batch_size=1000):
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    resized_images = []
    
    for i in range(num_batches):
        batch_paths = image_paths[i * batch_size:(i + 1) * batch_size]
        batch_images = []
        for path in batch_paths:
            img = Image.open(path)
            img = img.resize(target_shape[:2], Image.ANTIALIAS)
            batch_images.append(np.array(img, dtype=np.float32) / 255.0)
        resized_images.extend(batch_images)
    
    return np.array(resized_images, dtype=np.float32)

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
        self.study = study or optuna.create_study(direction='maximize')
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

            self.model = {
                "dqn": DQN,
                "ddpg": DDPG,
                "td3": TD3,
                "ppo": PPO
            }[self.algorithm]('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log=f"./{self.algorithm}/", learning_rate=self.learning_rate)

            self.model.learn(total_timesteps=1000)
            mean_reward, _ = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10)
            return mean_reward

        self.study.optimize(objective, n_trials=10)
        self.epsilon_decay = self.study.best_params['epsilon_decay']
        self.learning_rate = self.study.best_params['learning_rate']
        self.model = {
            "dqn": DQN,
            "ddpg": DDPG,
            "td3": TD3,
            "ppo": PPO
        }[self.algorithm]('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log=f"./{self.algorithm}/", learning_rate=self.learning_rate)

    def get_study(self):
        return self.study

def create_synthetic_dataset(X_train, y_train, num_images_per_class, num_epochs, cnn_model_name, rl_algorithm):
    rl_agent = RLAgent(X_train.shape[1:], len(np.unique(y_train)), rl_algorithm)
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(X_train)

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

def calculate_reward(state, next_state, label, model, original_data):
    state = state / 255.0
    next_state = next_state / 255.0
    similarity = np.mean(np.abs(state - next_state))  
    similarity_reward = 1.0 / (1.0 + similarity) 
    synthetic_data = np.expand_dims(next_state, axis=0) 
    predictions = model.predict(synthetic_data)
    predicted_label = np.argmax(predictions[0])
    classification_reward = 1.0 if predicted_label == label else 0.0
    total_reward = 0.5 * similarity_reward + 0.5 * classification_reward
    return total_reward

def visualize_samples(X_synthetic, y_synthetic, num_samples=5):
    fig, axes = plt.subplots(len(np.unique(y_synthetic)), num_samples, figsize=(num_samples, len(np.unique(y_synthetic))))
    fig.suptitle('Synthetic Data Samples', fontsize=16)

    for cls in np.unique(y_synthetic):
        class_samples = X_synthetic[y_synthetic == cls][:num_samples]
        for i, sample in enumerate(class_samples):
            axes[cls, i].imshow(sample)
            axes[cls, i].axis('off')
            if i == 0:
                axes[cls, i].set_title(f'Class {cls}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    epochs_dict = {
        "mnist": 10,
        "fashionmnist": 10,
        "cifar10": 25,
        "cifar100": 25
    }

    dataset_name = input("Enter dataset name (mnist, fashionmnist, cifar10, cifar100): ").lower()
    cnn_model_name = input("Enter CNN model name (vgg11bn, alexnet, lenet, resnet, densenet, inceptionv3): ").lower()
    rl_algorithm = input("Enter RL algorithm (dqn, ddpg, td3, ppo): ").lower()
    num_images_per_class = int(input("Enter number of images per class (1, 10, 50): "))

    num_epochs = epochs_dict.get(dataset_name, 10)

    (X_train, y_train), (_, _) = load_dataset(dataset_name)
    if len(X_train.shape) < 4:
        X_train = np.expand_dims(X_train, axis=-1)
    input_shape = model_input_shape[cnn_model_name]
    X_train = resize_images(X_train, input_shape)

    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=num_images_per_class * len(np.unique(y_train)))
    cnn_model = create_cnn_model(X_train.shape[1:], len(np.unique(y_train)), cnn_model_name)

    cnn_model.fit(X_train_subset, y_train_subset, epochs=num_epochs, batch_size=32, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

    _, acc = cnn_model.evaluate(X_train_subset, y_train_subset, verbose=0)
    print(f"Accuracy of CNN model on original dataset: {acc * 100:.2f}%")

    X_synthetic, y_synthetic = create_synthetic_dataset(X_train_subset, y_train_subset, num_images_per_class, num_epochs, cnn_model_name, rl_algorithm)

    _, acc = cnn_model.evaluate(X_synthetic, y_synthetic, verbose=0)
    print(f"Accuracy of CNN model on synthetic dataset: {acc * 100:.2f}%")

    visualize_samples(X_synthetic, y_synthetic)

    dataset_name_unseen = input("Enter completely unseen dataset name (mnist, fashionmnist, cifar10, cifar100): ").lower()
    (X_unseen, y_unseen), (_, _) = load_dataset(dataset_name_unseen)
    if len(X_unseen.shape) < 4:
        X_unseen = np.expand_dims(X_unseen, axis=-1)
    X_unseen = resize_images(X_unseen, input_shape)

    _, acc_unseen = cnn_model.evaluate(X_unseen, y_unseen, verbose=0)
    print(f"Accuracy of CNN model on unseen dataset: {acc_unseen * 100:.2f}%")
