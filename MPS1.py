#Importing Libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import random

#Loading MNIST Dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train[..., np.newaxis] / 255.0
X_test = X_test[..., np.newaxis] / 255.0

#Selecting 1000 training examples
X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=1000, random_state=42)

#Defining the CNN
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

#Training on original subset
original_model = create_cnn_model()
original_model.fit(X_train_subset, y_train_subset, epochs=5, batch_size=32, validation_split=0.2)

#Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

#A NN with hidden layers having 24 neurons each, using epsilon for exploring and exploiting, training in batches
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

#Creation of RL agent
state_size = X_train.shape[1] * X_train.shape[2]
action_size = 10  # Number of labels
agent = RLAgent(state_size, action_size)

#Function to create a synthetic dataset
def create_synthetic_dataset(X_train, y_train, num_examples):
    #Placeholder for synthetic dataset
    X_synthetic = np.zeros((num_examples, 28, 28, 1))
    y_synthetic = np.zeros((num_examples,), dtype=int)

    #Initialize with random selection from original data
    indices = np.random.choice(len(X_train), num_examples, replace=False)
    X_synthetic = X_train[indices]
    y_synthetic = y_train[indices]

    #Optimize the synthetic dataset using RL agent
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

#Creation of synthetic dataset with a defined output label of 437
X_synthetic, y_synthetic = create_synthetic_dataset(X_train_subset, y_train_subset, 437)

#Testing a different model for performance evaluation
synthetic_model = create_cnn_model()
synthetic_model.fit(X_synthetic, y_synthetic, epochs=5, batch_size=32, validation_split=0.2)

#Printing the output
test_loss, test_acc = synthetic_model.evaluate(X_test, y_test)
print(f"Test accuracy on synthetic dataset: {test_acc}")
