# MPS
# Dataset Distillation Method
These are machine learning models developed by me, featuring custom architectures. The synthetic datasets produced are primarily intended for classification tasks. Below is a detailed breakdown of each model and their functionalities:

## **MPS1**
This script initializes with 1000 randomly selected training examples from the MNIST dataset and creates a synthetic dataset.

**Key Components:**
- **Importing Libraries**:
  - TensorFlow, NumPy, and scikit-learn for various functionalities including neural networks and data manipulation.

- **Loading and Preprocessing MNIST Data**:
  - `tf.keras.datasets.mnist.load_data()` loads the MNIST dataset.
  - Data is normalized by dividing pixel values by 255.0.

- **Subset Selection**:
  - A subset of 1000 training examples is selected from the original MNIST dataset.

- **CNN Model Definition (`create_cnn_model`)**:
  - A simple CNN architecture with three convolutional layers, max-pooling layers, flattening, and dense layers. Uses ReLU activations for hidden layers and softmax for the output layer.

- **Training the CNN**:
  - Trained on the subset of the MNIST data.

- **Data Augmentation**:
  - Uses `ImageDataGenerator` for data augmentation to improve model generalization.

- **Reinforcement Learning (RL) Agent**:
  - An RL agent with a neural network model using dense layers. Includes methods for remembering experiences, acting based on state, and replaying experiences for learning.

- **Synthetic Dataset Creation**:
  - `create_synthetic_dataset` generates synthetic data using the RL agent. It initializes with random examples and optimizes the dataset using the RL agent.

- **Testing and Evaluation**:
  - Trains a CNN model on the synthetic dataset and evaluates performance on the MNIST test set.

**Performance**: Achieved a test accuracy of 84% on the MNIST dataset.

## **MPS1_with_GAN**
An extension of MPS1 incorporating Generative Adversarial Networks (GANs) to generate synthetic data.

**Key Components:**
- **GAN Architecture**:
  - Defines a GAN with a generator and discriminator network. The generator creates synthetic images, while the discriminator distinguishes between real and synthetic images.
  - Both networks are trained together: the generator aims to fool the discriminator, and the discriminator improves its ability to classify images as real or fake.

- **Training the GAN**:
  - Trained using the MNIST dataset. The generator creates synthetic data, and the discriminator is trained to differentiate between real and synthetic images.

- **Synthetic Data Generation**:
  - Uses the trained GAN to generate synthetic data, which is then combined with data from the RL agent.

- **Model Training and Evaluation**:
  - Trains a CNN model on the combined synthetic dataset from GAN and RL. Model checkpoints are used to save the best-performing model, which is evaluated on the MNIST test set.

**Performance**: Achieved a test accuracy of 77% on the MNIST dataset.

## **MPS2**
A more generalized script that supports multiple datasets and includes early stopping for training.

**Key Components:**
- **Dataset Loading**:
  - `load_dataset` function allows loading various datasets (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100) based on a given name.

- **CNN Model Definition**:
  - CNN model is defined similarly to previous scripts but with parameterized input shape and number of classes.

- **RL Agent**:
  - Similar to MPS1 with modifications:
    - Uses `softmax` activation in the final layer.
    - Uses `categorical_crossentropy` as the loss function.

- **Early Stopping**:
  - Implements early stopping to halt training when model performance ceases to improve on the validation set, preventing overfitting.

**Performance**:
  - MNIST: Achieved a 64% test accuracy.
  - CIFAR-10: Achieved a 31% test accuracy.
  - FashionMNIST: Achieved a 58% test accuracy.

**Comments**:
  - Provides detailed metrics for each epoch, including Training Loss, Testing Loss, Training Accuracy, and Testing Accuracy.

## NOTE : I have used Gradient Matching as the benchmark algorithm for comparison, with all developed algorithms evaluated against it.

## **Comparison with Gradient Matching Algorithm**

| **Metric**          | **MPS2**                              | **Gradient Matching**                     |
|---------------------|---------------------------------------|-------------------------------------------|
| **MNIST Test Accuracy**  | 64%                                   | 97%                                       |
| **FashionMNIST Test Accuracy** | 58%                                   | 83%                                       |
| **CIFAR-10 Test Accuracy** | 31%                                   | 51%                                       |
| **Computational Load**    | Low (Integrated GPU)                  | High                                      |
| **Training Time**         | 20 minutes                            | 4 hours                                   |

- **Gradient Matching**:
  - **Code Link**: [Gradient Matching Code](https://github.com/likith-sg/Gradient-Matching)
  - **Performance**: Achieves a test accuracy of 97% on MNIST, 83% on FashionMNIST and 51% on CIFAR-10.
  - **Computational Load**: High computational load with a training time of 4 hours.

- **MPS2**:
  - **Performance**:
    - MNIST: 64% test accuracy.
    - CIFAR-10: 31% test accuracy.
    - FashionMNIST: 58% test accuracy.
  - **Computational Load**: Low, with a training time of just 20 minutes.
  - **Compatibility**: Can run on any machine with an integrated GPU, no need for a dedicated GPU.

## **Code and Repository Information**
- **Code**: Includes comments to guide users through the synthetic data generation process. Modifications to the code are prohibited.
- **Repository**: Download or clone the repository from [Dataset Distillation Method](https://github.com/likith-sg/Dataset-Distillation-Method.git).

**IMPORTANT:**

## MPS-o and MPS-5o are still under development and have high computational load. It is recommended NOT to run these models until they are fully developed. 

## **Forking the repository is prohibited.**

## **MPS-o and MPS-5o**
These models are collaboratively being developed by me and the CVCSI research center at RV University. We are currently exploring different CNN models and RL deep learning architectures.

### **Contact**:
If you have any questions or need further assistance, please feel free to reach out.

## Happy coding!

**How to Access**:
- **Download as ZIP**: Click on the "Code" button on the repository page and select "Download ZIP" to get the entire repository in a ZIP file.
- **Clone the Repository**:
  ```bash
  git clone https://github.com/likith-sg/Dataset-Distillation-Method.git
