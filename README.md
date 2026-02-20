# MNIST Handwritten Digit Classification with TensorFlow

This project implements a basic Artificial Neural Network (ANN) to classify handwritten digits from the MNIST dataset using TensorFlow. It covers the entire pipeline from data preprocessing to model training, evaluation, and export.

## Project Structure

- [notebooks/mnist.ipynb](notebooks/mnist.ipynb): The main Jupyter notebook containing the code for:
    - Data loading and preprocessing (normalization and flattening).
    - Model architecture definition.
    - Custom training loop using `tf.GradientTape`.
    - Evaluation on the test set.
    - Visualization of model predictions.
    - Exporting the trained model.
- [model/](model/): Directory containing the exported TensorFlow `SavedModel`.
    - `saved_model.pb`: The model architecture and weights.
    - `fingerprint.pb`: Model fingerprint for version tracking.
- [src/](src/): Directory reserved for modularizing the code into Python scripts.

## Model Architecture

The model is a Multilayer Perceptron (MLP) with the following structure:
1. **Input Layer**: 784 neurons (28x28 pixel images flattened).
2. **Hidden Layer 1**: 128 neurons with Sigmoid activation.
3. **Hidden Layer 2**: 256 neurons with Sigmoid activation.
4. **Output Layer**: 10 neurons (representing digits 0-9) with Softmax activation.

## Training Details

- **Dataset**: MNIST Handwritten Digits.
- **Optimizer**: Stochastic Gradient Descent (SGD).
- **Loss Function**: Cross-Entropy.
- **Learning Rate**: 0.001.
- **Training Steps**: 3000.
- **Batch Size**: 256.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- `tensorflow`
- `numpy`
- `matplotlib`

### Usage

1. Open the [notebooks/mnist.ipynb](notebooks/mnist.ipynb) notebook.
2. Run the cells sequentially to train the model.
3. The model will be exported to the `./my_mnist_model` directory (mapped to [model/](model/) in the project structure).

## Results

After 3000 training steps, the model achieves a significant reduction in loss and a high classification accuracy on the test set. Detailed step-by-step logs and accuracy metrics can be found within the notebook.

**Training Accuracy - 97.6%** <br>
**Testing Accuracy - 95.9%**