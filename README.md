# Artificial Intelligence - Perceptron Activation Functions Classification 🤖

## Project Overview 📝

This repository contains the implementation of **Perceptron models** using various **activation functions**. The project explores the importance of activation functions in the training of neural networks, particularly for binary classification problems. The models are evaluated using a synthetic dataset, and we compare the performance of different perceptrons.

The following activation functions are implemented and used for training the models:

- **Sigmoid (Logistic) Activation Function** ⚡
- **Tanh (Hyperbolic Tangent) Activation Function** 🌐
- **ReLU (Rectified Linear Unit) Activation Function** 🔥
- **Leaky ReLU Activation Function** 🌊

## Questions and Answers 💡

### **Activation Functions**

Activation functions are essential for determining the output of a neural network. They map the resulting values between specific ranges (like 0 to 1, -1 to 1) and introduce non-linearity to the network.

#### **Types of Activation Functions:**
- **Linear Activation Function** ➡️
- **Non-linear Activation Functions** 🔄

The following are some widely used activation functions:

1. **Sigmoid (Logistic) Activation Function** 🟢
   ![image](https://github.com/user-attachments/assets/4df1a167-14ce-4377-a73f-073fbc00a286)
   - **Range**: (0, 1)
   - **Formula**:  
     \[
     \text{sigmoid}(x) = \frac{1}{1 + e^{-x}}
     \]
   - **Used for**: Binary classification problems. It is most commonly used in the output layer for binary classification tasks, where the goal is to produce a probability output.

3. **Tanh (Hyperbolic Tangent) Activation Function** 🌐
   - **Range**: (-1, 1)
   - **Formula**:  
     \[
     \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
     \]
   - **Used for**: Hidden layers of neural networks. It is similar to the sigmoid function but has a range of -1 to 1, making it better suited for deeper networks.

4. **ReLU (Rectified Linear Unit) Activation Function** 🔥
   - **Range**: [0, +∞)
   - **Formula**:  
     \[
     \text{ReLU}(x) = \max(0, x)
     \]
   - **Used for**: Hidden layers of deep networks. ReLU helps mitigate the vanishing gradient problem, enabling more efficient training and faster convergence.

5. **Leaky ReLU Activation Function** 🌊
   - **Range**: (-∞, +∞)
   - **Formula**:  
     \[
     \text{leaky\_ReLU}(x) = \max(0.01 \cdot x, x)
     \]
   - **Used for**: This variation of ReLU helps prevent the "dying ReLU" problem by allowing a small negative slope for negative inputs.

### **Perceptron Implementation**

We implemented different perceptrons using the activation functions mentioned above. Each perceptron was trained on a synthetic dataset, and the classification accuracy was calculated. The **Sigmoid (Logistic) Activation Function** achieved the highest accuracy of **0.928**.

---

### **Key Concepts in Machine Learning** 🧠

#### **1. Epochs** 🔄
In machine learning, an **epoch** refers to one complete pass through the entire training dataset. During each epoch, the model's parameters (weights and biases) are updated to minimize the error or loss function. Multiple epochs are often required to effectively train a model, allowing it to learn patterns from the data.

#### **2. Forward Propagation** ⬆️
**Forward propagation** is the process of passing input data through each layer of the neural network to generate predictions or outputs. It involves computing a weighted sum of the inputs at each layer, followed by passing the result through an activation function to generate the output.

#### **3. Backward Propagation** ⬇️
**Backward propagation** (or **backpropagation**) is the process of updating the model's parameters based on the error in the predictions. The gradient of the loss function with respect to the model parameters is computed using the chain rule of calculus, and the parameters are adjusted in the direction that minimizes the error.

#### **4. Bias in a Perceptron** ⚖️
The **bias** in a perceptron is an additional parameter added to the weighted sum of input features. It helps the model make better predictions when all input features are zero. The bias term alters the decision boundary and enables the model to fit more complex data interactions.

#### **5. XOR Problem and Multi-Layer Perceptron (MLP)** ❌
A **single-layer perceptron (SLP)** cannot solve the XOR (exclusive OR) problem because XOR is not linearly separable. A multi-layer perceptron (MLP) with one or more hidden layers is required to solve non-linear problems like XOR, as the additional layers allow the network to learn non-linear relationships.

---

## Project Structure 🏗️

- **`perceptron_activation_functions.py`**: Python script for implementing and training perceptron models with various activation functions.
- **`requirements.txt`**: File that lists the required dependencies for the project (e.g., NumPy, Scikit-learn).
- **`README.md`**: This file containing the project overview, setup instructions, and explanations.

## Installation 🔧

To run the project, make sure you have Python installed and then follow these steps:

### Clone the Repository 🛠️

```bash
git clone https://github.com/Bushra-Butt-17/Perceptron-Activation-Functions-Classification.git
cd Perceptron-Activation-Functions-Classification
```

### Install Dependencies 📦

```bash
pip install -r requirements.txt
```

### Usage 🚀
To train the perceptron models and evaluate their performance, run the following Python script:

```bash
python perceptron_activation_functions.py
```
---
The script will generate synthetic data, split it into training and testing sets, and train perceptron models using the specified activation functions. The classification accuracy and decision boundary plots for each model will be displayed.

## License 📝

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

