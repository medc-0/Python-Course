"""
09-neural-networks-basics.py

Neural Networks Fundamentals
----------------------------
Learn the basics of neural networks, from perceptrons to multi-layer networks.
Understand forward propagation, backpropagation, and activation functions.

What you'll learn
-----------------
1) Perceptron and single-layer neural networks
2) Multi-layer perceptrons (MLPs)
3) Activation functions and their properties
4) Forward and backward propagation
5) Building neural networks with TensorFlow/Keras

Key Concepts
------------
- Artificial neurons and weights
- Activation functions (sigmoid, ReLU, tanh)
- Loss functions and optimization
- Gradient descent and backpropagation
- Overfitting and regularization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

def perceptron_implementation():
    """Implement a simple perceptron from scratch"""
    print("=== Perceptron Implementation ===")
    
    class Perceptron:
        def __init__(self, learning_rate=0.01, n_iterations=1000):
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations
            self.weights = None
            self.bias = None
            self.errors = []
        
        def fit(self, X, y):
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0
            
            for _ in range(self.n_iterations):
                error_count = 0
                for i in range(n_samples):
                    linear_output = np.dot(X[i], self.weights) + self.bias
                    prediction = self.activate(linear_output)
                    
                    if prediction != y[i]:
                        self.weights += self.learning_rate * y[i] * X[i]
                        self.bias += self.learning_rate * y[i]
                        error_count += 1
                
                self.errors.append(error_count)
                if error_count == 0:
                    break
        
        def activate(self, x):
            return np.where(x >= 0, 1, -1)
        
        def predict(self, X):
            linear_output = np.dot(X, self.weights) + self.bias
            return self.activate(linear_output)
    
    # Generate linearly separable data
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                              n_redundant=0, n_clusters_per_class=1, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert to -1, 1 for perceptron
    
    # Train perceptron
    perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
    perceptron.fit(X, y)
    
    # Make predictions
    y_pred = perceptron.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"Perceptron accuracy: {accuracy:.3f}")
    print(f"Final weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias}")
    print(f"Converged in {len(perceptron.errors)} iterations")
    
    # Visualize decision boundary
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='s', label='Class -1')
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('perceptron_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return perceptron, accuracy

def activation_functions_demo():
    """Demonstrate different activation functions"""
    print("\n=== Activation Functions Demo ===")
    
    # Define activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def relu(x):
        return np.maximum(0, x)
    
    def tanh(x):
        return np.tanh(x)
    
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    # Generate x values
    x = np.linspace(-5, 5, 100)
    
    # Calculate activation function values
    y_sigmoid = sigmoid(x)
    y_relu = relu(x)
    y_tanh = tanh(x)
    y_leaky_relu = leaky_relu(x)
    
    # Plot activation functions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(x, y_sigmoid, 'b-', linewidth=2)
    axes[0, 0].set_title('Sigmoid')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('σ(x)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[0, 1].plot(x, y_relu, 'r-', linewidth=2)
    axes[0, 1].set_title('ReLU')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('ReLU(x)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[1, 0].plot(x, y_tanh, 'g-', linewidth=2)
    axes[1, 0].set_title('Tanh')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('tanh(x)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[1, 1].plot(x, y_leaky_relu, 'm-', linewidth=2)
    axes[1, 1].set_title('Leaky ReLU')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Leaky ReLU(x)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    print("Activation functions visualization saved as 'activation_functions.png'")
    plt.show()
    
    # Demonstrate gradient properties
    print("\nActivation Function Properties:")
    print("-" * 40)
    print("Sigmoid:  Range (0,1), Smooth gradient, Vanishing gradient problem")
    print("ReLU:     Range [0,∞), Simple, Dead neuron problem")
    print("Tanh:     Range (-1,1), Zero-centered, Vanishing gradient")
    print("Leaky ReLU: Range (-∞,∞), Prevents dead neurons")

def mlp_with_keras():
    """Build Multi-Layer Perceptron with Keras"""
    print("\n=== Multi-Layer Perceptron with Keras ===")
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=5, n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build MLP model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Architecture:")
    model.summary()
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.3f}")
    print(f"Test Loss: {test_loss:.3f}")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_training_history.png', dpi=300, bbox_inches='tight')
    print("MLP training history saved as 'mlp_training_history.png'")
    plt.show()
    
    return model, history, test_accuracy

def neural_network_regularization():
    """Demonstrate regularization techniques in neural networks"""
    print("\n=== Neural Network Regularization ===")
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=5, n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define different models
    models = {
        'No Regularization': keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ]),
        'Dropout': keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ]),
        'L2 Regularization': keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],),
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dense(32, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dense(1, activation='sigmoid')
        ]),
        'Early Stopping': keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
    }
    
    results = {}
    
    for name, model in models.items():
        # Compile model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train with different callbacks
        if name == 'Early Stopping':
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            history = model.fit(
                X_train_scaled, y_train,
                epochs=200,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            history = model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        results[name] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'history': history
        }
        
        print(f"{name:20}: Test Accuracy = {test_accuracy:.3f}, Test Loss = {test_loss:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(results.items()):
        history = result['history']
        
        # Plot training and validation accuracy
        axes[i].plot(history.history['accuracy'], label='Training', alpha=0.8)
        axes[i].plot(history.history['val_accuracy'], label='Validation', alpha=0.8)
        axes[i].set_title(f'{name}\nTest Acc: {result["test_accuracy"]:.3f}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Accuracy')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
    print("Regularization comparison saved as 'regularization_comparison.png'")
    plt.show()
    
    return results

def neural_network_architectures():
    """Compare different neural network architectures"""
    print("\n=== Neural Network Architectures Comparison ===")
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=5, n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define different architectures
    architectures = {
        'Shallow (2 layers)': keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(1, activation='sigmoid')
        ]),
        'Medium (3 layers)': keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ]),
        'Deep (5 layers)': keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ]),
        'Wide (more neurons)': keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
    }
    
    results = {}
    
    for name, model in architectures.items():
        # Compile model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        results[name] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'history': history,
            'params': model.count_params()
        }
        
        print(f"{name:20}: Test Acc = {test_accuracy:.3f}, Parameters = {model.count_params():,}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Test accuracy comparison
    names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in names]
    params = [results[name]['params'] for name in names]
    
    axes[0, 0].bar(names, accuracies, alpha=0.8)
    axes[0, 0].set_title('Test Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Parameters comparison
    axes[0, 1].bar(names, params, alpha=0.8, color='orange')
    axes[0, 1].set_title('Number of Parameters')
    axes[0, 1].set_ylabel('Parameters')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training curves
    for i, (name, result) in enumerate(results.items()):
        history = result['history']
        axes[1, 0].plot(history.history['val_accuracy'], label=name, alpha=0.8)
    
    axes[1, 0].set_title('Validation Accuracy During Training')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy vs Parameters
    axes[1, 1].scatter(params, accuracies, s=100, alpha=0.8)
    for i, name in enumerate(names):
        axes[1, 1].annotate(name, (params[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 1].set_xlabel('Number of Parameters')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Accuracy vs Model Complexity')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
    print("Architecture comparison saved as 'architecture_comparison.png'")
    plt.show()
    
    return results

def main():
    """Main function to run all neural network demonstrations"""
    print("Neural Networks Fundamentals")
    print("=" * 50)
    
    # Perceptron implementation
    perceptron_implementation()
    
    # Activation functions
    activation_functions_demo()
    
    # MLP with Keras
    mlp_with_keras()
    
    # Regularization techniques
    neural_network_regularization()
    
    # Architecture comparison
    neural_network_architectures()
    
    print("\n" + "=" * 50)
    print("Lesson 9 Complete!")
    print("Next: Learn deep learning and advanced neural network techniques")
    print("Key takeaway: Neural networks are powerful function approximators that can learn complex patterns")

if __name__ == "__main__":
    main()
