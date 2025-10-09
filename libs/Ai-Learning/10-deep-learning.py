"""
10-deep-learning.py

Deep Learning & Advanced Neural Networks
----------------------------------------
Explore advanced deep learning techniques including CNNs, RNNs, and modern
architectures. Learn about transfer learning and advanced optimization.

What you'll learn
-----------------
1) Convolutional Neural Networks (CNNs) for image processing
2) Recurrent Neural Networks (RNNs) for sequential data
3) Transfer learning and pre-trained models
4) Advanced optimization techniques
5) Modern architectures (ResNet, LSTM, GRU)

Key Concepts
------------
- Convolutional layers and filters
- Pooling and dropout layers
- Sequence modeling with RNNs
- Transfer learning with pre-trained models
- Advanced optimizers (Adam, RMSprop)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

def cnn_for_image_classification():
    """Build CNN for image classification"""
    print("=== CNN for Image Classification ===")
    
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Build CNN model
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("CNN Model Architecture:")
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.3f}")
    
    # Visualize training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('CNN Training Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('CNN Training Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_training.png', dpi=300, bbox_inches='tight')
    print("CNN training visualization saved as 'cnn_training.png'")
    plt.show()
    
    # Visualize some predictions
    predictions = model.predict(X_test[:10])
    predicted_classes = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        row, col = i // 5, i % 5
        axes[row, col].imshow(X_test[i].reshape(28, 28), cmap='gray')
        axes[row, col].set_title(f'True: {y_test[i]}, Pred: {predicted_classes[i]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('cnn_predictions.png', dpi=300, bbox_inches='tight')
    print("CNN predictions visualization saved as 'cnn_predictions.png'")
    plt.show()
    
    return model, history, test_accuracy

def rnn_for_sequence_modeling():
    """Build RNN for sequence modeling"""
    print("\n=== RNN for Sequence Modeling ===")
    
    # Generate synthetic sequence data
    def generate_sequences(n_samples=1000, seq_length=20, n_features=1):
        X = []
        y = []
        for _ in range(n_samples):
            # Generate random sequence
            sequence = np.random.randn(seq_length, n_features)
            # Target is the sum of the sequence
            target = np.sum(sequence)
            X.append(sequence)
            y.append(target)
        return np.array(X), np.array(y)
    
    # Generate data
    X, y = generate_sequences(n_samples=1000, seq_length=20, n_features=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training sequences shape: {X_train.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    
    # Build RNN model
    model = models.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(20, 1)),
        layers.LSTM(50, return_sequences=False),
        layers.Dense(25, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    print("RNN Model Architecture:")
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MAE: {test_mae:.3f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training history
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('RNN Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Predictions vs actual
    axes[1].scatter(y_test, y_pred, alpha=0.6)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title('RNN Predictions vs Actual')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rnn_results.png', dpi=300, bbox_inches='tight')
    print("RNN results visualization saved as 'rnn_results.png'")
    plt.show()
    
    return model, history, test_mae

def transfer_learning_demo():
    """Demonstrate transfer learning with pre-trained models"""
    print("\n=== Transfer Learning Demo ===")
    
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Use only first 1000 samples for demo (to speed up training)
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_test = X_test[:200]
    y_test = y_test[:200]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Load pre-trained VGG16 model (without top layers)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Transfer Learning Model Architecture:")
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.3f}")
    
    # Visualize training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Transfer Learning - Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('transfer_learning.png', dpi=300, bbox_inches='tight')
    print("Transfer learning visualization saved as 'transfer_learning.png'")
    plt.show()
    
    return model, history, test_accuracy

def advanced_optimizers():
    """Compare different optimizers"""
    print("\n=== Advanced Optimizers Comparison ===")
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=5, n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define optimizers
    optimizers_config = {
        'SGD': optimizers.SGD(learning_rate=0.01),
        'SGD with Momentum': optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'Adam': optimizers.Adam(learning_rate=0.001),
        'RMSprop': optimizers.RMSprop(learning_rate=0.001),
        'Adagrad': optimizers.Adagrad(learning_rate=0.01)
    }
    
    results = {}
    
    for opt_name, optimizer in optimizers_config.items():
        # Build model
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with specific optimizer
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
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
        results[opt_name] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'history': history
        }
        
        print(f"{opt_name:20}: Test Accuracy = {test_accuracy:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Test accuracy comparison
    names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in names]
    
    axes[0, 0].bar(names, accuracies, alpha=0.8)
    axes[0, 0].set_title('Test Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training curves
    for name, result in results.items():
        history = result['history']
        axes[0, 1].plot(history.history['val_accuracy'], label=name, alpha=0.8)
    
    axes[0, 1].set_title('Validation Accuracy During Training')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss curves
    for name, result in results.items():
        history = result['history']
        axes[1, 0].plot(history.history['val_loss'], label=name, alpha=0.8)
    
    axes[1, 0].set_title('Validation Loss During Training')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Convergence speed
    convergence_epochs = []
    for name, result in results.items():
        history = result['history']
        # Find epoch where validation accuracy stabilizes
        val_acc = history.history['val_accuracy']
        for i in range(10, len(val_acc)):
            if abs(val_acc[i] - val_acc[i-5]) < 0.001:
                convergence_epochs.append(i)
                break
        else:
            convergence_epochs.append(len(val_acc))
    
    axes[1, 1].bar(names, convergence_epochs, alpha=0.8, color='orange')
    axes[1, 1].set_title('Convergence Speed (Epochs)')
    axes[1, 1].set_ylabel('Epochs to Converge')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizers_comparison.png', dpi=300, bbox_inches='tight')
    print("Optimizers comparison saved as 'optimizers_comparison.png'")
    plt.show()
    
    return results

def modern_architectures():
    """Demonstrate modern neural network architectures"""
    print("\n=== Modern Neural Network Architectures ===")
    
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normalize and use subset for demo
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Use smaller subset for demo
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    X_test = X_test[:500]
    y_test = y_test[:500]
    
    # Define modern architectures
    architectures = {
        'Simple CNN': models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ]),
        'ResNet-like': models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(10, activation='softmax')
        ]),
        'DenseNet-like': models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.Conv2D(32, (1, 1), activation='relu'),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Concatenate(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (1, 1), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Concatenate(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(10, activation='softmax')
        ])
    }
    
    results = {}
    
    for name, model in architectures.items():
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n{name} Architecture:")
        print(f"Parameters: {model.count_params():,}")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=64,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        results[name] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'history': history,
            'params': model.count_params()
        }
        
        print(f"Test Accuracy: {test_accuracy:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy comparison
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
    for name, result in results.items():
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
    plt.savefig('modern_architectures.png', dpi=300, bbox_inches='tight')
    print("Modern architectures comparison saved as 'modern_architectures.png'")
    plt.show()
    
    return results

def main():
    """Main function to run all deep learning demonstrations"""
    print("🚀 Deep Learning & Advanced Neural Networks")
    print("=" * 50)
    
    # CNN for image classification
    cnn_for_image_classification()
    
    # RNN for sequence modeling
    rnn_for_sequence_modeling()
    
    # Transfer learning
    transfer_learning_demo()
    
    # Advanced optimizers
    advanced_optimizers()
    
    # Modern architectures
    modern_architectures()
    
    print("\n" + "=" * 50)
    print("✅ Lesson 10 Complete!")
    print("Next: Learn computer vision and image processing techniques")
    print("Key takeaway: Deep learning enables solving complex problems with large datasets")

if __name__ == "__main__":
    main()
