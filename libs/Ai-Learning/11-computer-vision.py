"""
11-computer-vision.py

Computer Vision & Image Processing
----------------------------------
Learn computer vision techniques for image analysis, object detection, and
image generation. Explore both traditional and deep learning approaches.

What you'll learn
-----------------
1) Image preprocessing and augmentation techniques
2) Feature extraction from images
3) Object detection and segmentation
4) Image classification with CNNs
5) Generative models for image creation

Key Concepts
------------
- Image filtering and enhancement
- Convolutional operations
- Transfer learning for vision tasks
- Data augmentation strategies
- Object detection frameworks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_sample_images
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import warnings
warnings.filterwarnings('ignore')

def image_preprocessing_techniques():
    """Demonstrate various image preprocessing techniques"""
    print("=== Image Preprocessing Techniques ===")
    
    # Load sample images
    try:
        images = load_sample_images()
        sample_image = images.images[0]  # Use the first image
    except:
        # Create a synthetic image if sample images not available
        sample_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    print(f"Original image shape: {sample_image.shape}")
    
    # Convert to PIL Image for processing
    pil_image = Image.fromarray(sample_image)
    
    # Apply various preprocessing techniques
    processed_images = {
        'Original': sample_image,
        'Grayscale': np.array(pil_image.convert('L')),
        'Blurred': np.array(pil_image.filter(ImageFilter.BLUR)),
        'Sharpened': np.array(pil_image.filter(ImageFilter.SHARPEN)),
        'Edge Enhanced': np.array(pil_image.filter(ImageFilter.EDGE_ENHANCE)),
        'Contrast Enhanced': np.array(ImageEnhance.Contrast(pil_image).enhance(2.0)),
        'Brightness Enhanced': np.array(ImageEnhance.Brightness(pil_image).enhance(1.5)),
        'Color Enhanced': np.array(ImageEnhance.Color(pil_image).enhance(1.5))
    }
    
    # Visualize preprocessing results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (name, img) in enumerate(processed_images.items()):
        if len(img.shape) == 2:  # Grayscale
            axes[i].imshow(img, cmap='gray')
        else:  # Color
            axes[i].imshow(img)
        axes[i].set_title(name)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_preprocessing.png', dpi=300, bbox_inches='tight')
    print("Image preprocessing visualization saved as 'image_preprocessing.png'")
    plt.show()
    
    return processed_images

def feature_extraction_traditional():
    """Demonstrate traditional feature extraction methods"""
    print("\n=== Traditional Feature Extraction ===")
    
    # Load Fashion-MNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Use subset for demo
    X_sample = X_train[:1000]
    y_sample = y_train[:1000]
    
    print(f"Sample data shape: {X_sample.shape}")
    
    # 1. Histogram of Oriented Gradients (HOG) - simplified
    def simple_hog_features(images):
        features = []
        for img in images:
            # Calculate gradients
            grad_x = np.gradient(img.astype(float), axis=1)
            grad_y = np.gradient(img.astype(float), axis=0)
            
            # Calculate magnitude and direction
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            
            # Simple histogram (8 bins)
            hist, _ = np.histogram(direction, bins=8, range=(-np.pi, np.pi), weights=magnitude)
            features.append(hist)
        
        return np.array(features)
    
    # 2. Local Binary Patterns (LBP) - simplified
    def simple_lbp_features(images):
        features = []
        for img in images:
            lbp = np.zeros_like(img)
            for i in range(1, img.shape[0]-1):
                for j in range(1, img.shape[1]-1):
                    center = img[i, j]
                    code = 0
                    code |= (img[i-1, j-1] >= center) << 7
                    code |= (img[i-1, j] >= center) << 6
                    code |= (img[i-1, j+1] >= center) << 5
                    code |= (img[i, j+1] >= center) << 4
                    code |= (img[i+1, j+1] >= center) << 3
                    code |= (img[i+1, j] >= center) << 2
                    code |= (img[i+1, j-1] >= center) << 1
                    code |= (img[i, j-1] >= center) << 0
                    lbp[i, j] = code
            
            # Histogram of LBP codes
            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            features.append(hist)
        
        return np.array(features)
    
    # Extract features
    print("Extracting HOG features...")
    hog_features = simple_hog_features(X_sample)
    
    print("Extracting LBP features...")
    lbp_features = simple_lbp_features(X_sample)
    
    print(f"HOG features shape: {hog_features.shape}")
    print(f"LBP features shape: {lbp_features.shape}")
    
    # Visualize feature distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # HOG features for first image
    axes[0, 0].bar(range(8), hog_features[0])
    axes[0, 0].set_title('HOG Features (First Image)')
    axes[0, 0].set_xlabel('Orientation Bin')
    axes[0, 0].set_ylabel('Magnitude')
    
    # LBP features for first image
    axes[0, 1].plot(lbp_features[0][:50])  # Show first 50 bins
    axes[0, 1].set_title('LBP Features (First Image)')
    axes[0, 1].set_xlabel('LBP Code')
    axes[0, 1].set_ylabel('Frequency')
    
    # Feature correlation
    hog_corr = np.corrcoef(hog_features.T)
    im1 = axes[1, 0].imshow(hog_corr, cmap='coolwarm')
    axes[1, 0].set_title('HOG Features Correlation')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # PCA on HOG features
    pca = PCA(n_components=2)
    hog_pca = pca.fit_transform(hog_features)
    
    scatter = axes[1, 1].scatter(hog_pca[:, 0], hog_pca[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
    axes[1, 1].set_title('HOG Features PCA (2D)')
    axes[1, 1].set_xlabel('PC1')
    axes[1, 1].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('feature_extraction.png', dpi=300, bbox_inches='tight')
    print("Feature extraction visualization saved as 'feature_extraction.png'")
    plt.show()
    
    return hog_features, lbp_features

def cnn_image_classification():
    """Build CNN for image classification with data augmentation"""
    print("\n=== CNN Image Classification with Data Augmentation ===")
    
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Use subset for demo
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Define class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    # Build CNN model
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
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
    
    # Train model with data augmentation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=20,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.3f}")
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training history
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Training Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=class_names, yticklabels=class_names)
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Sample predictions
    sample_indices = np.random.choice(len(X_test), 9, replace=False)
    for i, idx in enumerate(sample_indices):
        row, col = i // 3, i % 3
        if i < 6:  # First 6 in top row
            axes[1, 0].imshow(X_test[idx])
            axes[1, 0].set_title(f'True: {class_names[y_test[idx][0]]}\nPred: {class_names[predicted_classes[idx]]}')
            axes[1, 0].axis('off')
        else:  # Last 3 in bottom row
            axes[1, 1].imshow(X_test[idx])
            axes[1, 1].set_title(f'True: {class_names[y_test[idx][0]]}\nPred: {class_names[predicted_classes[idx]]}')
            axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('cnn_classification.png', dpi=300, bbox_inches='tight')
    print("CNN classification visualization saved as 'cnn_classification.png'")
    plt.show()
    
    return model, history, test_accuracy

def transfer_learning_vision():
    """Demonstrate transfer learning for computer vision tasks"""
    print("\n=== Transfer Learning for Computer Vision ===")
    
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Use subset for demo
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    X_test = X_test[:500]
    y_test = y_test[:500]
    
    # Resize images to match pre-trained model input size (224x224)
    X_train_resized = tf.image.resize(X_train, (224, 224))
    X_test_resized = tf.image.resize(X_test, (224, 224))
    
    print(f"Resized training data shape: {X_train_resized.shape}")
    print(f"Resized test data shape: {X_test_resized.shape}")
    
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
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
        X_train_resized, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test_resized, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.3f}")
    
    # Fine-tuning: Unfreeze some layers
    base_model.trainable = True
    for layer in base_model.layers[:-4]:  # Freeze all but last 4 layers
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune model
    history_finetune = model.fit(
        X_train_resized, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Final evaluation
    test_loss_final, test_accuracy_final = model.evaluate(X_test_resized, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_accuracy_final:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training history
    axes[0].plot(history.history['accuracy'], label='Transfer Learning', alpha=0.8)
    axes[0].plot(history_finetune.history['accuracy'], label='Fine-tuning', alpha=0.8)
    axes[0].set_title('Training Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation history
    axes[1].plot(history.history['val_accuracy'], label='Transfer Learning', alpha=0.8)
    axes[1].plot(history_finetune.history['val_accuracy'], label='Fine-tuning', alpha=0.8)
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transfer_learning_vision.png', dpi=300, bbox_inches='tight')
    print("Transfer learning visualization saved as 'transfer_learning_vision.png'")
    plt.show()
    
    return model, history, test_accuracy_final

def object_detection_basics():
    """Demonstrate basic object detection concepts"""
    print("\n=== Object Detection Basics ===")
    
    # Create synthetic image with objects
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add some "objects" (colored rectangles)
    image[50:100, 50:100] = [255, 0, 0]    # Red square
    image[120:170, 120:170] = [0, 255, 0]  # Green square
    image[30:80, 120:170] = [0, 0, 255]    # Blue square
    
    # Add noise
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Simple object detection using color segmentation
    def detect_objects_by_color(image, color_ranges):
        detections = []
        for color_name, (lower, upper) in color_ranges.items():
            # Create mask
            mask = cv2.inRange(image, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append({
                        'color': color_name,
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        return detections
    
    # Define color ranges (BGR format for OpenCV)
    color_ranges = {
        'red': (np.array([0, 0, 200]), np.array([50, 50, 255])),
        'green': (np.array([0, 200, 0]), np.array([50, 255, 50])),
        'blue': (np.array([200, 0, 0]), np.array([255, 50, 50]))
    }
    
    # Detect objects
    detections = detect_objects_by_color(image, color_ranges)
    
    print(f"Detected {len(detections)} objects:")
    for i, detection in enumerate(detections):
        print(f"  {i+1}. {detection['color']} object at {detection['bbox']} (area: {detection['area']})")
    
    # Visualize detections
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Image with detections
    result_image = image.copy()
    for detection in detections:
        x, y, w, h = detection['bbox']
        color = (0, 255, 0) if detection['color'] == 'red' else (255, 0, 0) if detection['color'] == 'green' else (0, 0, 255)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_image, detection['color'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    axes[1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Object Detections')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('object_detection.png', dpi=300, bbox_inches='tight')
    print("Object detection visualization saved as 'object_detection.png'")
    plt.show()
    
    return detections

def image_generation_basics():
    """Demonstrate basic image generation concepts"""
    print("\n=== Image Generation Basics ===")
    
    # Simple image generation using mathematical functions
    def generate_pattern(width, height, pattern_type='spiral'):
        x = np.linspace(-5, 5, width)
        y = np.linspace(-5, 5, height)
        X, Y = np.meshgrid(x, y)
        
        if pattern_type == 'spiral':
            R = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            Z = np.sin(R + theta)
        elif pattern_type == 'waves':
            Z = np.sin(X) * np.cos(Y)
        elif pattern_type == 'checkerboard':
            Z = np.sin(X * np.pi) * np.sin(Y * np.pi)
        else:  # random
            Z = np.random.randn(height, width)
        
        return Z
    
    # Generate different patterns
    patterns = {
        'Spiral': generate_pattern(200, 200, 'spiral'),
        'Waves': generate_pattern(200, 200, 'waves'),
        'Checkerboard': generate_pattern(200, 200, 'checkerboard'),
        'Random': generate_pattern(200, 200, 'random')
    }
    
    # Visualize patterns
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, (name, pattern) in enumerate(patterns.items()):
        im = axes[i].imshow(pattern, cmap='viridis')
        axes[i].set_title(name)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('image_generation.png', dpi=300, bbox_inches='tight')
    print("Image generation visualization saved as 'image_generation.png'")
    plt.show()
    
    # Simple autoencoder for image reconstruction
    print("\nBuilding simple autoencoder...")
    
    # Load Fashion-MNIST for autoencoder demo
    (X_train, _), (X_test, _) = fashion_mnist.load_data()
    
    # Normalize and reshape
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Use subset for demo
    X_train = X_train[:1000]
    X_test = X_test[:200]
    
    # Build autoencoder
    encoder = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same')
    ])
    
    decoder = models.Sequential([
        layers.Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(7, 7, 8)),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    
    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Train autoencoder
    history = autoencoder.fit(
        X_train, X_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, X_test),
        verbose=0
    )
    
    # Generate reconstructions
    reconstructed = autoencoder.predict(X_test[:10])
    
    # Visualize reconstructions
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    
    for i in range(10):
        # Original
        axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    
    plt.tight_layout()
    plt.savefig('autoencoder_reconstruction.png', dpi=300, bbox_inches='tight')
    print("Autoencoder reconstruction saved as 'autoencoder_reconstruction.png'")
    plt.show()
    
    return patterns, autoencoder, history

def main():
    """Main function to run all computer vision demonstrations"""
    print("👁️ Computer Vision & Image Processing")
    print("=" * 50)
    
    # Image preprocessing techniques
    image_preprocessing_techniques()
    
    # Traditional feature extraction
    feature_extraction_traditional()
    
    # CNN image classification
    cnn_image_classification()
    
    # Transfer learning for vision
    transfer_learning_vision()
    
    # Object detection basics
    object_detection_basics()
    
    # Image generation basics
    image_generation_basics()
    
    print("\n" + "=" * 50)
    print("Lesson 11 Complete!")
    print("Next: Learn natural language processing and text analysis")
    print("Key takeaway: Computer vision enables machines to understand and interpret visual information")

if __name__ == "__main__":
    main()
# run the model