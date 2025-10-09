"""
07-classification-algorithms.py

Classification Algorithms
-------------------------
Master classification algorithms for predicting categorical outcomes. Learn
about different classification techniques, their strengths, and when to use them.

What you'll learn
-----------------
1) Logistic regression for binary and multiclass classification
2) Decision trees and random forests
3) Support Vector Machines (SVM)
4) Naive Bayes classifier
5) K-Nearest Neighbors (KNN)
6) Model evaluation for classification

Key Concepts
------------
- Probability-based classification
- Tree-based methods
- Distance-based methods
- Ensemble methods
- Classification metrics (accuracy, precision, recall, F1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_classification_datasets():
    """Load and prepare classification datasets"""
    print("=== Classification Datasets ===")
    
    # Load Iris dataset (multiclass)
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Load Breast Cancer dataset (binary)
    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target
    
    # Create synthetic dataset
    X_synthetic, y_synthetic = make_classification(
        n_samples=1000, n_features=4, n_informative=4, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )
    
    datasets = {
        'Iris (Multiclass)': (X_iris, y_iris, iris.target_names),
        'Breast Cancer (Binary)': (X_cancer, y_cancer, cancer.target_names),
        'Synthetic (Binary)': (X_synthetic, y_synthetic, ['Class 0', 'Class 1'])
    }
    
    for name, (X, y, labels) in datasets.items():
        print(f"{name}:")
        print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"  Classes: {len(labels)} - {labels}")
        print(f"  Class distribution: {np.bincount(y)}")
        print()
    
    return datasets

def logistic_regression_demo():
    """Demonstrate logistic regression for classification"""
    print("=== Logistic Regression ===")
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit logistic regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test_scaled)
    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    
    # Feature importance (coefficients)
    feature_importance = np.abs(lr.coef_[0])
    top_features = np.argsort(feature_importance)[-10:]
    
    print(f"\nTop 10 most important features:")
    for i, idx in enumerate(reversed(top_features)):
        print(f"  {i+1:2d}. {cancer.feature_names[idx]}: {feature_importance[idx]:.3f}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('logistic_regression_roc.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return lr, accuracy, precision, recall, f1

def decision_trees_demo():
    """Demonstrate decision trees and random forests"""
    print("\n=== Decision Trees & Random Forests ===")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=3)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Decision Tree Accuracy: {dt_accuracy:.3f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
    
    # Visualize decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    plt.title('Decision Tree Visualization')
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Decision tree feature importance
    ax1.bar(iris.feature_names, dt.feature_importances_)
    ax1.set_title('Decision Tree - Feature Importance')
    ax1.set_ylabel('Importance')
    ax1.tick_params(axis='x', rotation=45)
    
    # Random forest feature importance
    ax2.bar(iris.feature_names, rf.feature_importances_)
    ax2.set_title('Random Forest - Feature Importance')
    ax2.set_ylabel('Importance')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dt, rf, dt_accuracy, rf_accuracy

def svm_demo():
    """Demonstrate Support Vector Machines"""
    print("\n=== Support Vector Machines ===")
    
    # Create synthetic dataset
    X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                              n_redundant=0, n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Different SVM kernels
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    svm_results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, kernel in enumerate(kernels):
        # Fit SVM
        svm = SVC(kernel=kernel, random_state=42, probability=True)
        svm.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        svm_results[kernel] = {'model': svm, 'accuracy': accuracy}
        
        # Decision boundary visualization
        h = 0.02
        x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
        y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[i].contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
        scatter = axes[i].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='RdYlBu', edgecolors='black')
        axes[i].set_title(f'SVM {kernel.capitalize()} Kernel\nAccuracy: {accuracy:.3f}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('svm_kernels.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("SVM Kernel Comparison:")
    for kernel, result in svm_results.items():
        print(f"  {kernel.capitalize():8}: {result['accuracy']:.3f}")
    
    return svm_results

def naive_bayes_demo():
    """Demonstrate Naive Bayes classifier"""
    print("\n=== Naive Bayes Classifier ===")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Gaussian Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Naive Bayes - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('naive_bayes_confusion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Class probabilities
    y_pred_proba = nb.predict_proba(X_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, class_name in enumerate(iris.target_names):
        axes[i].hist(y_pred_proba[:, i], bins=20, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Probability Distribution - {class_name}')
        axes[i].set_xlabel('Probability')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('naive_bayes_probabilities.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return nb, accuracy, cm

def knn_demo():
    """Demonstrate K-Nearest Neighbors"""
    print("\n=== K-Nearest Neighbors ===")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different k values
    k_values = range(1, 21)
    accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    # Find best k
    best_k = k_values[np.argmax(accuracies)]
    best_accuracy = max(accuracies)
    
    print(f"Best k: {best_k}")
    print(f"Best accuracy: {best_accuracy:.3f}")
    
    # Visualize k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best k = {best_k}')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title('KNN: k vs Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('knn_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Fit best model
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train_scaled, y_train)
    y_pred = best_knn.predict(X_test_scaled)
    
    # Detailed evaluation
    print(f"\nDetailed Classification Report (k={best_k}):")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    return best_knn, best_k, best_accuracy

def classification_model_comparison():
    """Compare all classification algorithms"""
    print("\n=== Classification Model Comparison ===")
    
    # Load dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                              n_redundant=2, n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Train and evaluate models
    results = {}
    
    print("Model Performance Comparison:")
    print("-" * 60)
    
    for name, model in models.items():
        # Fit model
        if name in ['SVM (RBF)', 'KNN (k=5)']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        if name in ['SVM (RBF)', 'KNN (k=5)']:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"{name:20}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    cv_means = [results[name]['cv_mean'] for name in names]
    cv_stds = [results[name]['cv_std'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
    axes[0, 0].bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision, Recall, F1 comparison
    precisions = [results[name]['precision'] for name in names]
    recalls = [results[name]['recall'] for name in names]
    f1_scores = [results[name]['f1'] for name in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    axes[0, 1].bar(x - width, precisions, width, label='Precision', alpha=0.8)
    axes[0, 1].bar(x, recalls, width, label='Recall', alpha=0.8)
    axes[0, 1].bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision, Recall, F1 Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cross-validation scores with error bars
    axes[1, 0].bar(names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Cross-Validation Accuracy')
    axes[1, 0].set_title('Cross-Validation Scores with Standard Deviation')
    axes[1, 0].tick_params(axis='x', rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance heatmap
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'cv_mean']
    heatmap_data = np.array([[results[name][metric] for metric in metrics] for name in names])
    
    im = axes[1, 1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_xticks(range(len(metrics)))
    axes[1, 1].set_xticklabels(metrics, rotation=45, ha='right')
    axes[1, 1].set_yticks(range(len(names)))
    axes[1, 1].set_yticklabels(names)
    axes[1, 1].set_title('Performance Heatmap')
    
    # Add text annotations
    for i in range(len(names)):
        for j in range(len(metrics)):
            axes[1, 1].text(j, i, f'{heatmap_data[i, j]:.3f}', ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.savefig('classification_comparison.png', dpi=300, bbox_inches='tight')
    print("Classification comparison visualization saved as 'classification_comparison.png'")
    plt.show()
    
    return results

def main():
    """Main function to run all classification demonstrations"""
    print("🎯 Classification Algorithms")
    print("=" * 50)
    
    # Load datasets
    load_classification_datasets()
    
    # Logistic regression
    logistic_regression_demo()
    
    # Decision trees and random forests
    decision_trees_demo()
    
    # Support Vector Machines
    svm_demo()
    
    # Naive Bayes
    naive_bayes_demo()
    
    # K-Nearest Neighbors
    knn_demo()
    
    # Model comparison
    classification_model_comparison()
    
    print("\n" + "=" * 50)
    print("✅ Lesson 7 Complete!")
    print("Next: Learn model evaluation and validation techniques")
    print("Key takeaway: Different classification algorithms work best for different types of data and problems")

if __name__ == "__main__":
    main()
