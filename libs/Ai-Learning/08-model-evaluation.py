"""
08-model-evaluation.py

Model Evaluation & Validation
-----------------------------
Learn comprehensive model evaluation techniques, validation strategies, and
performance metrics. Understand how to properly assess model performance
and avoid common pitfalls.

What you'll learn
-----------------
1) Cross-validation techniques (k-fold, stratified, time series)
2) Performance metrics for classification and regression
3) Bias-variance tradeoff and learning curves
4) Model selection and hyperparameter tuning
5) Overfitting detection and prevention

Key Concepts
------------
- Train/validation/test splits
- Cross-validation for robust evaluation
- Performance metrics interpretation
- Learning curves and model complexity
- Hyperparameter optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import (train_test_split, cross_val_score, KFold, 
                                   StratifiedKFold, learning_curve, validation_curve,
                                   GridSearchCV, RandomizedSearchCV)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, mean_squared_error, r2_score,
                           confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def cross_validation_techniques():
    """Demonstrate different cross-validation techniques"""
    print("=== Cross-Validation Techniques ===")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Different CV strategies
    cv_strategies = {
        'KFold (k=5)': KFold(n_splits=5, shuffle=True, random_state=42),
        'Stratified KFold (k=5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        'KFold (k=10)': KFold(n_splits=10, shuffle=True, random_state=42),
        'Stratified KFold (k=10)': StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    }
    
    results = {}
    
    print("Cross-Validation Results:")
    print("-" * 80)
    
    for model_name, model in models.items():
        results[model_name] = {}
        print(f"\n{model_name}:")
        
        for cv_name, cv in cv_strategies.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            mean_score = scores.mean()
            std_score = scores.std()
            
            results[model_name][cv_name] = {
                'scores': scores,
                'mean': mean_score,
                'std': std_score
            }
            
            print(f"  {cv_name:25}: {mean_score:.3f} (+/- {std_score * 2:.3f})")
    
    # Visualize CV results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (model_name, model_results) in enumerate(results.items()):
        cv_names = list(model_results.keys())
        means = [model_results[cv]['mean'] for cv in cv_names]
        stds = [model_results[cv]['std'] for cv in cv_names]
        
        x = np.arange(len(cv_names))
        axes[i].bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        axes[i].set_title(f'{model_name} - CV Comparison')
        axes[i].set_xlabel('Cross-Validation Strategy')
        axes[i].set_ylabel('Accuracy')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(cv_names, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cross_validation_comparison.png', dpi=300, bbox_inches='tight')
    print("Cross-validation comparison saved as 'cross_validation_comparison.png'")
    plt.show()
    
    return results

def learning_curves_analysis():
    """Analyze learning curves to understand bias-variance tradeoff"""
    print("\n=== Learning Curves Analysis ===")
    
    # Create datasets of different sizes
    X_small, y_small = make_classification(n_samples=200, n_features=4, n_informative=4,
                                          n_redundant=0, random_state=42)
    X_large, y_large = make_classification(n_samples=2000, n_features=4, n_informative=4,
                                          n_redundant=0, random_state=42)
    
    # Models with different complexity
    models = {
        'Simple (Logistic)': LogisticRegression(random_state=42, max_iter=1000),
        'Medium (Tree, max_depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Complex (Tree, no limit)': DecisionTreeClassifier(random_state=42)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (name, model) in enumerate(models.items()):
        # Learning curve for small dataset
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_small, y_small, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[i].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        axes[i].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        axes[i].set_title(f'{name} - Learning Curve')
        axes[i].set_xlabel('Training Set Size')
        axes[i].set_ylabel('Accuracy')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[3])
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    print("Learning curves saved as 'learning_curves.png'")
    plt.show()

def validation_curves_analysis():
    """Analyze validation curves for hyperparameter tuning"""
    print("\n=== Validation Curves Analysis ===")
    
    # Load data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                              n_redundant=2, random_state=42)
    
    # Decision Tree with different max_depth values
    param_range = range(1, 21)
    train_scores, val_scores = validation_curve(
        DecisionTreeClassifier(random_state=42), X, y,
        param_name='max_depth', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree - Validation Curve (Max Depth)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('validation_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal parameter
    optimal_depth = param_range[np.argmax(val_mean)]
    print(f"Optimal max_depth: {optimal_depth}")
    print(f"Best validation score: {val_mean[np.argmax(val_mean)]:.3f}")

def hyperparameter_optimization():
    """Demonstrate hyperparameter optimization techniques"""
    print("\n=== Hyperparameter Optimization ===")
    
    # Load data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                              n_redundant=2, random_state=42)
    
    # Grid Search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    print(f"Grid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Random Search
    from scipy.stats import randint, uniform
    
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [3, 5, 7, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }
    
    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=50, cv=5,
                                      scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X, y)
    
    print(f"\nRandom Search Results:")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.3f}")
    
    # Compare results
    results_comparison = {
        'Grid Search': grid_search.best_score_,
        'Random Search': random_search.best_score_
    }
    
    plt.figure(figsize=(8, 6))
    methods = list(results_comparison.keys())
    scores = list(results_comparison.values())
    
    bars = plt.bar(methods, scores, alpha=0.8, color=['blue', 'green'])
    plt.ylabel('Best CV Score')
    plt.title('Grid Search vs Random Search')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.savefig('hyperparameter_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return grid_search, random_search

def model_selection_strategies():
    """Demonstrate different model selection strategies"""
    print("\n=== Model Selection Strategies ===")
    
    # Load data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                              n_redundant=2, random_state=42)
    
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    # Train and evaluate on validation set
    val_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_results[name] = val_accuracy
        print(f"{name:20}: Validation Accuracy = {val_accuracy:.3f}")
    
    # Select best model
    best_model_name = max(val_results, key=val_results.get)
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best validation accuracy: {val_results[best_model_name]:.3f}")
    
    # Evaluate best model on test set
    test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Cross-validation on full training set
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    print(f"CV mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return val_results, best_model_name, test_accuracy

def performance_metrics_comprehensive():
    """Comprehensive demonstration of performance metrics"""
    print("\n=== Comprehensive Performance Metrics ===")
    
    # Load data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                              n_redundant=2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("Classification Metrics:")
    print("-" * 30)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"ROC AUC:   {roc_auc:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metrics bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    values = [accuracy, precision, recall, f1, roc_auc]
    axes[1, 0].bar(metrics, values, alpha=0.8, color='skyblue')
    axes[1, 0].set_title('Performance Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1, 1].plot(recall_curve, precision_curve, color='darkorange', lw=2)
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Precision-Recall Curve')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    print("Performance metrics visualization saved as 'performance_metrics.png'")
    plt.show()
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'roc_auc': roc_auc, 'cm': cm
    }

def main():
    """Main function to run all model evaluation demonstrations"""
    print("Model Evaluation & Validation")
    print("=" * 50)
    
    # Cross-validation techniques
    cross_validation_techniques()
    
    # Learning curves analysis
    learning_curves_analysis()
    
    # Validation curves analysis
    validation_curves_analysis()
    
    # Hyperparameter optimization
    hyperparameter_optimization()
    
    # Model selection strategies
    model_selection_strategies()
    
    # Comprehensive performance metrics
    performance_metrics_comprehensive()
    
    print("\n" + "=" * 50)
    print("Lesson 8 Complete!")
    print("Next: Learn neural networks and deep learning fundamentals")
    print("Key takeaway: Proper model evaluation is crucial for building reliable AI systems")

if __name__ == "__main__":
    main()
