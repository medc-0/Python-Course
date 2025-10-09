"""
06-linear-regression.py

Linear Regression & Regression Techniques
----------------------------------------
Master linear regression and related regression techniques. Learn to build,
evaluate, and interpret regression models for predicting continuous values.

What you'll learn
-----------------
1) Simple and multiple linear regression
2) Polynomial regression for non-linear relationships
3) Regularization techniques (Ridge, Lasso, Elastic Net)
4) Model evaluation metrics for regression
5) Feature importance and model interpretation

Key Concepts
------------
- Ordinary Least Squares (OLS)
- Residual analysis and assumptions
- Overfitting and underfitting
- Regularization to prevent overfitting
- Cross-validation for model selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def simple_linear_regression():
    """Demonstrate simple linear regression"""
    print("=== Simple Linear Regression ===")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 1) * 10
    y = 2.5 * X.flatten() + 1.5 + np.random.randn(100) * 2
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    
    # Evaluate model
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Training MSE: {train_mse:.3f}")
    print(f"Test MSE: {test_mse:.3f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Coefficient: {lr.coef_[0]:.3f}")
    print(f"Intercept: {lr.intercept_:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training data
    axes[0].scatter(X_train, y_train, alpha=0.7, color='blue', label='Training data')
    axes[0].plot(X_train, y_pred_train, color='red', linewidth=2, label='Fitted line')
    axes[0].set_title('Training Data')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Test data
    axes[1].scatter(X_test, y_test, alpha=0.7, color='green', label='Test data')
    axes[1].plot(X_test, y_pred_test, color='red', linewidth=2, label='Predictions')
    axes[1].set_title('Test Data')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_linear_regression.png', dpi=300, bbox_inches='tight')
    print("Simple linear regression visualization saved as 'simple_linear_regression.png'")
    plt.show()
    
    return lr, X_train, X_test, y_train, y_test

def multiple_linear_regression():
    """Demonstrate multiple linear regression"""
    print("\n=== Multiple Linear Regression ===")
    
    # Generate multiple features
    X, y = make_regression(n_samples=1000, n_features=5, n_informative=5,
                          noise=10, random_state=42)
    
    # Create feature names
    feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {feature_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit multiple linear regression
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = mlr.predict(X_train)
    y_pred_test = mlr.predict(X_test)
    
    # Evaluate model
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Training MSE: {train_mse:.3f}")
    print(f"Test MSE: {test_mse:.3f}")
    
    # Feature importance (coefficients)
    print(f"\nFeature coefficients:")
    for i, (feature, coef) in enumerate(zip(feature_names, mlr.coef_)):
        print(f"  {feature}: {coef:.3f}")
    print(f"  Intercept: {mlr.intercept_:.3f}")
    
    # Visualize feature importance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Coefficients
    axes[0].bar(feature_names, mlr.coef_)
    axes[0].set_title('Feature Coefficients')
    axes[0].set_xlabel('Features')
    axes[0].set_ylabel('Coefficient Value')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Predicted vs Actual
    axes[1].scatter(y_test, y_pred_test, alpha=0.7)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title('Predicted vs Actual Values')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiple_linear_regression.png', dpi=300, bbox_inches='tight')
    print("Multiple linear regression visualization saved as 'multiple_linear_regression.png'")
    plt.show()
    
    return mlr, feature_names

def polynomial_regression():
    """Demonstrate polynomial regression for non-linear relationships"""
    print("\n=== Polynomial Regression ===")
    
    # Generate non-linear data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y = 0.5 * X**2 + 2 * X + 1 + np.random.randn(100) * 0.5
    
    # Reshape for sklearn
    X = X.reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit different polynomial degrees
    degrees = [1, 2, 3, 5, 10]
    results = {}
    
    for degree in degrees:
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        # Fit linear regression on polynomial features
        lr = LinearRegression()
        lr.fit(X_train_poly, y_train)
        
        # Make predictions
        y_pred_train = lr.predict(X_train_poly)
        y_pred_test = lr.predict(X_test_poly)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        results[degree] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'model': lr,
            'poly_features': poly_features
        }
        
        print(f"Degree {degree:2d}: Train R² = {train_r2:.3f}, Test R² = {test_r2:.3f}")
    
    # Visualize polynomial fits
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    
    for i, degree in enumerate(degrees):
        ax = axes[i]
        
        # Plot original data
        ax.scatter(X_train, y_train, alpha=0.7, color='blue', label='Training data')
        ax.scatter(X_test, y_test, alpha=0.7, color='red', label='Test data')
        
        # Plot polynomial fit
        X_plot_poly = results[degree]['poly_features'].transform(X_plot)
        y_plot = results[degree]['model'].predict(X_plot_poly)
        ax.plot(X_plot, y_plot, color='green', linewidth=2, label=f'Degree {degree}')
        
        ax.set_title(f'Polynomial Degree {degree}\nTest R² = {results[degree]["test_r2"]:.3f}')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('polynomial_regression.png', dpi=300, bbox_inches='tight')
    print("Polynomial regression visualization saved as 'polynomial_regression.png'")
    plt.show()
    
    return results

def regularization_techniques():
    """Demonstrate Ridge, Lasso, and Elastic Net regularization"""
    print("\n=== Regularization Techniques ===")
    
    # Generate data with many features (some irrelevant)
    X, y = make_regression(n_samples=100, n_features=20, n_informative=5,
                          noise=10, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features for regularization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'Lasso (α=1.0)': Lasso(alpha=1.0),
        'Elastic Net (α=0.1)': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Elastic Net (α=1.0)': ElasticNet(alpha=1.0, l1_ratio=0.5)
    }
    
    results = {}
    
    print("Model Performance:")
    print("-" * 60)
    
    for name, model in models.items():
        # Fit model
        if name == 'Linear Regression':
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        # Count non-zero coefficients (for Lasso)
        if hasattr(model, 'coef_'):
            n_features = np.sum(np.abs(model.coef_) > 1e-6)
        else:
            n_features = len(model.coef_)
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'n_features': n_features,
            'model': model
        }
        
        print(f"{name:20}: Test R² = {test_r2:.3f}, Features = {n_features:2d}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² scores
    names = list(results.keys())
    train_r2s = [results[name]['train_r2'] for name in names]
    test_r2s = [results[name]['test_r2'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[0].bar(x - width/2, train_r2s, width, label='Training R²', alpha=0.8)
    axes[0].bar(x + width/2, test_r2s, width, label='Test R²', alpha=0.8)
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Training vs Test R² Scores')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Number of features used
    n_features = [results[name]['n_features'] for name in names]
    axes[1].bar(names, n_features, alpha=0.8)
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Number of Features Used')
    axes[1].set_title('Feature Selection (Lasso Effect)')
    axes[1].tick_params(axis='x', rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regularization_techniques.png', dpi=300, bbox_inches='tight')
    print("Regularization visualization saved as 'regularization_techniques.png'")
    plt.show()
    
    return results

def regression_evaluation_metrics():
    """Demonstrate various regression evaluation metrics"""
    print("\n=== Regression Evaluation Metrics ===")
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # Calculate various metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Additional metrics
    residuals = y_test - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    print("Regression Evaluation Metrics:")
    print("-" * 40)
    print(f"Mean Squared Error (MSE):     {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"Mean Absolute Error (MAE):    {mae:.3f}")
    print(f"R² Score:                     {r2:.3f}")
    print(f"Mean of Residuals:            {mean_residual:.6f}")
    print(f"Std of Residuals:             {std_residual:.3f}")
    
    # Residual analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Regression Model Evaluation', fontsize=16)
    
    # 1. Predicted vs Actual
    axes[0, 0].scatter(y_test, y_pred, alpha=0.7)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predicted vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.7)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram of residuals
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot for normality
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_evaluation.png', dpi=300, bbox_inches='tight')
    print("Regression evaluation visualization saved as 'regression_evaluation.png'")
    plt.show()
    
    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'residuals': residuals, 'y_test': y_test, 'y_pred': y_pred
    }

def main():
    """Main function to run all regression demonstrations"""
    print("📈 Linear Regression & Regression Techniques")
    print("=" * 50)
    
    # Simple linear regression
    simple_linear_regression()
    
    # Multiple linear regression
    multiple_linear_regression()
    
    # Polynomial regression
    polynomial_regression()
    
    # Regularization techniques
    regularization_techniques()
    
    # Evaluation metrics
    regression_evaluation_metrics()
    
    print("\n" + "=" * 50)
    print("✅ Lesson 6 Complete!")
    print("Next: Learn classification algorithms and techniques")
    print("Key takeaway: Regression models predict continuous values and require different evaluation metrics")

if __name__ == "__main__":
    main()
