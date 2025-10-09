"""
02-data-foundations.py

Data Foundations with NumPy & Pandas
------------------------------------
Master the fundamental libraries for AI/ML: NumPy for numerical computing
and Pandas for data manipulation. These are the building blocks of all AI work.

What you'll learn
-----------------
1) NumPy arrays and operations for numerical computing
2) Pandas DataFrames for structured data manipulation
3) Data loading, cleaning, and basic analysis
4) Essential operations for AI/ML preprocessing

Key Concepts
------------
- Vectorized operations (much faster than loops)
- Broadcasting (operations between arrays of different shapes)
- Indexing and slicing for data selection
- Missing value handling
- Data types and conversions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston
import warnings
warnings.filterwarnings('ignore')

def numpy_fundamentals():
    """Demonstrate essential NumPy operations for AI/ML"""
    print("=== NumPy Fundamentals ===")
    
    # 1. Creating arrays
    arr1d = np.array([1, 2, 3, 4, 5])
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr_zeros = np.zeros((3, 4))
    arr_ones = np.ones((2, 3))
    arr_range = np.arange(0, 10, 2)
    arr_random = np.random.randn(3, 3)
    
    print(f"1D Array: {arr1d}")
    print(f"2D Array shape: {arr2d.shape}")
    print(f"Zeros array:\n{arr_zeros}")
    print(f"Random array:\n{arr_random}")
    
    # 2. Array operations (vectorized)
    print(f"\nArray operations:")
    print(f"Original: {arr1d}")
    print(f"Add 10: {arr1d + 10}")
    print(f"Multiply by 2: {arr1d * 2}")
    print(f"Square: {arr1d ** 2}")
    print(f"Sum: {arr1d.sum()}")
    print(f"Mean: {arr1d.mean()}")
    print(f"Standard deviation: {arr1d.std()}")
    
    # 3. Broadcasting example
    print(f"\nBroadcasting example:")
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    vector = np.array([10, 20, 30])
    result = matrix + vector  # Broadcasting
    print(f"Matrix:\n{matrix}")
    print(f"Vector: {vector}")
    print(f"Result (broadcasted):\n{result}")
    
    return arr1d, arr2d, matrix

def pandas_data_manipulation():
    """Demonstrate Pandas for data manipulation"""
    print("\n=== Pandas Data Manipulation ===")
    
    # 1. Create DataFrame from dictionary
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print(f"\nDataFrame info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # 2. Data selection and filtering
    print(f"\nData selection examples:")
    print(f"First 3 rows:\n{df.head(3)}")
    print(f"Age column: {df['age'].tolist()}")
    print(f"IT department employees:\n{df[df['department'] == 'IT']}")
    print(f"High salary (>60000):\n{df[df['salary'] > 60000]}")
    
    # 3. Data aggregation
    print(f"\nAggregation examples:")
    print(f"Average age: {df['age'].mean():.1f}")
    print(f"Salary statistics:\n{df['salary'].describe()}")
    print(f"Department counts:\n{df['department'].value_counts()}")
    
    # 4. Data transformation
    print(f"\nData transformation:")
    df['salary_category'] = df['salary'].apply(
        lambda x: 'High' if x > 60000 else 'Medium' if x > 50000 else 'Low'
    )
    print(f"With salary categories:\n{df[['name', 'salary', 'salary_category']]}")
    
    return df

def handle_missing_data():
    """Demonstrate missing data handling"""
    print("\n=== Missing Data Handling ===")
    
    # Create DataFrame with missing values
    data_with_nulls = {
        'temperature': [20, 22, np.nan, 25, 23],
        'humidity': [60, np.nan, 70, 65, 68],
        'pressure': [1013, 1015, 1012, np.nan, 1014]
    }
    df_missing = pd.DataFrame(data_with_nulls)
    print("DataFrame with missing values:")
    print(df_missing)
    print(f"\nMissing values count:\n{df_missing.isnull().sum()}")
    
    # Handle missing values
    print(f"\nHandling missing values:")
    df_filled = df_missing.fillna(df_missing.mean())
    print(f"Filled with mean:\n{df_filled}")
    
    df_dropped = df_missing.dropna()
    print(f"Dropped rows with NaN:\n{df_dropped}")
    
    return df_missing, df_filled, df_dropped

def load_real_datasets():
    """Load and explore real datasets"""
    print("\n=== Real Dataset Exploration ===")
    
    # Load Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    iris_df['species_name'] = iris_df['species'].map(
        {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    )
    
    print("Iris Dataset:")
    print(f"Shape: {iris_df.shape}")
    print(f"Features: {list(iris_df.columns[:-2])}")
    print(f"Species distribution:\n{iris_df['species_name'].value_counts()}")
    print(f"\nFirst few rows:\n{iris_df.head()}")
    
    # Basic statistics
    print(f"\nStatistical summary:\n{iris_df.describe()}")
    
    return iris_df

def data_visualization_basics():
    """Create basic visualizations for data exploration"""
    print("\n=== Data Visualization ===")
    
    # Load data
    iris_df = load_real_datasets()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Iris Dataset Analysis', fontsize=16)
    
    # 1. Histogram of sepal length
    axes[0, 0].hist(iris_df['sepal length (cm)'], bins=20, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Sepal Length Distribution')
    axes[0, 0].set_xlabel('Sepal Length (cm)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Scatter plot: sepal length vs width
    colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    for species in colors:
        data = iris_df[iris_df['species_name'] == species]
        axes[0, 1].scatter(data['sepal length (cm)'], data['sepal width (cm)'], 
                          c=colors[species], label=species, alpha=0.7)
    axes[0, 1].set_title('Sepal Length vs Width')
    axes[0, 1].set_xlabel('Sepal Length (cm)')
    axes[0, 1].set_ylabel('Sepal Width (cm)')
    axes[0, 1].legend()
    
    # 3. Box plot by species
    species_data = [iris_df[iris_df['species_name'] == species]['petal length (cm)'] 
                   for species in ['setosa', 'versicolor', 'virginica']]
    axes[1, 0].boxplot(species_data, labels=['setosa', 'versicolor', 'virginica'])
    axes[1, 0].set_title('Petal Length by Species')
    axes[1, 0].set_ylabel('Petal Length (cm)')
    
    # 4. Correlation heatmap
    numeric_cols = iris_df.select_dtypes(include=[np.number]).columns
    corr_matrix = iris_df[numeric_cols].corr()
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 1].set_title('Feature Correlation')
    axes[1, 1].set_xticks(range(len(numeric_cols)))
    axes[1, 1].set_yticks(range(len(numeric_cols)))
    axes[1, 1].set_xticklabels(numeric_cols, rotation=45)
    axes[1, 1].set_yticklabels(numeric_cols)
    
    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'data_analysis.png'")
    plt.show()

def main():
    """Main function to run all demonstrations"""
    print("📊 Data Foundations for AI/ML")
    print("=" * 50)
    
    # NumPy fundamentals
    numpy_fundamentals()
    
    # Pandas data manipulation
    pandas_data_manipulation()
    
    # Missing data handling
    handle_missing_data()
    
    # Real dataset exploration
    load_real_datasets()
    
    # Data visualization
    data_visualization_basics()
    
    print("\n" + "=" * 50)
    print("Lesson 2 Complete!")
    print("Next: Learn data preprocessing and feature engineering")
    print("Key takeaway: Clean, well-structured data is the foundation of good AI models")

if __name__ == "__main__":
    main()
