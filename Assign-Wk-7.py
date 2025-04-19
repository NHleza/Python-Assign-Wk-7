# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset from seaborn's repository
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"  # Replace with your CSV file path
    data = pd.read_csv(url)
    
    # Display the first few rows
    print("First Few Rows of the Dataset:")
    print(data.head()) 
    
    # Explore the structure of the dataset
    print("\nDataset Info:")
    print(data.info())
    
    print("\nMissing Values Check:")
    print(data.isnull().sum())
    
    # Clean the dataset (no missing values in this example, but here's how to handle them)
    data.dropna(inplace=True)  # Removes rows with missing values
    # Alternatively, data.fillna(value, inplace=True) can fill missing values

except FileNotFoundError:
    print("Error: File not found. Please provide a valid dataset path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Task 2: Basic Data Analysis
# Compute basic statistics for numerical columns
print("\nBasic Statistics:")
print(data.describe())

# Perform groupings (mean petal_length per species)
species_grouping = data.groupby("species")["petal_length"].mean()
print("\nMean Petal Length per Species:")
print(species_grouping)

# Task 3: Data Visualization
# 1. Line Chart: Trend of sepal_length across the dataset
plt.figure(figsize=(8, 4))
plt.plot(data["sepal_length"], label="Sepal Length Trend")
plt.title("Sepal Length Trend Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length")
plt.legend()
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(8, 4))
species_grouping.plot(kind='bar', color='skyblue')
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (Mean)")
plt.show()

# 3. Histogram: Distribution of petal_length
plt.figure(figsize=(8, 4))
sns.histplot(data["petal_length"], kde=True, color='purple')
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 4))
sns.scatterplot(x=data["sepal_length"], y=data["petal_length"], hue=data["species"])
plt.title("Relationship Between Sepal Length and Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend(title="Species")
plt.show()
