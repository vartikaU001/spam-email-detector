import pandas as pd

# Load cleaned data
df = pd.read_csv('data/cleaned_data.csv')

print("✅ File loaded successfully!\n")

# Show column names
print("📌 Column names:")
print(df.columns)

# Show first 5 rows
print("\n📄 Sample data:")
print(df.head())

# Check for missing values
print("\n🔍 Null values in each column:")
print(df.isnull().sum())

# Check label distribution
print("\n📊 Label distribution:")
print(df['label'].value_counts())

# Unique values in label column
print("\n🔢 Unique label values:")
print(df['label'].unique())
