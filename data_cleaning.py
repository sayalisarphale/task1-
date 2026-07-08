import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for graphs
os.makedirs("output_images", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset.csv")

print("Original Dataset:")
print(df.head())

# -------------------------
# Handle Missing Values
# -------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# -------------------------
# Remove Duplicates
# -------------------------
df.drop_duplicates(inplace=True)

# -------------------------
# Detect and Remove Outliers
# -------------------------
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

# -------------------------
# Save Cleaned Dataset
# -------------------------
df.to_csv("cleaned_dataset.csv", index=False)

# -------------------------
# Histogram
# -------------------------
plt.figure(figsize=(6,4))
df.hist(figsize=(10,8))
plt.tight_layout()
plt.savefig("output_images/histogram.png")
plt.close()

# -------------------------
# Box Plot
# -------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=df[numeric_columns])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output_images/boxplot.png")
plt.close()

# -------------------------
# Scatter Plot
# -------------------------
if len(numeric_columns) >= 2:
    plt.figure(figsize=(6,4))
    plt.scatter(df[numeric_columns[0]], df[numeric_columns[1]])
    plt.xlabel(numeric_columns[0])
    plt.ylabel(numeric_columns[1])
    plt.tight_layout()
    plt.savefig("output_images/scatterplot.png")
    plt.close()

# -------------------------
# Correlation Heatmap
# -------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm")
plt.tight_layout()
plt.savefig("output_images/heatmap.png")
plt.close()

print("Data cleaning completed successfully.")
print("Cleaned dataset saved as cleaned_dataset.csv")
print("Graphs saved inside output_images folder.")
