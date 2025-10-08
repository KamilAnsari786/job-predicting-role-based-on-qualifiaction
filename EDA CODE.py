
# Import libraries

import pandas as pd
import numpy as np
import random
from google.colab import files  # for downloading in Colab
import seaborn as sns
import matplotlib.pyplot as plt

# Generate dataset (1000 rows)

degrees = ['B.Sc', 'M.Sc', 'B.Tech', 'M.Tech', 'bsc', 'msc', 'Bsc', 'Msc', None]
majors = ['Computer Science', 'Data Science', 'Information Technology',
          'Software Engineering', 'Artificial Intelligence', None, 'Data scienc', 'Comp Science']
skills1 = ['Python', 'Java', 'C++', 'HTML', 'JavaScript', 'SQL', None, 'python', 'java']
skills2 = ['Machine Learning', 'React', 'Django', 'Node.js', 'Deep Learning', 'Flask', None, 'react', 'flask']
job_roles = ['Data Scientist', 'Frontend Developer', 'Backend Developer',
             'Full Stack Developer', 'AI Engineer', 'Software Engineer', None]
certifications = ['TensorFlow Certified', 'Certified Web Developer', 'AWS Certified',
                  'Google Cloud Certified', 'Microsoft Azure Certified', 'Spring Certified', None]

data = {
    "S. No.": range(1, 1001),
    "Degree": [random.choice(degrees) for _ in range(1000)],
    "Major": [random.choice(majors) for _ in range(1000)],
    "Skill1": [random.choice(skills1) for _ in range(1000)],
    "Skill2": [random.choice(skills2) for _ in range(1000)],
    "JobRole": [random.choice(job_roles) for _ in range(1000)],
    "Certification": [random.choice(certifications) for _ in range(1000)],
    "ExperienceYears": [abs(int(np.random.normal(5, 3))) for _ in range(1000)]
}

df = pd.DataFrame(data)


df = pd.concat([df, df.sample(10, random_state=42)], ignore_index=True)


outlier_indices = np.random.choice(df.index, size=5, replace=False)
df.loc[outlier_indices, 'ExperienceYears'] = [50, 60, 70, 80, 90]


print("✅ Raw dataset info:")
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())

print("\n--- Numerical Features ---")
print(df.describe())

# Visualize numerical feature
sns.boxplot(x=df['ExperienceYears'])
plt.title("Boxplot of ExperienceYears")
plt.show()


# EDA - Categorical Features

print("\n--- Categorical Features ---")
for col in ['Degree','Major','Skill1','Skill2','JobRole','Certification']:
    print(f"\nValue counts for {col}:\n", df[col].value_counts(dropna=False))

# Handling Missing Values

# For simplicity, fill categorical missing values with 'Unknown'
categorical_cols = ['Degree','Major','Skill1','Skill2','JobRole','Certification']
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# For numerical, fill with median
df['ExperienceYears'] = df['ExperienceYears'].fillna(df['ExperienceYears'].median())

print("\n✅ Missing values after filling:")
print(df.isnull().sum())


# Handling Outliers (simple cap method)

mean_exp = df['ExperienceYears'].mean()
std_exp = df['ExperienceYears'].std()
upper_limit = mean_exp + 3*std_exp
lower_limit = mean_exp - 3*std_exp

df['ExperienceYears'] = np.where(df['ExperienceYears']>upper_limit, upper_limit, df['ExperienceYears'])
df['ExperienceYears'] = np.where(df['ExperienceYears']<lower_limit, lower_limit, df['ExperienceYears'])


# Remove duplicates

df = df.drop_duplicates().reset_index(drop=True)


# Show cleaned dataset

print("\n✅ Cleaned dataset info:")
print(df.head())
print("\nRows after cleaning:", df.shape[0])
print("Columns:", df.shape[1])


# Save and download cleaned dataset

file_name = "job_prediction_cleaned_dataset.csv"
df.to_csv(file_name, index=False)
files.download(file_name)



# FOR VISUALIZATION 

# Enable inline plots
%matplotlib inline
sns.set(style="whitegrid", palette="pastel")


# Generate Raw Dataset (Same as Before)

degrees = ['B.Sc', 'M.Sc', 'B.Tech', 'M.Tech', 'bsc', 'msc', 'Bsc', 'Msc', None]
majors = ['Computer Science', 'Data Science', 'Information Technology',
          'Software Engineering', 'Artificial Intelligence', None, 'Data scienc', 'Comp Science']
skills1 = ['Python', 'Java', 'C++', 'HTML', 'JavaScript', 'SQL', None, 'python', 'java']
skills2 = ['Machine Learning', 'React', 'Django', 'Node.js', 'Deep Learning', 'Flask', None, 'react', 'flask']
job_roles = ['Data Scientist', 'Frontend Developer', 'Backend Developer',
             'Full Stack Developer', 'AI Engineer', 'Software Engineer', None]
certifications = ['TensorFlow Certified', 'Certified Web Developer', 'AWS Certified',
                  'Google Cloud Certified', 'Microsoft Azure Certified', 'Spring Certified', None]

data = {
    "S. No.": range(1, 1001),
    "Degree": [random.choice(degrees) for _ in range(1000)],
    "Major": [random.choice(majors) for _ in range(1000)],
    "Skill1": [random.choice(skills1) for _ in range(1000)],
    "Skill2": [random.choice(skills2) for _ in range(1000)],
    "JobRole": [random.choice(job_roles) for _ in range(1000)],
    "Certification": [random.choice(certifications) for _ in range(1000)],
    "ExperienceYears": [abs(int(np.random.normal(5, 3))) for _ in range(1000)]
}

df = pd.DataFrame(data)
df = pd.concat([df, df.sample(10, random_state=42)], ignore_index=True)
outlier_indices = np.random.choice(df.index, size=5, replace=False)
df.loc[outlier_indices, 'ExperienceYears'] = [50, 60, 70, 80, 90]

# Visualize Missing Values

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="Reds")
plt.title("Missing Values in Raw Dataset", fontsize=14)
plt.show()


# Numerical Feature Distribution

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df["ExperienceYears"], kde=True)
plt.title("Distribution of ExperienceYears (Raw)")

plt.subplot(1, 2, 2)
sns.boxplot(x=df["ExperienceYears"])
plt.title("Boxplot of ExperienceYears (Raw)")
plt.show()


# Categorical Feature Distributions

categorical_cols = ['Degree', 'Major', 'Skill1', 'Skill2', 'JobRole', 'Certification']

for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    df[col].value_counts(dropna=False).head(10).plot(kind='bar', color='skyblue')
    plt.title(f"Top Categories in {col}")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Cleaning Process

# Fill missing values
df[categorical_cols] = df[categorical_cols]()()
