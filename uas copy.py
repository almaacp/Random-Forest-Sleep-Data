#%% Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree

path_dataset = 'Sleep_health_and_lifestyle_dataset.csv'

#%% Load the dataset
dataset = pd.read_csv(path_dataset)

#%% Display basic information
print("Dataset Head:")
print(dataset.head())

print("\nDataset Info:")
print(dataset.info())

#%% Handle missing values (drop rows with NaN)
print("Check Missing Values Before Cleaning:")
print(dataset.isnull().sum())

# Handle missing values (replace NaN in 'Sleep Disorder' with 'none')
dataset['Sleep Disorder'] = dataset['Sleep Disorder'].fillna('none')

# Drop rows with NaN values in other columns
dataset = dataset.dropna(subset=[col for col in dataset.columns if col != 'Sleep Disorder'])

print("Check Missing Values After Cleaning:")
print(dataset.isnull().sum())

#%% Plot Class Distribution
plt.figure(figsize=(8, 6))
class_counts = dataset['Sleep Disorder'].value_counts()
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
for i, count in enumerate(class_counts.values):
    plt.text(i, count + 2, f'{count:.1f}', ha='center', fontsize=12)
plt.title('Distribusi Kelas', fontsize=16)
plt.xlabel('Gangguan Tidur', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)
plt.show()

#%% Descriptive statistics
descriptive_stats = dataset.describe()

# Unique values in each categorical column
unique_values = {}
cat_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder', 'Blood Pressure']
for col in cat_columns:
    unique_values[col] = dataset[col].unique()

print(descriptive_stats)
print(unique_values)

#%% Data Transformation and Cleaning 

# Correcting the inconsistency in 'BMI Category'
dataset['BMI Category'].replace({'Normal Weight': 'Normal'}, inplace=True)

# Splitting the 'Blood Pressure' column into 'Systolic' and 'Diastolic' columns
dataset['Systolic'] = dataset['Blood Pressure'].str.split('/').str[0].astype(int)
dataset['Diastolic'] = dataset['Blood Pressure'].str.split('/').str[1].astype(int)

dataset.drop(['Blood Pressure'], axis=1, inplace=True)

sns.set(style="white")
sns.set_palette(palette='Set3')

# List of key numerical variables
num_vars = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()

#%% Correlation matrix
corr_matrix = dataset[num_vars].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation matrix of numerical variables', fontsize=16)
plt.show()

#%% Preprocessing: Select features and label
X = dataset.drop(columns=['Sleep Disorder'])
y = dataset['Sleep Disorder']

#%% One-hot encoding (if necessary)
X = pd.get_dummies(X, drop_first=True)

#%% Test different train-test splits
splits = {
    "80:20": 0.2,
    "75:25": 0.25,
    "60:40": 0.4
}

results = {}

for split_name, test_size in splits.items():
    print(f"\n### Evaluating Split: {split_name} ###")
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict labels for the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[split_name] = accuracy
    print(f"\nAkurasi Random Forest (Split {split_name}): {accuracy:}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(f"Confusion Matrix Heatmap ({split_name})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualization of Random Forest accuracy
    plt.figure(figsize=(6, 4))
    plt.bar([split_name], [accuracy], color='royalblue')
    plt.ylim(0, 1)
    for i, v in enumerate([accuracy]):
        plt.text(i, v + 0.01, f"{v:}", ha='center', fontsize=12)
    plt.title(f"Akurasi Random Forest ({split_name}): {accuracy:}")
    plt.ylabel("Akurasi")
    plt.show()
    
    # Visualize a Decision Tree from Random Forest
    plt.figure(figsize=(15, 10))
    tree = model.estimators_[0]
    plot_tree(tree, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True, fontsize=10)
    plt.title(f'Decision Tree Visualization ({split_name})')
    plt.show()
