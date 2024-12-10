#%% Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#%% Input dataset path
path_dataset = 'Sleep_health_and_lifestyle_dataset.csv'

#%% Load the dataset
dataset = pd.read_csv(path_dataset)

#%% Display basic information
print("Dataset Head:")
print(dataset.head())

print("\nDataset Info:")
print(dataset.info())

#%% Handle missing values (drop rows with NaN)
print("Before cleaning:")
print(dataset.isnull().sum())
dataset = dataset.dropna()
print("After cleaning:")
print(dataset.isnull().sum())

#%% Data visualization: Count plot
# Assuming a categorical column named 'Sleep Disorder'
sns.countplot(data=dataset, x='Sleep Disorder')  # Change 'Sleep Disorder' to the appropriate column name
plt.title("Count Plot")
plt.show()

#%% Data visualization: Heatmap of correlations
# Select only numeric columns
numeric_data = dataset.select_dtypes(include=['number'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Correlation")
plt.show()

#%% Preprocessing: Select features and label
X = dataset.drop(columns=['Sleep Disorder'])  # Change 'Sleep Disorder' to your label column name
y = dataset['Sleep Disorder']  # Change 'Sleep Disorder' to your label column name

#%% One-hot encoding (if necessary)
X = pd.get_dummies(X, drop_first=True)

#%% Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#%% Predict labels for the test set
y_pred = model.predict(X_test)

#%% Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAkurasi Random Forest (Testing):", accuracy)

#%% Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%% Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

#%% Plot confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix Heatmap")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

#%% Visualization Accuration of Random Forest
plt.figure(figsize=(6, 4))
plt.bar(['Random Forest'], [accuracy], color='royalblue')
plt.ylim(0, 1)
for i, v in enumerate([accuracy]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=12)
plt.title(f"Akurasi Random Forest: {accuracy:.2f}")
plt.ylabel("Akurasi")
plt.show()