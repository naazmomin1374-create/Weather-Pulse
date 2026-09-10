import pandas as pd

data = pd.read_csv("data/weather_data.csv")

print(data)

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:")
print(data.duplicated().sum())

# Features (Input)
X = data[["temperature", "humidity", "pressure", "wind_speed"]]

# Target (Output)
y = data["weather"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:")
print(X_train)

print("\nTesting data:")
print(X_test)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:")
print(X_train)

print("\nTesting data:")
print(X_test)

print("\nTraining labels:")
print(y_train)

print("\nTesting labels:")
print(y_test)

from sklearn.tree import DecisionTreeClassifier

# Create the model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Make predictions
predictions = model.predict(X_test)

print("\nPredicted weather:")
print(predictions)

print("\nActual weather:")
print(y_test.values)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(accuracy * 100, "%")

# Load event classification dataset
event_data = pd.read_csv("data/event_data.csv")

print("\nEvent Dataset:")
print(event_data.head())