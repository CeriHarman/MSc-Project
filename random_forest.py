import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

#generate synthetic data - using information from the 'three basic' scenarios
def generate_synthetic_data(num_samples, scenario):
    data = []
    for _ in range(num_samples):
        if scenario == "increasing":
            diversity = np.random.normal(loc=0.000095, scale=0.000020)
            tajimas_d = np.random.normal(loc=0.0012, scale=0.36)
            wattersons_theta = np.random.normal(loc=9.5, scale=1.9)
            num_snps = np.random.normal(loc=26.8, scale=5.4)

        elif scenario == "decreasing":
            diversity = np.random.normal(loc=0.00032, scale=0.00004)
            tajimas_d = np.random.normal(loc=0.03, scale=0.20)
            wattersons_theta = np.random.normal(loc=31.7, scale=3.7)
            num_snps = np.random.normal(loc=89.8, scale=10.4)

        elif scenario == "stable":
            diversity = np.random.normal(loc=0.00040, scale=0.00004)
            tajimas_d = np.random.normal(loc=-0.003, scale=0.19)
            wattersons_theta = np.random.normal(loc=40.3, scale=3.6)
            num_snps = np.random.normal(loc=114.0, scale=10.3)
            
        afs = np.random.dirichlet(np.ones(5))
        feature_vector = [diversity, tajimas_d, wattersons_theta, num_snps] + list(afs)
        data.append(feature_vector)
    return data

num_samples_per_scenario = 100

increasing_population_data = generate_synthetic_data(num_samples_per_scenario, "increasing")
decreasing_population_data = generate_synthetic_data(num_samples_per_scenario, "decreasing")
stable_population_data = generate_synthetic_data(num_samples_per_scenario, "stable")

data = increasing_population_data + decreasing_population_data + stable_population_data
labels = ['increasing'] * num_samples_per_scenario + ['decreasing'] * num_samples_per_scenario + ['stable'] * num_samples_per_scenario

#sort the data
data = np.array(data)
labels = np.array(labels)

scaler = StandardScaler()
data = scaler.fit_transform(data)

#split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

#best params
print(f"Best parameters found: {grid.best_params_}")


y_pred = grid.predict(X_test)
print("RandomForest Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=['increasing', 'decreasing', 'stable'])

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['increasing', 'decreasing', 'stable'], yticklabels=['increasing', 'decreasing', 'stable'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model
joblib.dump(grid, 'random_forest_population_model.pkl')

# Load the model 
loaded_model = joblib.load('random_forest_population_model.pkl')

# New sample values - input summary stat values here
new_sample_values = [
    3.91488888888889e-05,   #nucelotide diversity
    0.023968560194865148,   #tajimas d
    3.9095525319119093,     #wattersons theta 
    11.06,                  #number of snps
    0.2, 0.2, 0.2, 0.2, 0.2 #afs
]
new_sample = scaler.transform([new_sample_values])
prediction = loaded_model.predict(new_sample)
print(f"Predicted population scenario: {prediction[0]}")
