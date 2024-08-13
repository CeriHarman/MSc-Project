import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

#generate the data - using input from the 'three basic scenarios'
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

# prep the data
data = np.array(data)
labels = np.array(labels)

scaler = StandardScaler()
data = scaler.fit_transform(data)

label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)
one_hot_labels = to_categorical(integer_encoded_labels)

# split the data
X_train, X_test, y_train, y_test = train_test_split(data, one_hot_labels, test_size=0.3, random_state=42)

# MLP
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

# train
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

#loss
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# predict for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
report = classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_)
print('Classification Report:')
print(report)

# Plot Loss/Epoch Graph
plt.figure(figsize=(10, 6))

# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Loss Vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
