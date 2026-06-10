# ============================================
# PROJECT 2: Handwritten Digit Recognizer
# Using: TensorFlow, NumPy, Matplotlib
# Dataset: MNIST (loads automatically)
# ============================================

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

print(f"TensorFlow version: {tf.__version__}")

# -------------------------------
# 2. LOAD MNIST DATASET (Auto-download)
# -------------------------------
print("\nLoading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training images: {X_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Test images: {X_test.shape}")
print(f"Test labels: {y_test.shape}")

# -------------------------------
# 3. EXPLORE THE DATA
# -------------------------------
print("\n" + "="*50)
print("DATA EXPLORATION")
print("="*50)

print(f"\nImage shape: {X_train[0].shape} (28x28 pixels)")
print(f"Pixel value range: {X_train[0].min()} to {X_train[0].max()}")
print(f"Label example (first image): {y_train[0]}")

# Count digits in dataset
unique, counts = np.unique(y_train, return_counts=True)
print("\nDigit distribution in training set:")
for digit, count in zip(unique, counts):
    print(f"  Digit {digit}: {count} images")

# -------------------------------
# 4. VISUALIZE DIGITS WITH MATPLOTLIB
# -------------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(X_train[i], cmap='gray')
    axes[i].set_title(f"Digit: {y_train[i]}")
    axes[i].axis('off')

plt.suptitle('Sample Handwritten Digits from MNIST Dataset')
plt.tight_layout()
plt.show()

# Show multiple examples of the same digit (digit 5)
digit_5_indices = np.where(y_train == 5)[0][:10]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

for i, idx in enumerate(digit_5_indices):
    axes[i].imshow(X_train[idx], cmap='gray')
    axes[i].set_title(f"Digit 5 - Sample {i+1}")
    axes[i].axis('off')

plt.suptitle('Multiple Examples of the Digit "5"')
plt.tight_layout()
plt.show()

# -------------------------------
# 5. PREPROCESS DATA
# -------------------------------
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Normalize pixel values (0-255 → 0-1)
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0

print(f"Before normalization: min={X_train.min()}, max={X_train.max()}")
print(f"After normalization: min={X_train_normalized.min():.2f}, max={X_train_normalized.max():.2f}")

# Reshape for TensorFlow (not needed but good to see)
print(f"\nOriginal shape: {X_train_normalized.shape}")
print("No reshaping needed for this model - TensorFlow handles it")

# -------------------------------
# 6. BUILD NEURAL NETWORK
# -------------------------------
print("\n" + "="*50)
print("BUILDING NEURAL NETWORK")
print("="*50)

model = tf.keras.Sequential([
    # Flatten 28x28 image to 784 pixels
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Hidden layer with 128 neurons
    tf.keras.layers.Dense(128, activation='relu', name='hidden_layer'),
    
    # Dropout to prevent overfitting (randomly turns off 20% of neurons)
    tf.keras.layers.Dropout(0.2),
    
    # Output layer: 10 neurons (digits 0-9)
    tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
])

# Show model architecture
model.summary()

# -------------------------------
# 7. COMPILE MODEL
# -------------------------------
print("\n" + "="*50)
print("COMPILING MODEL")
print("="*50)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")

# -------------------------------
# 8. TRAIN MODEL
# -------------------------------
print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

history = model.fit(
    X_train_normalized, y_train,
    epochs=10,  # 10 passes through the data
    validation_split=0.2,  # Use 20% of training for validation
    verbose=1
)

print("\nTraining complete!")

# -------------------------------
# 9. EVALUATE ON TEST DATA
# -------------------------------
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# -------------------------------
# 10. PLOT TRAINING HISTORY
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# 11. MAKE PREDICTIONS (Test images)
# -------------------------------
print("\n" + "="*50)
print("MAKING PREDICTIONS")
print("="*50)

# Pick 10 random test images
random_indices = np.random.choice(len(X_test), 10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i, idx in enumerate(random_indices):
    # Get image and true label
    image = X_test[idx]
    true_label = y_test[idx]
    
    # Predict
    prediction = model.predict(image.reshape(1, 28, 28), verbose=0)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Display
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f"True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.1f}%")
    axes[i].axis('off')
    
    # Color code: green=correct, red=wrong
    if predicted_label == true_label:
        axes[i].title.set_color('green')
    else:
        axes[i].title.set_color('red')

plt.suptitle('Model Predictions on Test Images (Green=Correct, Red=Wrong)')
plt.tight_layout()
plt.show()

# -------------------------------
# 12. SHOW WHERE MODEL FAILS
# -------------------------------
print("\n" + "="*50)
print("ANALYZING ERRORS")
print("="*50)

# Get all predictions on test set
predictions = model.predict(X_test_normalized, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

# Find misclassified examples
misclassified = np.where(predicted_labels != y_test)[0]

print(f"Total test images: {len(y_test)}")
print(f"Correct predictions: {len(y_test) - len(misclassified)}")
print(f"Wrong predictions: {len(misclassified)}")
print(f"Error rate: {len(misclassified)/len(y_test)*100:.2f}%")

# Show first 5 misclassified examples
if len(misclassified) > 0:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes = axes.ravel()
    
    for i, idx in enumerate(misclassified[:5]):
        axes[i].imshow(X_test[idx], cmap='gray')
        axes[i].set_title(f"True: {y_test[idx]}\nPred: {predicted_labels[idx]}")
        axes[i].axis('off')
    
    plt.suptitle('Examples of Misclassified Digits')
    plt.tight_layout()
    plt.show()

# -------------------------------
# 13. CLASSIFICATION REPORT
# -------------------------------
print("\n" + "="*50)
print("CLASSIFICATION REPORT (Per-digit performance)")
print("="*50)

print(classification_report(y_test, predicted_labels, 
                            target_names=[str(i) for i in range(10)]))

# -------------------------------
# 14. SAVE MODEL (Optional)
# -------------------------------
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

# Save as .h5 file (you can reuse later)
model.save('digit_recognizer_model.h5')
print("Model saved as 'digit_recognizer_model.h5'")

# -------------------------------
# 15. FUNCTION TO PREDICT YOUR OWN DIGITS
# -------------------------------
def predict_digit(image_array):
    """
    Predict a single digit from a 28x28 array
    """
    if image_array.max() > 1:
        image_array = image_array / 255.0
    
    prediction = model.predict(image_array.reshape(1, 28, 28), verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return digit, confidence

# Test with a sample
sample_image = X_test[0]
pred_digit, conf = predict_digit(sample_image)
print(f"\nTest prediction on first test image: Predicted {pred_digit} with {conf:.1f}% confidence")

print("\n" + "="*50)
print("PROJECT COMPLETE!")
print("="*50)