import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# 1. Load dataset
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

# 2. Build the model using GNews Swivel embeddings
embedding_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

# Create the KerasLayer outside the Sequential model definition
# Wrap the KerasLayer in a Lambda layer to ensure compatibility
embedding_layer = hub.KerasLayer(embedding_url, input_shape=[], dtype=tf.string, trainable=True)
# This Lambda layer applies the embedding_layer to its input
embedding_lambda = tf.keras.layers.Lambda(lambda x: embedding_layer(x))


model = tf.keras.Sequential([
    embedding_lambda, # Use the Lambda layer wrapping the embedding_layer here
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 3. Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
history = model.fit(
    train_data.batch(32),
    validation_data=validation_data.batch(32),
    epochs=25
)

# 5. Evaluate on test data
loss, accuracy = model.evaluate(test_data.batch(32))
print(f"Test Accuracy: {accuracy:.4f}")

# 6. Try on new data
examples = [
    "What a great and thrilling movie!",
    "The plot was very boring and predictable."
]
# Convert the list of examples to a tf.data.Dataset
examples_dataset = tf.data.Dataset.from_tensor_slices(examples)

# Now you can use predict with the dataset
predictions = model.predict(examples_dataset.batch(1)) # Batch size 1 for individual predictions

for text, pred in zip(examples, predictions):
    print(f"{text} -> {'Positive' if pred[0] > 0.5 else 'Negative'}")
