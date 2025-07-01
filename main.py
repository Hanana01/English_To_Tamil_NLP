# main.py

import numpy as np
import tensorflow as tf
import pickle
import os
from sklearn.model_selection import train_test_split

# ------------------------------
# Step 1: Load and Prepare Data
# ------------------------------

with open('train.en', 'r', encoding='utf-8') as f:
    input_texts = ['<start> ' + line.strip() + ' <end>' for line in f]

with open('train.ta', 'r', encoding='utf-8') as f:
    target_texts = ['<start> ' + line.strip() + ' <end>' for line in f]

# Tokenize English
input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
input_tokenizer.fit_on_texts(input_texts)
input_tensor = input_tokenizer.texts_to_sequences(input_texts)
input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, padding='post')

# Tokenize Tamil
target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
target_tokenizer.fit_on_texts(target_texts)
target_tensor = target_tokenizer.texts_to_sequences(target_texts)
target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, padding='post')

# Save tokenizers
with open('inp_tokenizer.pkl', 'wb') as f:
    pickle.dump(input_tokenizer, f)
with open('targ_tokenizer.pkl', 'wb') as f:
    pickle.dump(target_tokenizer, f)

# Save metadata
meta = {
    'input_vocab_size': len(input_tokenizer.word_index) + 1,
    'target_vocab_size': len(target_tokenizer.word_index) + 1,
    'max_length_input': input_tensor.shape[1],
    'max_length_target': target_tensor.shape[1]
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

# ------------------------------
# Step 2: Define Model Classes
# ------------------------------

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super().__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state = self.gru(x)
        return output, state

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        return self.fc(output), state

# ------------------------------
# Step 3: Train-Test Split
# ------------------------------

input_train, input_val, target_train, target_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# ------------------------------
# Step 4: Create Dataset
# ------------------------------

BUFFER_SIZE = len(input_train)
BATCH_SIZE = 2 if BUFFER_SIZE < 64 else 64
steps_per_epoch = max(1, BUFFER_SIZE // BATCH_SIZE)

print(f"✅ Training samples: {BUFFER_SIZE}")
print(f"✅ Using batch size: {BATCH_SIZE}")
print(f"✅ Steps per epoch: {steps_per_epoch}")

dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# ------------------------------
# Step 5: Initialize Model
# ------------------------------

embedding_dim = 256
units = 512

encoder = Encoder(meta['input_vocab_size'], embedding_dim, units)
decoder = Decoder(meta['target_vocab_size'], embedding_dim, units)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.not_equal(real, 0)
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    return tf.reduce_mean(loss_ * mask)

@tf.function
def train_step(inp, targ):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

# ------------------------------
# Step 6: Train the Model
# ------------------------------

EPOCHS = 10
os.makedirs('checkpoints', exist_ok=True)

for epoch in range(EPOCHS):
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ)
        total_loss += batch_loss

    print(f"Epoch {epoch+1} / {EPOCHS} - Loss: {total_loss / steps_per_epoch:.4f}")

# ------------------------------
# Step 7: Save Model Weights
# ------------------------------

encoder.save_weights('checkpoints/encoder.weights.h5')
decoder.save_weights('checkpoints/decoder.weights.h5')

print("✅ Training complete. Weights saved to checkpoints/")
