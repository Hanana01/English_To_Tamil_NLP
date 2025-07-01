# translator.py

import tensorflow as tf
import numpy as np
import pickle

# ----------------------
# Encoder, Attention, Decoder Classes
# ----------------------

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

# ----------------------
# Load Tokenizers and Metadata
# ----------------------

with open('inp_tokenizer.pkl', 'rb') as f:
    inp_tokenizer = pickle.load(f)
with open('targ_tokenizer.pkl', 'rb') as f:
    targ_tokenizer = pickle.load(f)
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)

input_vocab_size = meta['input_vocab_size']
target_vocab_size = meta['target_vocab_size']
max_length_input = meta['max_length_input']
max_length_target = meta['max_length_target']

embedding_dim = 256
units = 512

# ----------------------
# Build and Load Encoder/Decoder
# ----------------------

encoder = Encoder(input_vocab_size, embedding_dim, units)
decoder = Decoder(target_vocab_size, embedding_dim, units)

# Build models with dummy inputs before loading weights
dummy_input = tf.zeros((1, max_length_input), dtype=tf.int32)
dummy_target = tf.zeros((1, 1), dtype=tf.int32)
enc_output, enc_hidden = encoder(dummy_input)
_ = decoder(dummy_target, enc_hidden, enc_output)

# Load weights
encoder.load_weights('checkpoints/encoder.weights.h5')
decoder.load_weights('checkpoints/decoder.weights.h5')

# ----------------------
# Evaluate Function
# ----------------------

def evaluate(sentence):
    sentence = '<start> ' + sentence.strip() + ' <end>'
    inputs = inp_tokenizer.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    enc_out, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']], 0)

    for _ in range(max_length_target):
        predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = targ_tokenizer.index_word.get(predicted_id, '')

        if predicted_word == '<end>':
            break

        result += predicted_word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip()

def translate(sentence):
    return evaluate(sentence)

# ----------------------
# Command Line Interface
# ----------------------

if __name__ == "__main__":
    print("English to Tamil Translator (type 'exit' to quit)\n")
    while True:
        inp = input("Input (English): ")
        if inp.lower() == 'exit':
            break
        translation = translate(inp)
        print("Output (Tamil):", translation)
        print("-" * 50)
