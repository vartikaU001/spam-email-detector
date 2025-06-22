import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from utils import VOCAB_SIZE

df = pd.read_csv('data/cleaned_data.csv')
texts = df['cleaned_text'].astype(str).tolist()
labels = df['label'].values

MAX_LEN = 100
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), callbacks=[EarlyStopping(patience=2)])
model.save('spam_lstm_model.h5')
print("✅ Model saved.")

with open('src/tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved.")
