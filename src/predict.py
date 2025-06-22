from utils import clean_text, load_tokenizer_from_cleaned_data
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100
model = load_model('spam_lstm_model.h5')
tokenizer = load_tokenizer_from_cleaned_data()

def predict_message(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded, verbose=0)[0][0]
    label = "SPAM" if pred > 0.5 else "NOT SPAM"
    return f"🔍 {label} (Confidence: {pred:.2f})"

if __name__ == "__main__":
    while True:
        msg = input("📩 Enter email text (or type 'exit'): ")
        if msg.lower() == "exit":
            break
        print(predict_message(msg))
