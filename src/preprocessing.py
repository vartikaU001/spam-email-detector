import pandas as pd
from utils import clean_text
import nltk
nltk.download('stopwords')

df = pd.read_csv('data/combined_data.csv')
df = df.dropna(subset=['label', 'text'])

if df['label'].dtype == 'object':
    df['label'] = df['label'].str.lower().map({'spam': 1, 'ham': 0, 'not spam': 0, 'legit': 0})
df = df.dropna(subset=['label'])

df['cleaned_text'] = df['text'].astype(str).apply(clean_text)
df.to_csv('data/cleaned_data.csv', index=False)
print("✅ Cleaned data saved.")
