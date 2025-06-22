# 📧 Spam Email Detector using LSTM

This is a **deep learning-based spam detection web app** built with **TensorFlow, LSTM**, and **Streamlit**.  
The model detects whether an email message is **Spam** or **Not Spam** using natural language processing techniques.

![demo](https://img.shields.io/badge/Machine%20Learning-LSTM-blueviolet) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Python](https://img.shields.io/badge/Python-3.9+-blue)

---

## 🚀 Features

- 🧐 Trained on real-world spam emails using LSTM (Long Short-Term Memory)
- ⚙️ Tokenizer and model optimized for **fast, real-time predictions**
- 📊 Dynamic **confidence meter** using Plotly
- 🕵️‍♀️ **AI Forensic Summary** to explain model results

---

## 📁 Project Structure

```bash
📂 Spam Email Detector/
🗄 src/
│   ├── streamlit_app.py           # Main Streamlit UI app
│   ├── model.py                   # Model training code
│   ├── preprocessing.py           # Dataset cleaning pipeline
│   ├── predict.py                 # CLI predictor
│   ├── utils.py                   # Helper functions (clean_text, tokenizer loader)
│   ├── tokenizer.pickle           # Saved tokenizer (used at prediction time)
│   └── spam_lstm_model.h5         # Trained LSTM model
🗄 data/
│   └── sample_data.csv            # Optional: Small sample data (full data ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/vartikaU001/spam-email-detector.git
cd spam-email-detector/src
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r ../requirements.txt
```

### 3. Download the Dataset
The dataset used in this project was downloaded from Kaggle:  
**SMS Spam Collection Dataset**

Place the full dataset CSV in `data/combined_data.csv`, then run:

```bash
python preprocessing.py
python model.py
```
This will clean the data and train the model.

---

## ▶️ Run the App
```bash
streamlit run streamlit_app.py
```

---

## 🔪 Test via Command Line (optional)
```bash
python predict.py
```

---

## 📆 Requirements
Make sure you have Python 3.9+ installed.

```txt
numpy
pandas
matplotlib
scikit-learn
tensorflow
nltk
streamlit
plotly
```
All packages are listed in `requirements.txt`.

---

## 🙌 Author
**Vartika Upadhyay**  
🌐 GitHub: [github.com/vartikaU001](https://github.com/vartikaU001)  
📝 Built with ❤️ using LSTM + Streamlit
