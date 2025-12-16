import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

nltk.download('punkt')

df = pd.read_csv("imdb_top_1000.csv")
texts = df['Overview'].astype(str).values
labels = df['Genre'].astype(str).values
labels = [g.split(",")[0].strip() for g in labels]

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

texts = [" ".join(word_tokenize(t.lower())) for t in texts]

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=100, padding='post')

x_train, x_test, y_train, y_test = train_test_split(
    padded, labels_encoded, test_size=0.2, random_state=42
)

model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    LSTM(64),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

def predict_genre(text):
    tokens = " ".join(word_tokenize(text.lower()))
    seq = tokenizer.texts_to_sequences([tokens])
    padded_seq = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(padded_seq)[0]
    predicted_class = np.argmax(pred)
    return label_encoder.inverse_transform([predicted_class])[0]

print(predict_genre("A young wizard begins his journey at a school of magic."))
print(predict_genre("An undercover cop infiltrates a powerful crime syndicate."))

model.save("movie_genre_classifier.h5")
print("Model saved as movie_genre_classifier.h5")



with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)



nltk.download('punkt')


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

max_len = 100
model = load_model("movie_genre_classifier.h5")

st.title("🎬 Movie Genre Classifier")
st.write("Enter a movie overview and I’ll predict its genre")

text_input = st.text_area("Movie Overview:")

if st.button("Predict Genre"):
    if text_input.strip() == "":
        st.warning("Please enter a movie description.")
    else:
        tokens = " ".join(word_tokenize(text_input.lower()))
        seq = tokenizer.texts_to_sequences([tokens])
        padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
        pred = model.predict(padded_seq)[0]
        predicted_class = np.argmax(pred)
        genre = label_encoder.inverse_transform([predicted_class])[0]
        st.success(f"Predicted Genre: **{genre}**")
