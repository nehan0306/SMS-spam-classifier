import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')


# Function to remove stop words, punctuation marks and to return the lemmatized statement
def new_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    txt = []
    for char in message:
        if char.isalnum() and char not in stopwords.words('english') and char not in string.punctuation:
            txt.append(char)

    text = []
    for word in txt:
        text.append(lemmatizer.lemmatize(word))

    return " ".join(text)



tfidf = pickle.load(open('vectorizer_sms.pkl', 'rb'))
model = pickle.load(open('model_sms.pkl', 'rb'))

st.title('SMS Spam Classifier')
input_sms = st.text_input('Enter the message')

if st.button('Predict'):
    trans_sms = new_message(input_sms)
    vect_inp = tfidf.transform([trans_sms])
    result = model.predict(vect_inp)

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
