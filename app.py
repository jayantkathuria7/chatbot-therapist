import streamlit as st
import numpy as np
import pandas as pd
import re
from keras.preprocessing.sequence import pad_sequences
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import json

# Load your trained model
model = load_model('chatbot_therapist.keras')

# Load tokenizer and label encoder
with open('intents.json', 'r') as f:
    data = json.load(f)

dic = {"tag": [], "patterns": [], "responses": []}
for example in data['intents']:
    for pattern in example['patterns']:
        dic['patterns'].append(pattern)
        dic['tag'].append(example['tag'])
        dic['responses'].append(example['responses'])

df = pd.DataFrame.from_dict(dic)

tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])

ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')
lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])


# Function to preprocess and generate responses
def generate_answer(tokenizer, model, lbl_enc, df, pattern):
    pattern = pattern.lower()

    # Preprocess input pattern
    text = [pattern]
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)  # Remove non-alphabetic characters except apostrophes
    txt = txt.lower()  # Convert to lowercase
    txt = txt.split()  # Split into words
    txt = " ".join(txt)  # Join back into a single string
    text = [txt]  # Place in a list (needed for texts_to_sequences)

    # Convert text to sequences and pad
    x_test = tokenizer.texts_to_sequences(text)
    x_test = pad_sequences(x_test, padding='post', maxlen=X.shape[1])

    # Predict using the model
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=-1)[0]  # Get the index of the highest probability

    # Decode the predicted tag to get responses
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]

    # Return a random response from possible responses
    return random.choice(responses)


# Streamlit app code with enhanced chatbot interface
def main():
    st.title('Chatbot Therapist')

    st.markdown("""
        Welcome to the Chatbot Therapist! Type your message below and press Enter to chat.
        Type **quit** to end the chat.
    """)

    # Input area for user to type messages
    user_input = st.text_input('You:')

    if user_input:
        if user_input.lower() == 'quit':
            st.text('Chat ended.')
        else:
            # Display user input in a chat bubble
            st.text_area('You:', value=user_input, height=50)

            # Generate and display model response
            response = generate_answer(tokenizer, model, lbl_enc, df, user_input)
            st.text_area('Model:', value=response, height=100)

    # Add a divider to separate chat area
    st.markdown("---")


if __name__ == '__main__':
    main()
