# Chatbot Therapist

## Overview
The **Chatbot Therapist** project aims to build a conversational agent capable of engaging users in therapeutic dialogue. The chatbot leverages machine learning models trained on a dataset of text conversations between therapists and patients to generate empathetic and supportive responses. This project demonstrates the potential of AI in mental health applications.

## Demo
You can interact with the live **Chatbot Therapist** [here](https://chatbot-therapist.streamlit.app/).

## Dataset
The dataset used for training the chatbot is sourced from [Kaggle](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data). It contains a collection of mental health conversation data, with labeled emotions and therapeutic topics. The dataset includes dialogue between therapists and patients, which is used to train the chatbot to recognize and respond appropriately to user inputs.

## Project Structure

- **intents.json**: Contains the dataset used for training, structured with intents, patterns, and responses.
- **[Chatbot Therapist.ipynb](https://github.com/jayantkathuria7/chatbot-therapist/blob/master/Chatbot%20Therapist.ipynb)**: Jupyter notebook containing the full source code for data processing, model training, and evaluation.
- **[app.py](https://github.com/jayantkathuria7/chatbot-therapist/blob/master/app.py)**: Python script for the chatbot’s Streamlit interface, allowing users to interact with the trained model in real time.
- **README.md**: This file, providing an overview, setup instructions, and usage details.
- **requirements.txt**: List of Python dependencies required to run the project.

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone https://github.com/jayantkathuria7/chatbot-therapist.git
   cd chatbot-therapist
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data).

4. **Running the Chatbot:**
   After installing the dependencies and downloading the dataset, run the following command to start the Streamlit app:
   ```
   streamlit run app.py
   ```
   This will launch the interactive chatbot interface in your web browser, where you can start a conversation with the therapist.

## Future Improvements
- Adding support for additional languages.
- Incorporating sentiment analysis for more personalized responses.
- Expanding the dataset to include more diverse conversational patterns.
- Integrating a more advanced NLP model for improved contextual understanding.

## Acknowledgments
- The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data).
- The project uses **Streamlit** for building the user interface and **scikit-learn** for machine learning model implementation.
