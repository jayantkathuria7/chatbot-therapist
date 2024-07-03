# Chatbot Therapist

## Overview
This repository contains a project to build a Chatbot Therapist using a dataset sourced from Kaggle. The goal of the project is to create a conversational agent that can engage users in therapeutic dialogue based on patterns learned from the dataset.

## Demo
Check out the live demo of the Chatbot Therapist [here](https://chatbot-therapist.streamlit.app/).

## Dataset
The dataset used in this project is obtained from [Kaggle](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data). It consists of (describe briefly what the dataset contains, e.g., text conversations between therapists and patients, labeled with emotions or topics).

## Project Structure
- intents.json: Datset used.
- **[Chatbot Therapist.ipynb](https://github.com/jayantkathuria7/chatbot-therapist/blob/master/Chatbot%20Therapist.ipynb)** : Jupyter notebook containing source code containing scripts and modules for data processing, model training.
- **[app.py](https://github.com/jayantkathuria7/chatbot-therapist/blob/master/app.py)**: Source code containing python scripts using streamlit of the interface.
- **README.md**: This file, providing an overview of the project, setup instructions, and usage details.
- **requirements.txt**: List of Python packages required to run the project.

## Setup Instructions
1. **Clone the repository:**
   ```
   git clone https://github.com/your_username/chatbot-therapist.git
   cd chatbot-therapist
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data)


4. **Running the Chatbot:**
  ```
  streamlit run app.py
  ```
  Launches the interactive chatbot interface where users can engage in a conversation with the therapist.
