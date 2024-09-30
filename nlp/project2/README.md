## Setup

- You may want to set up a new virtual environment with `python3 -m venv env`, then install required packages with `python3 -m pip install -r requirements.txt`.
- Install Tensorflow and required Nvidia tools for CUDA from [here](https://www.tensorflow.org/install/pip).
- Install nltk modules with `nltk.download('popular')` and `nltk.download('vader_lexicon')` in a python console.
- Install Spacy language model with `python3 -m spacy download en_core_web_md`.

## Start the chatbot

- A trained model is saved as "model.keras". To train the model again, use `python3 train_model.py`.
- A Knowledge base is saved as "knowledge_base.pkl" and "knowledge_base.txt". To create the Knowledge base again, use `python3 knowledge_base.py`.
- Start the chatbot with `python3 chatbot.py`.
