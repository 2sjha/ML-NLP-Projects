# NLP Project - 1

## Setup and Run

1. `python -m venv env` and `source ./env/bin/activate` for virtual env creation and activation. Activation command may be different on Windows.

2. `pip install -r requirements.txt`

3. `python -m spacy download en_core_web_md` to download spacy data.

4. For NLTK data, you'll need to open a python terminal and then use download commands
  - Open a python console with `python` then
  - `import nltk`
  - `nltk.download('vader_lexicon')`
  - `nltk.download('popular')`

5. Finally, start the chatbot with `python chatbot.py`. Knowledge base for the chatbot is already saved as `knowledge_base.pkl`.

## Previous Steps

1. `mkdir -p crawled_files && python crawler.py` to crawl text from an initial set of links. The script saves the crawled text in the `crawled_files` directory.

2. `mkdir -p cleaned_files && python cleaner.py` to clean the crawled text. The script saves the cleaned text in the `cleaned_files` directory.

3. `python important_terms.py` to find the important terms. The important terms must be saved as a `IMPORTANT_TERMS` list in `filters.py`. For our usecase we manually filtered the `IMPORTANT_TERMS` list into `PEOPLE` and `OTHERS`.

4. `python knowledge_base.py` to create the knowledge base from the `IMPORTANT_TERMS` list for the chatbot. The knowledge base dictionary is saved as `knowledge_base.txt` and `knowledge_base.pkl`.
