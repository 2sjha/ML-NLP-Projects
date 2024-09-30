"""
Extracts the project1_datasets.zip and then reads all text files
to get the vocabulary and the dataset
"""

import zipfile
from os import listdir
from os.path import isfile, join, exists
from sys import stderr
from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Datasets:
    """
    Contains vocabulary and processed dataset
    """

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def __init__(self, test_train: str):
        self.ham_files = {"enron1": [], "enron2": [], "enron4": []}
        self.spam_files = {"enron1": [], "enron2": [], "enron4": []}

        self.ham_dataset = {"enron1": {}, "enron2": {}, "enron4": {}}
        self.spam_dataset = {"enron1": {}, "enron2": {}, "enron4": {}}

        self.vocabulary = {"enron1": {}, "enron2": {}, "enron4": {}}
        self.vocab_len = {"enron1": 0, "enron2": 0, "enron4": 0}

        if not self.is_dataset_extracted():
            self.extract_enron_datasets()
        self.get_dataset_files(test_train)
        self.create_vocabulary_and_datasets()

    def is_dataset_extracted(self) -> bool:
        """
        Checks if the dataset is already extracted
        """
        return (
            exists("./../project1_datasets")
            and exists("./../project1_datasets/enron1")
            and exists("./../project1_datasets/enron2")
            and exists("./../project1_datasets/enron4")
        )

    def extract_enron_datasets(self):
        """
        Extracts the main zip file and subsequently extracts
        the enron_test and enron_train zip files
        """
        with zipfile.ZipFile("./../project1_datasets.zip", "r") as zip_ref:
            zip_ref.extractall("./..")

        with zipfile.ZipFile("./../project1_datasets/enron1_test.zip", "r") as zip_ref:
            zip_ref.extractall("./../project1_datasets")
        with zipfile.ZipFile("./../project1_datasets/enron1_train.zip", "r") as zip_ref:
            zip_ref.extractall("./../project1_datasets")

        with zipfile.ZipFile("./../project1_datasets/enron2_test.zip", "r") as zip_ref:
            zip_ref.extractall("./../project1_datasets/enron2")
        with zipfile.ZipFile("./../project1_datasets/enron2_train.zip", "r") as zip_ref:
            zip_ref.extractall("./../project1_datasets/enron2")

        with zipfile.ZipFile("./../project1_datasets/enron4_test.zip", "r") as zip_ref:
            zip_ref.extractall("./../project1_datasets")
        with zipfile.ZipFile("./../project1_datasets/enron4_train.zip", "r") as zip_ref:
            zip_ref.extractall("./../project1_datasets")

    def get_dataset_files(self, test_train: str):
        """
        Accumulates all filenames inside the zip into ham and spam
        """
        dtst_names = ["enron1", "enron2", "enron4"]

        for dt_st in dtst_names:
            ham_path = join("./../project1_datasets/", dt_st, test_train, "ham")
            spam_path = join("./../project1_datasets/", dt_st, test_train, "spam")

            for hm_f in listdir(ham_path):
                if isfile(join(ham_path, hm_f)):
                    self.ham_files[dt_st].append(join(dt_st, test_train, "ham", hm_f))

            for spm_f in listdir(spam_path):
                if isfile(join(spam_path, spm_f)):
                    self.spam_files[dt_st].append(
                        join(dt_st, test_train, "spam", spm_f)
                    )

    def get_words_from_email(self, email: str) -> List[str]:
        """
        Retrieves relevant words from an email
        """
        # Tokenize Words
        email_words = word_tokenize(email)

        # Remove stopwords
        filtered_words = [
            word for word in email_words if word.casefold() not in self.stop_words
        ]

        # Reduce words to their lemma / core meaning
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in filtered_words]
        return lemmatized_words

    def create_vocabulary_and_datasets(self):
        """
        Reads all ham/spam text files and creates the datasets & vocabulary for our models
        """
        dtst_names = ["enron1", "enron2", "enron4"]
        for dt_st in dtst_names:
            for ham_file in self.ham_files[dt_st]:
                try:
                    with open(
                        "./../project1_datasets/" + ham_file, "r", encoding="utf-8"
                    ) as ham_email:
                        ham_email_str = ham_email.read()
                        ham_email_words = self.get_words_from_email(ham_email_str)

                        # Add the processed email to dataset
                        self.ham_dataset[dt_st][ham_file] = ham_email_words

                        # Add words to vocabulary only if they're not already present
                        # Unique words vocabulary
                        for word in ham_email_words:
                            if word not in self.vocabulary[dt_st]:
                                self.vocabulary[dt_st][word] = self.vocab_len[dt_st]
                                self.vocab_len[dt_st] += 1
                except UnicodeError as error:
                    print(
                        "Error occured: " + str(error) + " in " + ham_file, file=stderr
                    )

            for spam_file in self.spam_files[dt_st]:
                try:
                    with open(
                        "./../project1_datasets/" + spam_file, "r", encoding="utf-8"
                    ) as spam_email:
                        spam_email_str = spam_email.read()
                        spam_email_words = self.get_words_from_email(spam_email_str)

                        # Add the processed email to dataset
                        self.spam_dataset[dt_st][spam_file] = spam_email_words

                        # Add words to vocabulary only if they're not already present
                        # Unique words vocabulary
                        for word in spam_email_words:
                            if word not in self.vocabulary[dt_st]:
                                self.vocabulary[dt_st][word] = self.vocab_len[dt_st]
                                self.vocab_len[dt_st] += 1
                except UnicodeError as error:
                    print(
                        "Error occured: " + str(error) + " in " + spam_file, file=stderr
                    )
