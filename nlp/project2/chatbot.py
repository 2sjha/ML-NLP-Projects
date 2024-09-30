"""
Script to run a terminal chatbot using the knowledge base
"""

import os

# Suppress Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL

import math
import random
import spacy
import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from string import punctuation
from nltk import word_tokenize, pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from spacy.language import Language
from utils import PEOPLE, OTHERS, save_dict_as_txt, load_pickle, load_qa_dataset


# Prompts for the user to request for some question
ASK_PROMPTS = [
    ("Go ahead {}, Ask me something.", 1),
    ("Go ahead {}, Ask me something else.", 1),
    ("Employees of Dunder Mifflin, LINE UP! {} has a question.", 1),
    ("Employees of Dunder Mifflin, LINE UP! {} has another question.", 1),
    (
        'I can tell you more about "The Office". Ask me about its characters or places.',
        0,
    ),
    (
        "I know all the secrets of the employees of Dunder Mifflin. Ask me about someone.",
        0,
    ),
]


def yes_or_no(user_input: str) -> int:
    """
    Checks if the user input contains Yes or No.
    Returns +1 if it contains Yes, -1 if it contains No, 0 if neither
    """
    yes_words = ["yes", "yep", "yeah", "yesh", "yeppers", "sure", "okay", "aye", "y"]
    no_words = ["no", "nope", "nah", "nay", "n"]

    for yes in yes_words:
        if yes in user_input:
            return 1
    for no in no_words:
        if no in user_input:
            return -1

    return 0


def print_initial_info_prompt():
    """
    Prints initial info about the show
    """
    # Randomly select some characters and other things to prompt the user
    random_people = random.sample(PEOPLE, 3)
    random_other = random.sample(OTHERS, 2)
    print(
        "I can talk about characters from the show like",
        str(random_people)[1:-1] + ".",
        "I also have some info about other things from the show like",
        str(random_other)[1:-1] + ".\n",
    )


def get_name(user_input: str, spacy_nlp: Language) -> str:
    """
    Returns a Person's Name by using NER or manual methods
    """
    # If the user_input just one word, then most likely that is their name
    if len(user_input.split(" ")) == 1:
        return user_input

    # Try NER next (works well with American/English Names)
    spacy_ner = spacy_nlp(user_input)
    ents = spacy_ner.ents
    if len(ents) > 0:
        user_name = ents[0].text
        return user_name

    # Then try POS tagging and use the first proper noun
    words = word_tokenize(user_input)
    pos_tags = pos_tag(words)

    proper_nouns = [word for (word, pos) in pos_tags if pos in ["NNP", "NNPS"]]

    if not proper_nouns:
        print("I couldn't get your name")
        user_name = "Anonymous"
    else:
        user_name = proper_nouns[0]

    return user_name


def get_likes_dislikes(
    user_input: str, spacy_nlp: Language, sia: SentimentIntensityAnalyzer
) -> tuple[Set[str], Set[str]]:
    """
    Uses Sentiment Analysis to identify if the user likes or dislikes something from the user_input
    """
    likes_dislikes = set()

    # Do Sentiment Analysis to figure out if the input tells us anything about they like or dislike
    scores = sia.polarity_scores(user_input)
    if scores["compound"] > 0:
        sentiment = "positive"
    elif scores["compound"] < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        return (None, None)

    # Try NER first to identify interesting elements.
    spacy_ner = spacy_nlp(user_input)
    ents = spacy_ner.ents
    if len(ents) > 0:
        for ent in ents:
            likes_dislikes.add(ent.text)
    else:
        # Then try POS tagging and use nouns
        words = word_tokenize(user_input)
        pos_tags = pos_tag(words)

        nouns = [
            word for (word, pos) in pos_tags if pos in ["NNP", "NNPS", "NN", "NNS"]
        ]
        likes_dislikes = set(nouns)

    if likes_dislikes and sentiment == "positive":
        return (likes_dislikes, None)
    elif likes_dislikes and sentiment == "negative":
        return (None, likes_dislikes)
    else:
        return (None, None)


def create_user_model(user_name: str) -> Dict[str, Any]:
    """
    Creates a default user model with likes and dislikes
    """
    user_model = {"name": [user_name], "likes": set(), "dislikes": set()}
    save_dict_as_txt(user_model, "um_" + user_name.lower() + ".txt")
    return user_model


def update_user_model(
    user_model: Dict[str, Any], likes: Set[str], dislikes: Set[str]
) -> Dict[str, Any]:
    """
    Updates the user model with new likes or dislikes information
    """
    if not user_model or not user_model["name"]:
        return

    user_name = user_model["name"][0]

    if likes:
        user_model["likes"].update(likes)

    if dislikes:
        user_model["dislikes"].update(dislikes)

    if likes or dislikes:
        save_dict_as_txt(user_model, "um_" + user_name.lower() + ".txt")

    return user_model


def cosine_similarity(user_input: str, info_item: str, stop_words: Set[str]) -> float:
    """
    Calculates cosine similarity between user_input and an info_item strings
    """
    user_tokens = word_tokenize(user_input)
    info_tokens = word_tokenize(info_item)

    vocab = {}
    for word in user_tokens:
        if word not in stop_words and word not in punctuation and len(word) > 3:
            vocab[word.lower()] = 0
    for word in info_tokens:
        if word not in stop_words and word not in punctuation and len(word) > 3:
            vocab[word.lower()] = 0

    vocab = dict(sorted(vocab.items(), key=lambda x: x[0]))
    input_dict = dict(vocab)
    for word in user_tokens:
        if word in vocab:
            input_dict[word.lower()] += 1

    info_dict = dict(vocab)
    for word in info_tokens:
        if word in vocab:
            info_dict[word.lower()] += 1

    input_vec = list(input_dict.values())
    info_vec = list(info_dict.values())

    dot = 0.0
    input_norm = 0.0
    info_norm = 0.0
    for idx, _ in enumerate(input_vec):
        dot += input_vec[idx] * info_vec[idx]

        input_norm += input_vec[idx] * input_vec[idx]
        info_norm += info_vec[idx] * info_vec[idx]

    input_norm = math.sqrt(input_norm)
    info_norm = math.sqrt(info_norm)

    if not input_norm or not info_norm:
        return 0.0
    else:
        cos_sim = dot / (input_norm * info_norm)
        return cos_sim


def add_user_model_to_userinput(user_input: str, user_model: Dict[str, Any]) -> str:
    """
    Concatenates Likes and Dislikes to the user input, so that it can be used for response generation
    """
    res = ""
    if user_model["likes"]:
        res += "likes " + " ".join(user_model["likes"]) + " "
    if user_model["dislikes"]:
        res += "dislikes " + " ".join(user_model["dislikes"]) + " "
    return res + user_input


def make_inference_models(
    encoder_inputs: tf.keras.layers.Input,
    encoder_states: List[tf.keras.layers.Layer],
    decoder_embedding: tf.keras.layers.Layer,
    decoder_lstm: tf.keras.layers.Layer,
    decoder_dense: tf.keras.layers.Layer,
    decoder_inputs: tf.keras.layers.Input,
) -> Tuple[tf.keras.models.Model, tf.keras.models.Model]:
    """
    Creates inference models for encoder and decoder.
    """
    encoder_model = tf.keras.models.Model(
        encoder_inputs, encoder_states, name="encoder_model"
    )

    decoder_state_input_h = tf.keras.layers.Input(
        shape=(200,), name="decoder_state_input_h"
    )
    decoder_state_input_c = tf.keras.layers.Input(
        shape=(200,), name="decoder_state_input_c"
    )

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states,
        name="decoder_model",
    )

    return (encoder_model, decoder_model)


def extract_components_from_model(
    model: tf.keras.models.Model,
) -> Tuple[
    tf.keras.layers.Input,
    List[tf.keras.layers.Layer],
    tf.keras.layers.Layer,
    tf.keras.layers.Layer,
    tf.keras.layers.Layer,
    tf.keras.layers.Input,
]:
    """
    Extracts components (layers) from the trained model.
    """
    encoder_inputs = model.input[0]
    _, state_h, state_c = model.get_layer("encoder_lstm").output
    encoder_states = [state_h, state_c]

    decoder_inputs = model.input[1]
    decoder_embedding = model.get_layer("decoder_embedding").output
    decoder_lstm = model.get_layer("decoder_lstm")
    decoder_dense = model.get_layer("decoder_dense")

    return (
        encoder_inputs,
        encoder_states,
        decoder_embedding,
        decoder_lstm,
        decoder_dense,
        decoder_inputs,
    )


def str_to_tokens(
    sentence: str,
    tokenizer: tf.keras.preprocessing.text.Tokenizer,
    maxlen_questions: int,
):
    """
    Convert a sentence to tokens using the given tokenizer.
    """
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(tokenizer.word_index[word])
    return tf.keras.preprocessing.sequence.pad_sequences(
        [tokens_list], maxlen=maxlen_questions, padding="post"
    )


def get_kb_response(
    user_input: str,
    knowledge_base: Dict[str, List[str]],
    stop_words: Set[str],
    ml_response_found: bool,
) -> str:
    """
    Creates a response for the user_input from the knowledge base
    """

    response = ""
    kb_term = ""
    potential_responses = []
    for term in knowledge_base.keys():
        # Check which term is contained within the user_input
        parts = term.lower().split(" ")
        for part in parts:
            if part in user_input.lower():
                kb_term = term

    # Some KB term found for the user input, then
    # find similar sentences for the user input using cosine similarity
    if kb_term:
        for info in knowledge_base[kb_term]:
            # Add this info item as a potential response info
            # only if cosine similarity greater than threshold
            if cosine_similarity(user_input, info, stop_words) > 0.1:
                potential_responses.append(info)

    # Choose a random response from a list of potential responses
    if potential_responses or ml_response_found:
        if potential_responses:
            response += random.sample(potential_responses, 1)[0]
        response += "\nIf you're not satisfied with my answer,"
    else:
        response = "\nI couldnt find an answer,"

    # Add Optional Google Search for the response
    response += " here's a google search link for your question: "
    response += "https://www.google.com/search?q="

    for word in user_input.split(" "):
        response += word + "+"
    response = response[:-1] + ".\n"

    return response


def get_ml_response(
    user_input: str,
    enc_model: tf.keras.models.Model,
    dec_model: tf.keras.models.Model,
    tokenizer: tf.keras.preprocessing.text.Tokenizer,
    maxlen_questions: int,
    maxlen_answers: int,
) -> str:
    """
    Returns a response generated by the NN encoder, decoder model.
    """
    states_values = enc_model.predict(
        str_to_tokens(user_input, tokenizer, maxlen_questions)
    )
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index["start"]
    stop_condition = False
    decoded_response = ""
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict(
            [empty_target_seq] + states_values, verbose=0
        )
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_response += " {}".format(word)
                sampled_word = word

        if sampled_word == "end" or len(decoded_response.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    return " ".join(decoded_response.split()[:-1])


def run_chatbot(knowledge_base: Dict[str, List[str]]):
    """
    Driver to run the chatbot
    """
    # Load Spacy NLP for NER tasks
    spacy_nlp = spacy.load("en_core_web_md")
    sia = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words("english"))

    # Load Tokenizer for ML responses
    qna_df = load_qa_dataset()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(qna_df["question"] + qna_df["answer"])

    # encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(qna_df["question"])
    maxlen_questions = max([len(x) for x in tokenized_questions])

    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences(qna_df["answer"])
    maxlen_answers = max([len(x) for x in tokenized_answers])

    # Load the trained model
    model = tf.keras.models.load_model("model.keras")

    # Extract components from the model
    (
        encoder_inputs,
        encoder_states,
        decoder_embedding,
        decoder_lstm,
        decoder_dense,
        decoder_inputs,
    ) = extract_components_from_model(model)

    enc_model, dec_model = make_inference_models(
        encoder_inputs,
        encoder_states,
        decoder_embedding,
        decoder_lstm,
        decoder_dense,
        decoder_inputs,
    )

    # Print Introduction
    print(
        'Hello, I\'m DMI-bot, a chatbot about the TV show "The Office".',
        'You can exit the chat with "Exit" or "exit" or "EXIT".\nWhat\'s your name?',
    )

    # Request User name for the user model
    user_input = input("User: ")
    if user_input.lower() == "exit":
        exit()

    # Create and save a default user model from user's name
    user_name = get_name(user_input, spacy_nlp)
    user_model = create_user_model(user_name)

    # Ask if the user has watched the show, if yes add it to their likes
    print("\nHello", user_name + "!", 'Have you watched the show "The Office (US)"?')
    user_input = input(user_name + ": ")
    if user_input.lower() == "exit":
        exit()

    # If couldn't detect that user has watched the show,
    # then print some info explaining about the show
    yn = yes_or_no(user_input.lower())
    if yn == -1:
        print(
            "\nThe Office is an American mockumentary sitcom television series",
            "that depicts the everyday work lives of office employees at the",
            "Scranton, Pennsylvania, branch of the fictional Dunder Mifflin Paper Company.",
        )
        print_initial_info_prompt()
    elif yn == 1:
        update_user_model(user_model, likes={"The Office"}, dislikes=None)
        print(
            "\nNice. I can talk about the employees of Dunder Mifflin",
            "and some other things from the show.",
        )
    else:
        print("\nI couldn't understand your response.")
        print_initial_info_prompt()

    # Main Loop for the Chatbot
    while True:
        # Randomly select prompts for the user
        random_prompt = random.sample(ASK_PROMPTS, 1)[0]
        if random_prompt[1]:
            print(random_prompt[0].format(user_name))
        else:
            print(random_prompt[0])

        # Request User input
        user_input = input(user_name + ": ")

        # Exit if User inputs Exit
        if user_input.lower() == "exit":
            exit()

        # Check if we can figure out likes, dislikes from the input
        # and update the user model if something found
        likes, dislikes = get_likes_dislikes(user_input, spacy_nlp, sia)
        update_user_model(user_model, likes, dislikes)

        try:
            user_input_with_likes_dislikes = add_user_model_to_userinput(
                user_input, user_model
            )
            ml_response = get_ml_response(
                user_input_with_likes_dislikes,
                enc_model,
                dec_model,
                tokenizer,
                maxlen_questions,
                maxlen_answers,
            )
        except Exception as _:
            ml_response = ""
        kb_response = get_kb_response(
            user_input, knowledge_base, stop_words, ml_response != ""
        )

        # Print the response
        print("\n" + ml_response + kb_response)


if __name__ == "__main__":
    kb = load_pickle("knowledge_base.pkl")
    run_chatbot(kb)
