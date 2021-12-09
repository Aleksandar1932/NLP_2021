import typing
import pandas as pd
import os
from dotenv import load_dotenv
from nltk import word_tokenize
from nltk.corpus import stopwords
ENGLISH_STOPWORDS = stopwords.words('english')

load_dotenv()

TOPICS_PATH = os.getenv(
    "TOPICS_PATH") or "/mnt/d/Downloads/OpinosisDataset1.0_0/topics"
TOPIC_FILE_EXTENSION = os.getenv("TOPIC_FILE_EXTENSION") or "txt.data"
SUMMARIES_PATH = os.getenv(
    "SUMMARIES_PATH") or "/mnt/d/Downloads/OpinosisDataset1.0_0/summaries-gold"


def __discover_topic_names() -> typing.List[str]:
    # discover the topic names
    topic_names = []
    for (_, _, filenames) in os.walk(TOPICS_PATH):
        topic_names.extend(filenames)
        break

    topic_names = [filename.replace(
        f".{TOPIC_FILE_EXTENSION}", "") for filename in topic_names]
    return topic_names


def __read_full_texts(topic_names: typing.List[str]) -> typing.Dict[str, str]:
    full_texts = {}
    for topic in topic_names:
        with open(os.path.join(TOPICS_PATH, f"{topic}.{TOPIC_FILE_EXTENSION}"), "rb") as f:
            content = f.read()
            full_texts[topic] = content.decode("unicode_escape")
    return full_texts


def __read_summaries(topic_names: typing.List[str]) -> typing.Dict[str, typing.List[str]]:
    summaries = {}
    for topic in topic_names:
        summaries[topic] = []
        for (dirpath, _, filenames) in os.walk(os.path.join(SUMMARIES_PATH, topic)):
            for filename in filenames:
                with open(os.path.join(dirpath, filename), "rb") as f:
                    content = f.read()
                    summaries[topic].append(content.decode("unicode_escape"))
    return summaries


def __make_df(topic_names: typing.List[str], full_texts: typing.Dict[str, str], summaries: typing.Dict[str, typing.List[str]]) -> pd.DataFrame:
    data = pd.DataFrame([{"id": topic, "text": full_texts[topic],
                        "summary": summaries[topic][0]} for topic in topic_names])
    return data


def load_data() -> pd.DataFrame:
    topic_names = __discover_topic_names()
    full_texts = __read_full_texts(topic_names)
    summaries = __read_summaries(topic_names)
    data = __make_df(topic_names, full_texts, summaries)

    return data


def __remove_punctuation(token: str) -> str:
    punctuation_regex = '!"#$&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return ' '.join(word.strip(punctuation_regex) for word in token.split())


def nlp_pipeline(token: str) -> typing.List[str]:
    token = __remove_punctuation(token)
    tokens = word_tokenize(token.lower())
    tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]
    return tokens


def create_vocabulary(sentence_tokens):
    vocab = set()
    for tokens in sentence_tokens:
        vocab.update(tokens)

    vocab = list(vocab)
    word_to_id = {word: index for word, index in zip(vocab, range(len(vocab)))}
    id_to_word = {index: word for word, index in zip(vocab, range(len(vocab)))}
    return vocab, word_to_id, id_to_word
