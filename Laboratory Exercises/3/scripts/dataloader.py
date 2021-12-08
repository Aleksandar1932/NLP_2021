import typing
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

TOPICS_PATH = os.getenv("TOPICS_PATH") or "/mnt/d/Downloads/OpinosisDataset1.0_0/topics"
TOPIC_FILE_EXTENSION = os.getenv("TOPIC_FILE_EXTENSION") or "txt.data"
SUMMARIES_PATH = os.getenv("SUMMARIES_PATH") or  "/mnt/d/Downloads/OpinosisDataset1.0_0/summaries-gold"


def _discover_topic_names() -> typing.List[str]:
    # discover the topic names
    topic_names = []
    for (_, _, filenames) in os.walk(TOPICS_PATH):
        topic_names.extend(filenames)
        break

    topic_names = [filename.replace(
        f".{TOPIC_FILE_EXTENSION}", "") for filename in topic_names]
    return topic_names


def _read_full_texts(topic_names: typing.List[str]) -> typing.Dict[str, str]:
    full_texts = {}
    for topic in topic_names:
        with open(os.path.join(TOPICS_PATH, f"{topic}.{TOPIC_FILE_EXTENSION}"), "rb") as f:
            content = f.read()
            full_texts[topic] = content.decode("unicode_escape")
    return full_texts


def _read_summaries(topic_names: typing.List[str]) -> typing.Dict[str, typing.List[str]]:
    summaries = {}
    for topic in topic_names:
        summaries[topic] = []
        for (dirpath, _, filenames) in os.walk(os.path.join(SUMMARIES_PATH, topic)):
            for filename in filenames:
                with open(os.path.join(dirpath, filename), "rb") as f:
                    content = f.read()
                    summaries[topic].append(content.decode("unicode_escape"))
    return summaries


def _make_df(topic_names: typing.List[str], full_texts: typing.Dict[str, str], summaries: typing.Dict[str, typing.List[str]]) -> pd.DataFrame:
    data = pd.DataFrame([{"id": topic, "text": full_texts[topic],
                        "summary": summaries[topic][0]} for topic in topic_names])
    return data


def load_data() -> pd.DataFrame:
    topic_names = _discover_topic_names()
    full_texts = _read_full_texts(topic_names)
    summaries = _read_summaries(topic_names)
    data = _make_df(topic_names, full_texts, summaries)

    return data
