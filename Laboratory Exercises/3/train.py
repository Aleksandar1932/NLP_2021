import subprocess

import os
import ast
import sys
from uuid import uuid4
subprocess.call(['open', lines_kml_flyingpath]);


import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import wandb
from wandb.keras import WandbCallback

from scripts.model import create_model
from scripts.loader import load_embeddings_pkl

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME") or "[NLP] lab-03 | text-summarization"
LEN_VOCABULARY = os.getenv("LEN_VOCABULARY") or 7188
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH") or 'data/embedding_matrix_glove_50.pkl'

embeddings = load_embeddings_pkl(EMBEDDINGS_PATH)
df = pd.read_csv('data/opinosis-training.csv')

padded_texts = np.array([ast.literal_eval(sample) for sample in df.texts.to_list()])
padded_summaries = np.array([ast.literal_eval(sample) for sample in df.summaries.to_list()])
next_words = np.array([ast.literal_eval(sample) for sample in df.next_words.to_list()])

max_texts_length = len(padded_texts[0])
max_summaries_length = len(padded_summaries[0])

with wandb.init(project=WANDB_PROJECT_NAME):
    config = wandb.config
    model = create_model(max_texts_length, max_summaries_length, LEN_VOCABULARY, 50, embeddings)
    model.compile(optimizer=Adam(lr=config.learning_rate), loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    model.fit([np.array(padded_texts), np.array(padded_summaries)],
            np.array(next_words),
            validation_split=0.1,
            batch_size=64, epochs=15, verbose=1,
            callbacks=[WandbCallback()]
            )