import os
import sys
import pickle
import logging
from logging import getLogger

import numpy as np


def load_embeddings_pkl(path):
    with open(path, 'rb') as f:
        embedding_matrix = pickle.load(f)
    return embedding_matrix


def load_embeddings(vocabulary, embedding_size=50, embedding_type='glove', embeddings_path='/mnt/d/Downloads', dump_path='./data'):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = getLogger('pretrained_embeddings')

    if os.path.exists(f'{dump_path}/embedding_matrix_{embedding_type}_{embedding_size}.pkl'):
        logger.info('Loading embedding matrix from file')
        with open(f'{dump_path}/embedding_matrix_{embedding_type}_{embedding_size}.pkl', 'rb') as f:
            embedding_matrix = pickle.load(f)

    else:
        if embedding_type == 'glove':
            embeddings_index = {}
            f = open(f'{embeddings_path}/glove.6B.{embedding_size}d.txt')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

            logger.info(f'Found {len(embeddings_index)} word vectors.')

            embedding_matrix = np.zeros((len(vocabulary), embedding_size))
            logger.info('Creating embedding matrix')
            for i in range(len(vocabulary)):
                if vocabulary[i] in embeddings_index.keys():
                    embedding_matrix[i] = embeddings_index[vocabulary[i]]
                else:
                    embedding_matrix[i] = np.random.standard_normal(
                        embedding_size)
            with open(f'{dump_path}/embedding_matrix_{embedding_type}_{embedding_size}.pkl', 'wb') as f:
                pickle.dump(embedding_matrix, f)
        else:
            logger.error(
                f'No pretrained embeddings found for {embedding_type}')

    return embedding_matrix
