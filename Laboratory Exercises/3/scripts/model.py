from uuid import uuid4

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def create_model(texts_size, summaries_size, vocabulary_size, embedding_size, embeddings=None, name=None) -> Model:
    name = f"Encoder-Decoder-{str(uuid4())}" if name is None else name

    # Encoder
    encoder_inputs = Input(shape=(texts_size,), name="encoder_inputs")
    encoder_embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size,
                                  weights=[embeddings],
                                  trainable=False)(encoder_inputs)

    encoder = LSTM(128, return_state=True, name="encoder")
    encoder(encoder_embedding)

    _, state_h, state_c = encoder(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(summaries_size,), name="decoder_inputs")
    decoder_embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size,
                                  weights=[embeddings],
                                  trainable=False)(decoder_inputs)

    decoder = LSTM(128, return_state=True, name="decoder")
    decoder_outputs, _, _ = decoder(
        decoder_embedding, initial_state=encoder_states)

    decoder_outputs = Dense(
        vocabulary_size, activation='softmax', name='decoder_dense')(decoder_outputs)

    # Compile the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model._name = name
    return model


def decode(model, input_sent, word_to_id, id_to_word, padding_size, verbose=False):
    generated_sentence = []
    generated_sentence.append(word_to_id['<START>'])

    for i in range(padding_size):
        output_sent = pad_sequences([generated_sentence], padding_size-1)
        predictions = model.predict(
            [np.expand_dims(input_sent, axis=0), output_sent])
        next_word = np.argmax(predictions)
        if verbose:
            print(id_to_word[next_word])
        generated_sentence.append(next_word)

    generated_sentence.append(word_to_id['</END>'])
    return generated_sentence


def convert(sentences, id_to_word):
    output_sentences = []
    for sent in sentences:
        output_sentences.append(' '.join([id_to_word[i] for i in sent]))

    return output_sentences


def calculate_bleu(refences, hypothesis, verbose=False):
    scores = []
    for pred, gt in zip(hypothesis, refences):
        score = sentence_bleu(gt, pred)
        scores.append(score)

        if verbose:
            print(f"Gold Truth: {gt}")
            print(f"Reference Sentence: {pred}")
            print('BLEU score = {}'.format(score))

    return np.mean(scores)


def calculate_rouge(refences, hypothesis, verbose=False):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = []

    for pred, gt in zip(hypothesis, refences):
        res = scorer.score(pred, gt)
        scores.append(res)

        if verbose:
            print(f"Gold Truth: {gt}")
            print(f"Reference Sentence: {pred}")
            print('Rouge score = {}'.format(res))

    return scores
