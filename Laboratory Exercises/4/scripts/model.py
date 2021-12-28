from transformers import TFBertForSequenceClassification, TFAlbertForSequenceClassification, TFRobertaForSequenceClassification, TFDistilBertForSequenceClassification
from transformers import BertTokenizerFast, AlbertTokenizerFast, RobertaTokenizerFast, DistilBertTokenizerFast
from tensorflow.keras import backend as K

from scripts.utils import tokenize

models_pretrained = {
    TFBertForSequenceClassification.__name__: "bert-base-cased",
    TFAlbertForSequenceClassification.__name__: "albert-base-v2",
    TFRobertaForSequenceClassification.__name__: "roberta-base",
    TFDistilBertForSequenceClassification.__name__: "distilbert-base-cased",
}

models_tokenizers = {
    TFBertForSequenceClassification.__name__: BertTokenizerFast,
    TFAlbertForSequenceClassification.__name__: AlbertTokenizerFast,
    TFRobertaForSequenceClassification.__name__: RobertaTokenizerFast,
    TFDistilBertForSequenceClassification.__name__: DistilBertTokenizerFast,
}


def get_model_and_data(model_cls, num_labels, X_train, X_test):
    """
    Returns a model and data for the given model class.
    :param model_cls: Model class
    :param num_labels: Number of labels
    :param X_train: Training data
    :param X_test: Testing data
    :return: Model, training data, testing data    
    """
    print(
        f"Creating {model_cls.__name__}-{models_pretrained[model_cls.__name__]} with {num_labels} labels")
    print(
        f"Tokenizing data with {models_tokenizers[model_cls.__name__].__name__}")

    tokenizer = models_tokenizers[model_cls.__name__].from_pretrained(
        models_pretrained[model_cls.__name__])
    train_input_ids, train_attention_masks = tokenize(X_train, tokenizer)
    test_input_ids, test_attention_masks = tokenize(X_test, tokenizer)

    return model_cls.from_pretrained(models_pretrained[model_cls.__name__], num_labels=num_labels), \
        train_input_ids, train_attention_masks, \
        test_input_ids, test_attention_masks,


# Custom metrics to calculate recall, precision and f1 score

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
