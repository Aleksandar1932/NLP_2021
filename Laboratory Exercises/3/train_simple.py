import random
import os

import subprocess
import wandb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME") or "[NLP] lab-03 | text-summarization"


def generate_data(num_samples = 100,x_size=10):
    x = [[random.random() for _ in range(x_size)] for _ in range(num_samples)]
    y = [random.random() for _ in range(num_samples)]
    return x, y

X, y = generate_data()


def create_model(x_size, y_size):
    model = Sequential()
    model.add(Dense(10, input_dim=x_size, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    return model

wandb.init(project=WANDB_PROJECT_NAME)
config = wandb.config

model = create_model(10, 1)
model.compile(optimizer=Adam(lr=config.learning_rate), loss='mse', metrics=['mse', 'mae'])
model.fit(X, y, epochs=100, batch_size=10)

wandb.log({"loss": model.history.history['loss'][-1]})