#oggpnosn
#hkhr

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
import tensorflow_probability as tfp

def vectorize_dataset(dataset_filepath):
    """Data pre-processing: Convert raw data into feature vector"""
    target_variable = "Survived"
    categorical_variables = ["Pclass",
                             "Sex",
                             "Embarked"]
    ratio_variables = ["Age",
                       "SibSp",
                       "Parch",
                       "Fare"]


    dataset = pd.read_csv(dataset_filepath)
    dataset = dataset[categorical_variables+ratio_variables+[target_variable]]

    # dropping all the rows with any NA element.
    dataset = dataset.dropna(axis=0)

    # Convert all the variables to numeric values.
    categorical_dataset = pd.get_dummies(dataset[categorical_variables],
                                         columns=categorical_variables)
    ratio_dataset = dataset[ratio_variables]
    vectorized_dataset = pd.concat([categorical_dataset, ratio_dataset], axis=1)


    X = np.array(vectorized_dataset)
    Y = np.array(dataset[target_variable])
    return (X,Y)


# Learn a model
def get_homoscedastic_model():
    # Model specification.
    model = tf.keras.Sequential()
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def get_hetroscedastic_model():
    # Model specification.
    model = tf.keras.Sequential()
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(2))
    # Instead of point-estimate, now we output distribution!
    model.add(tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(
                                            loc=tf.sigmoid(t[..., 0:1]), scale=tf.math.softplus(t[..., 1:2]))))


    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='mse',
                  metrics=['mae'])
    return model


def one_hot(a, num_classes=2):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

# Evaluate model
def evaluate_model(model_fn, features, labels, no_of_folds=5):
    splitter = StratifiedKFold(n_splits=no_of_folds)
    for train_index, test_index in splitter.split(features, labels):
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        model = model_fn()
        model.fit(features_train, labels_train, epochs=20, batch_size=32)

        # print(model.predict(features_test))
        print(model.evaluate(features_test, labels_test))


# Deploy the model as an api.
if __name__ == "__main__":
    X, Y = vectorize_dataset("./Data/train.csv")

    print(evaluate_model(get_hetroscedastic_model, X, Y))
