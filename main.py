#oggpnosn
#hkhr

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
import tensorflow_probability as tfp

# Enable eager execution.
tf.enable_eager_execution()

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
    # model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='mse',
                  metrics=['accuracy'])
    return model

# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
          tfp.distributions.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])

# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
          tfp.distributions.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])

def get_aleatoric_model():
    # Model specification.
    model = tf.keras.Sequential()
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(2))
    # Instead of point-estimate, now we output distribution!
    model.add(tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(
                                            loc=tf.sigmoid(t[..., 0:1]), scale=1e-6 + tf.math.softplus(0.05*t[..., 1:2]))))

    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss=negloglik,
                  metrics=['mae'])
    return model


def get_epistemic_model():
    # Model specification.
    model = tf.keras.Sequential()
    model.add(tfp.layers.DenseVariational(32, posterior_mean_field, prior_trainable, activation="relu"))
    model.add(layers.Dropout(.2))
    model.add(tfp.layers.DenseVariational(16, posterior_mean_field, prior_trainable, activation="relu"))
    model.add(tfp.layers.DenseVariational(2, posterior_mean_field, prior_trainable))
    # Instead of point-estimate, now we output distribution!
    model.add(tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Normal(
                                            loc=tf.sigmoid(t[..., 0:1]), scale=tf.math.softplus(t[..., 1:2]))))

    # negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss="mse",
                  metrics=['mae'])
    return model

def one_hot(a, num_classes=2):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

# Evaluate model
def evaluate_homoscedastic_model(model_fn, features, labels, no_of_folds=3):
    splitter = StratifiedKFold(n_splits=no_of_folds)
    for train_index, test_index in splitter.split(features, labels):
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        model = model_fn()
        model.fit(features_train, labels_train, epochs=30, batch_size=32)

        predictions = model.predict(features_test)

        result = np.concatenate((predictions, np.expand_dims(labels_test, axis=1)), axis=1)
        result = sorted(result, key=lambda x:x[0])
        for i in range(predictions.shape[0]):
            print(result[i][0], "\t", result[i][1])

        for i in range(predictions.shape[0]):
            print((predictions[i][0]>.5) == labels_test[i], labels_test[i], predictions[i])
        print(model.evaluate(features_test, labels_test))

def evaluate_hetroskedastic_model(model_fn, features, labels, no_of_folds=3):
    splitter = StratifiedKFold(n_splits=no_of_folds)
    for train_index, test_index in splitter.split(features, labels):
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        model = model_fn()
        model.fit(features_train, labels_train, epochs=600, batch_size=32,
                  validation_data=(features_test, labels_test))

        # print(features_test)

        predictions = model(features_test)
        predictions_mean = predictions.mean()
        predictions_stdev = predictions.stddev()

        # print(predictions.stddev())
        for i in range(predictions.shape[0]):
            print(labels_test[i], (predictions_mean[i]-2*predictions_stdev[i]).numpy(),
                  predictions_mean[i].numpy(), (predictions_mean[i]+2*predictions_stdev[i]).numpy())

            # print(labels_test[i], predictions_mean[i].numpy(),
            #       predictions_stdev[i].numpy())



# Deploy the model as an api.
if __name__ == "__main__":
    X, Y = vectorize_dataset("./Data/train.csv")

    # print(evaluate_model(get_hetroscedastic_model, X, Y))
    # print(evaluate_homoscedastic_model(get_homoscedastic_model, X, Y))

    evaluate_hetroskedastic_model(get_aleatoric_model, X, Y)
    # evaluate_hetroskedastic_model(get_epistemic_model, X, Y)
    # get_epistemic_model
