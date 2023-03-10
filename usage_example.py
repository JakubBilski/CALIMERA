import numpy as np

from calimera import CALIMERA
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.metrics import accuracy_score


def load_example_data():
    # example data from https://timeseriesclassification.com/
    X_train, y_train = load_from_tsfile_to_dataframe(
        'Univariate_ts/DodgerLoopDay/DodgerLoopDay_TRAIN.ts'
    )
    X_train = np.asarray([x[0] for x in X_train.to_numpy()])
    X_test, y_test = load_from_tsfile_to_dataframe(
        'Univariate_ts/DodgerLoopDay/DodgerLoopDay_TRAIN.ts'
    )
    X_test = np.asarray([x[0] for x in X_test.to_numpy()])
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_example_data()

    delay_penalty = 1.0
    model = CALIMERA(delay_penalty=delay_penalty)
    model.fit(X_train, y_train)

    stop_timestamps, y_pred = model.test(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    earliness = sum(stop_timestamps) / (X_train.shape[1] * X_train.shape[0])
    cost = 1.0 - accuracy + delay_penalty * earliness
    print(f'Accuracy: {accuracy}\nEarliness: {earliness}\nCost: {cost}')
