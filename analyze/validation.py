from analyze.load_data import TitanicDataLoad
from analyze.feature_importance import analze_feature_importance
from model.first_model import create_model_and_train
from model.first_model import labelEncoder
from sklearn.utils import shuffle
import numpy as np
from sklearn import metrics


def validate_model():
    data = TitanicDataLoad()
    data.loadFile("/Users/nicolaepetridean/jde/projects/titanic/try/data/")

    train_data_X = labelEncoder(data.train_data_X, 1)

    X, y = shuffle(train_data_X, data.train_data_Y, random_state=0)

    X_train = X[:-150, :]
    Y_train = y[:-150:, :]

    X_dev = X[X_train.shape[0]:, :]
    Y_dev = y[Y_train.shape[0]:, :]

    se = create_model_and_train(X_train, Y_train, X_dev)

    se_array = se.to_numpy()

    se_prediction = np.array([1 if value[0] >= 0.5 else 0 for value in se_array])

    compare = np.concatenate((se_prediction.reshape(150, 1), Y_dev), axis=1)

    concat_with_x = np.concatenate((X_dev, compare), axis=1)

    prediction_mistakes = [concat_with_x[i] for i, row in enumerate(concat_with_x) if concat_with_x[i][6] != concat_with_x[i][7]]

    print(metrics.accuracy_score(compare[:, 1].astype(int), compare[:, 0].astype(int)))


if __name__ == "__main__":
    validate_model()
