from analyze.load_data import TitanicDataLoad
from analyze.feature_importance import analze_feature_importance
from model.first_model import create_model_and_train
from model.first_model import labelEncoder

if __name__ == "__main__":
    data = TitanicDataLoad()
    data.loadFile("/Users/nicolaepetridean/jde/projects/titanic/try/data/")

    #analze_feature_importance(data.train_data[:,-2], data.train_data[-2])

    train_data_X = labelEncoder(data.train_data_X, 1)

    test_data_X = labelEncoder(data.test_data_X, 1)

    #train_data_X = labelEncoder(train_data_X, 6)

    create_model_and_train(train_data_X, data.train_data_Y, test_data_X)


