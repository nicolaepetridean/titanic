from analyze.load_data import TitanicDataLoad
from analyze.feature_importance import analze_feature_importance

if __name__ == "__main__":
    data = TitanicDataLoad()
    data.loadFile("/Users/nicolaepetridean/jde/projects/titanic/try/data/")

    analze_feature_importance(data.train_data[:,-2], data.train_data[-2])


