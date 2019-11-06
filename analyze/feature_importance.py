import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


def analze_feature_importance(x_train, y_train):
    rf = RandomForestRegressor(n_estimators=100,
                               n_jobs=-1,
                               oob_score=True,
                               bootstrap=True,
                               random_state=42)
    rf.fit(x_train, y_train)

    print(rf.get_support())
