from sklearn.preprocessing import StandardScaler


def normalize_data(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    return (X_train, X_test)

