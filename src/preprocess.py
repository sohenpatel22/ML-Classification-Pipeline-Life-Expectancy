from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(X_train, X_test):
    # Impute missing values
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled, imputer, scaler