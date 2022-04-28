import mlflow
from sklearn import *

mlflow.sklearn.autolog()

def train_simple(random_state=1121218):
    """
    A function to train simple models on the metadata.
    """
    (x_train, y_train), (x_test, y_test) = get_metadata(random_state=random_state)

    model = RandomForestRegressor(
        n_estimators=1500,
        random_state=random_state,
        max_depth=5,
        min_samples_split=3,
        max_features="sqrt",
    )

    with mlflow.start_run():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rmse_test = mean_squared_error(y_test, y_pred, squared=False)

    mlflow.end_run()

train_simple()