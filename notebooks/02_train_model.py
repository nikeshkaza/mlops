import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
path="/Workspace/Users/nikesh.kaza@accenture.com/mlops/data"
train = pd.read_csv(path+"/train.csv")
X = train.drop("medv", axis=1)
y = train["medv"]

mlflow.start_run(nested=True)
model = LinearRegression().fit(X, y)
mlflow.sklearn.log_model(model, "model")
mlflow.log_metric("rmse", mean_squared_error(y, model.predict(X), squared=False))
mlflow.end_run()
