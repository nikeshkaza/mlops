import mlflow.sklearn
import pandas as pd

path="/Workspace/Users/nikesh.kaza@accenture.com/mlops/data"
model = mlflow.sklearn.load_model("runs:/45ae62c532e948c6b175c63080c3543f/model")
test = pd.read_csv(path+"/test.csv")
X_test = test.drop("medv", axis=1)

predictions = model.predict(X_test)
print(predictions[:5])
