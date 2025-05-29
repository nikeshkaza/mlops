import pandas as pd
from sklearn.model_selection import train_test_split

path="/Workspace/Users/nikesh.kaza@accenture.com/mlops/data"
df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
df.to_csv(path+"/boston.csv", index=False)  # Save to DBFS

# Split data
train, test = train_test_split(df, test_size=0.2, random_state=42)
train.to_csv(path+"/train.csv", index=False)
test.to_csv(path+"/test.csv", index=False)
