import pandas as pd

data = pd.DataFrame(columns=("X", "Y", "click"));
data.to_csv("DataFile_Name.csv", index=False);
