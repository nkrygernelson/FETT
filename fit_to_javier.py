import pandas as pd
import os 
data = pd.read_json(os.path.join("data", "home_sol","javier_latest.json" ))
print(data.head(10))
print(data.columns)
print(data["Band Gap (eV)"])
print(data["Energy above Hull (eV/at)"])
