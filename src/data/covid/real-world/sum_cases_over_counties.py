import pandas as pd

# import dataframe and sum up 'newly infected'
df = pd.read_csv("src/data/covid/real-world/raw/NEW_cases_rki.csv")
df = df.groupby("day_idx", as_index=False).agg({"newly_infected": "sum"})
df = df.reset_index(drop=True)
df.columns = ["Day", "Infected"]

# export dataframe as csv
df.to_csv("src/data/covid/real-world/raw/summed_cases_over_counties.csv", index=False)