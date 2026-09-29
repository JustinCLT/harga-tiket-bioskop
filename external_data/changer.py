import pandas as pd
df = pd.read_csv("missing_movies_metadata.csv")

df.loc['13+'] = 'Remaja'
df.to_csv("AllDetails.csv", index=False)
print(df)