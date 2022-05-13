import pandas as pd

urls = pd.read_csv("recipe_urls.csv")
data = pd.read_csv("cleaned_recipes.csv")

combined = pd.concat([urls, data], axis=1)


combined.to_csv("combined_recipes.csv", index = False)
