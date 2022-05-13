import pandas as pd
from bs4 import BeautifulSoup 
import numpy as np

data = pd.read_csv("collected_recipes.csv")

for i in range(len(data["Instructions"])):
	soup = BeautifulSoup(data["Instructions"][i], features = "html.parser")
	title = BeautifulSoup(data["Title"][i], features = "html.parser")
	ingr = BeautifulSoup(data["Ingredients"][i], features = "html.parser")
	# print(soup.get_text())
	data["Instructions"][i] = soup.get_text()
	data["Ingredients"][i] = ingr.get_text()
	data["Title"][i] = title.get_text()

# soup = BeautifulSoup(data["Instructions"])

# print(soup.get_text())
# print(data["Instructions"])

data.to_csv("cleaned_recipes.csv", index = False)
