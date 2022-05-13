import pandas as pd
from bs4 import BeautifulSoup 
import requests
import time 
import numpy as np
import json

url = "https://www.jamieoliver.com/recipes/category/course/mains/"
page = requests.get(url)

recipe_url_df = pd.DataFrame() 

soup = BeautifulSoup(page.text, "html.parser")

recipe_urls = pd.Series([a.get("href") for a in soup.find_all("a")])

recipe_urls = recipe_urls[(recipe_urls.str.count("-")>0) 
                         & (recipe_urls.str.contains("/recipes/")==True)
                         & (recipe_urls.str.contains("-recipes/")==True)
                         & (recipe_urls.str.contains("course")==False)
                         & (recipe_urls.str.contains("books")==False)
                         & (recipe_urls.str.endswith("recipes/")==False)
                         ].unique()

df = pd.DataFrame({"recipe_urls":recipe_urls})
df['recipe_urls'] = "https://www.jamieoliver.com" + df['recipe_urls'].astype('str')

recipe_url_df = recipe_url_df.append(df).copy()

recipe_url_df.to_csv("recipe_urls.csv", sep="\t", index=False)

#next part
#===============================

recipe_df = pd.read_csv("recipe_urls.csv")

titles = []
ingredients = []
instructions = []
cook_time = []
keywords = []

soupjs = []

for i in range(len(recipe_df['recipe_urls'])):
    new_url = recipe_df['recipe_urls'][i]
    # print(new_url)
    new_soup = BeautifulSoup(requests.get(new_url).content, 'html.parser')
    # print(new_soup)
    res = json.loads(new_soup.find('script', type="application/ld+json").string)
    print(res)
    titles.append(res['name'])
    ingredients.append(res['recipeIngredient'])
    if 'totalTime' in res:
        cook_time.append(res['totalTime'])
    else:
        cook_time.append(None)
    instructions.append(res['recipeInstructions'])
    soupjs.append(res)
    # print(instructions)
    print(f'{i} is now done')
    time.sleep(5)




recipes = pd.DataFrame({'Title': titles, 'Ingredients': ingredients, 'Instructions': instructions, 'Cooking Time': cook_time})
recipes.to_csv("collected_recipes.csv", index = False)

soupjs = pd.DataFrame(soupjs)
soupjs.to_csv("uncleaned.csv", index = False)
