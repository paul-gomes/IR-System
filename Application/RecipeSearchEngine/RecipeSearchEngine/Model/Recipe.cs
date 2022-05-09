using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RecipeSearchEngine.Model
{
    public class Recipe
    {
        public string DocId { get; set; }
        public string Title { get; set; }
        public string Ingredients { get; set; }
        public string Urls { get; set; }
        public string Cooking_time { get; set; }
    }
}
