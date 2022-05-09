using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using RecipeSearchEngine.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RecipeSearchEngine.ViewComponents
{
    public class SearchContentViewComponent : ViewComponent
    {
        private readonly ILogger<SearchContentViewComponent> _logger;

        public SearchContentViewComponent(ILogger<SearchContentViewComponent> logger)
        {
            _logger = logger;
        }

        public async Task<IViewComponentResult> InvokeAsync(List<Recipe> recipes)
        {
            return View(recipes);
        }
    }
}
