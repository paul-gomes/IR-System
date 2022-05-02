using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using RecipeSearchEngine.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace RecipeSearchEngine.Pages
{
    public class IndexModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;
        private readonly IConfiguration _configuration;

        public IndexModel(ILogger<IndexModel> logger, IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
        }

        public void OnGet()
        {

        }

        public async Task<ActionResult> OnPostSearchAsync([FromBody] string query)
        {
            try
            {
                List<Recipe> recipes = new List<Recipe>();
                var url = $"{_configuration.GetSection("Api_url").Value.ToString()}{query}";
                var searchHtml = "<table class='table'><tbody>";
                using (var client = new HttpClient())
                {
                    var response = await client.GetAsync(requestUri:url);
                    string json = await response.Content.ReadAsStringAsync();
                    recipes= JsonConvert.DeserializeObject<List<Recipe>>(json);
                    foreach(var r in recipes)
                    {
                        var ct = String.IsNullOrEmpty(r.Cooking_time)  || r.Cooking_time == "NaN"? "Not Available" : r.Cooking_time.Replace("PT", "");
                        var ingredients = r.Ingredients.Trim('[', ']')
                                              .Split(",")
                                              .Select(x => x.Trim('"'))
                                              .ToArray();
                        ingredients = ingredients.Select(i => i.Replace("'", "").Trim()).ToArray();
                        string eachResult = $"<tr><td><div class='row'>" +
                            $"<h3><a href='{r.Urls}'>{r.Title}</a></h3>" +
                            $"<p class='m-2'>[ Prepare Time: {ct} ]</p>" +
                            $"<div class='row m-3 w-100'><b>Ingredients:</b> {String.Join(",", ingredients)} </div>" +
                            $"</div></td></tr>";
                        searchHtml += eachResult;

                    }
                    searchHtml += "</tbody></table>";

                }  
                return new JsonResult(searchHtml);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, string.Empty);
                return BadRequest();
            }
        }
    }
}
