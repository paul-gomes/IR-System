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
                var url = $"{_configuration.GetSection("Api_url").Value.ToString()}search/{query}";
                using (var client = new HttpClient())
                {
                    var response = await client.GetAsync(requestUri:url);
                    string json = await response.Content.ReadAsStringAsync();
                    recipes= JsonConvert.DeserializeObject<List<Recipe>>(json);
                }
                return ViewComponent("SearchContent", recipes);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, string.Empty);
                return BadRequest();
            }
        }

        public async Task<ActionResult> OnPostExpandQueryAsync([FromQuery] string query, [FromQuery] string irrelevant, [FromBody] List<string> relevant)
        {
            try
            {
                List<Recipe> recipes = new List<Recipe>();
                var url = $"{_configuration.GetSection("Api_url").Value.ToString()}expand/{query}/{irrelevant}/{@String.Join(", ", relevant)}";
                using (var client = new HttpClient())
                {
                    var response = await client.GetAsync(requestUri: url);
                    string json = await response.Content.ReadAsStringAsync();
                    recipes = JsonConvert.DeserializeObject<List<Recipe>>(json);
                }
                return ViewComponent("SearchContent", recipes);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, string.Empty);
                return BadRequest();
            }
        }
    }
}
