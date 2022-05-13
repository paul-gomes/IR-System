#pragma checksum "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\Components\SearchContent\default.cshtml" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "d0ff05883f34f0109a4ac1d4452d1a34dc37cf46"
// <auto-generated/>
#pragma warning disable 1591
[assembly: global::Microsoft.AspNetCore.Razor.Hosting.RazorCompiledItemAttribute(typeof(RecipeSearchEngine.Pages.Components.SearchContent.Pages_Components_SearchContent_default), @"mvc.1.0.view", @"/Pages/Components/SearchContent/default.cshtml")]
namespace RecipeSearchEngine.Pages.Components.SearchContent
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Mvc;
    using Microsoft.AspNetCore.Mvc.Rendering;
    using Microsoft.AspNetCore.Mvc.ViewFeatures;
#nullable restore
#line 1 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\_ViewImports.cshtml"
using RecipeSearchEngine;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\_ViewImports.cshtml"
using RecipeSearchEngine.Model;

#line default
#line hidden
#nullable disable
    [global::Microsoft.AspNetCore.Razor.Hosting.RazorSourceChecksumAttribute(@"SHA1", @"d0ff05883f34f0109a4ac1d4452d1a34dc37cf46", @"/Pages/Components/SearchContent/default.cshtml")]
    [global::Microsoft.AspNetCore.Razor.Hosting.RazorSourceChecksumAttribute(@"SHA1", @"ae497d8bb63a5366fd6a5f51d55218115177c9c3", @"/Pages/_ViewImports.cshtml")]
    public class Pages_Components_SearchContent_default : global::Microsoft.AspNetCore.Mvc.Razor.RazorPage<List<RecipeSearchEngine.Model.Recipe>>
    {
        #pragma warning disable 1998
        public async override global::System.Threading.Tasks.Task ExecuteAsync()
        {
            WriteLiteral("\r\n");
            WriteLiteral("\r\n");
#nullable restore
#line 4 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\Components\SearchContent\default.cshtml"
  
    int page = 1;
    int recipeCount = 1; 
    string pclass = "page_1";
    foreach (var r in Model)
    {
        if (page > 10 && page < 21)
        {
            pclass = "page_2";
        }
        else if (page > 20 && page < 31)
        {
            pclass = "page_3";
        }
        else if (page > 30 && page < 41)
        {
            pclass = "page_4";
        }
        else if (page > 40 && page < 51)
        {
            pclass = "page_5";
        }
        else
        {
            pclass = "page_1";
        }
        var ct = String.IsNullOrEmpty(r.Cooking_time) || r.Cooking_time == "NaN" ? "Not Available" : r.Cooking_time.Replace("PT", "");
        var ingredients = r.Ingredients.Trim('[', ']')
        .Split(",")
        .Select(x => x.Trim('"'))
        .ToArray();
        ingredients = ingredients.Select(i => i.Replace("'", "").Trim()).ToArray();

#line default
#line hidden
#nullable disable
            WriteLiteral("        <div");
            BeginWriteAttribute("class", " class=\"", 981, "\"", 996, 1);
#nullable restore
#line 36 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\Components\SearchContent\default.cshtml"
WriteAttributeValue("", 989, pclass, 989, 7, false);

#line default
#line hidden
#nullable disable
            EndWriteAttribute();
            WriteLiteral(">\r\n            <div class=\"row\">\r\n                <div class=\"col\">\r\n                    <h3><a");
            BeginWriteAttribute("href", " href=\'", 1092, "\'", 1106, 1);
#nullable restore
#line 39 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\Components\SearchContent\default.cshtml"
WriteAttributeValue("", 1099, r.Urls, 1099, 7, false);

#line default
#line hidden
#nullable disable
            EndWriteAttribute();
            WriteLiteral(" target=\"_blank\">");
#nullable restore
#line 39 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\Components\SearchContent\default.cshtml"
                                                     Write(r.Title);

#line default
#line hidden
#nullable disable
            WriteLiteral("</a></h3>\r\n                    <input");
            BeginWriteAttribute("class", " class=\"", 1169, "\"", 1208, 3);
            WriteAttributeValue("", 1177, "form-check-input", 1177, 16, true);
            WriteAttributeValue(" ", 1193, "r_", 1194, 3, true);
#nullable restore
#line 40 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\Components\SearchContent\default.cshtml"
WriteAttributeValue("", 1196, recipeCount, 1196, 12, false);

#line default
#line hidden
#nullable disable
            EndWriteAttribute();
            WriteLiteral(" type=\"checkbox\"");
            BeginWriteAttribute("id", " id=\"", 1225, "\"", 1238, 1);
#nullable restore
#line 40 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\Components\SearchContent\default.cshtml"
WriteAttributeValue("", 1230, r.DocId, 1230, 8, false);

#line default
#line hidden
#nullable disable
            EndWriteAttribute();
            WriteLiteral(" style=\"right:0;\">\r\n                    <h5>[Prepare Time: ");
#nullable restore
#line 41 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\Components\SearchContent\default.cshtml"
                                  Write(ct);

#line default
#line hidden
#nullable disable
            WriteLiteral("]</h5>\r\n                    <div class=\"row m-3 w-100\">\r\n                        <b>Ingredients: </b>\r\n                        ");
#nullable restore
#line 44 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\Components\SearchContent\default.cshtml"
                   Write(String.Join(", ", ingredients));

#line default
#line hidden
#nullable disable
            WriteLiteral("\r\n                    </div>\r\n                </div>\r\n            </div>\r\n            <hr />\r\n        </div>\r\n");
#nullable restore
#line 50 "C:\Users\apgomes\source\repos\PAUL\GradSchool\5. Spring2022\Information Retrieval\Project\IR-System\Application\RecipeSearchEngine\RecipeSearchEngine\Pages\Components\SearchContent\default.cshtml"
        page++;
        recipeCount++;
    }

#line default
#line hidden
#nullable disable
        }
        #pragma warning restore 1998
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.ViewFeatures.IModelExpressionProvider ModelExpressionProvider { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.IUrlHelper Url { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.IViewComponentHelper Component { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.Rendering.IJsonHelper Json { get; private set; }
        [global::Microsoft.AspNetCore.Mvc.Razor.Internal.RazorInjectAttribute]
        public global::Microsoft.AspNetCore.Mvc.Rendering.IHtmlHelper<List<RecipeSearchEngine.Model.Recipe>> Html { get; private set; }
    }
}
#pragma warning restore 1591
