

function search() {
    var query = $("#queryTerm").val();
    if (query === "") {
        alert("Please enter a query term!");
    }
    else {
        $.ajax({
            type: "POST",
            url: "?handler=Search",
            beforeSend: function (xhr) {
                xhr.setRequestHeader("XSRF-TOKEN",
                    $('input:hidden[name="__RequestVerificationToken"]').val());
            },
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify(query)
        }).done(function (response) {
            $("#searchResult").html(response);
        }).fail(function (response) {
            alert("There was an error adding the project.");
            console.log(response);
        });
    }
}
