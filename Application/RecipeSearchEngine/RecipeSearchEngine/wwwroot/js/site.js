

function search() {
    var query = $("#queryTerm").val();
    if (query === "") {
        alert("Please enter a query term!");
    }
    else {
        $("#spinner").removeClass("hide");
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
            $("#spinner").addClass("hide");
            $("#searchResult").html(response);
            $(".page_2").addClass("hidden");
            $(".page_3").addClass("hidden");
            $(".page_4").addClass("hidden");
            $(".page_5").addClass("hidden");

            $(".p_" + 1).addClass("active");
            $(".p_" + 2).removeClass("active");
            $(".p_" + 3).removeClass("active");
            $(".p_" + 4).removeClass("active");
            $(".p_" + 5).removeClass("active");
            $("#pagination_btm").removeClass("hidden");

            $("#expandQuery").removeClass("hide");
        }).fail(function (response) {
            alert("There was an error. Please try again!");
            console.log(response);
        });
    }
}

function changePage(pageNum) {
    var activePage = pageNum;
    var notActivePage = [1, 2, 3, 4, 5];
    notActivePage.splice((pageNum - 1), 1);

    $(".page_" + notActivePage[0]).addClass("hidden");
    $(".page_" + notActivePage[1]).addClass("hidden");
    $(".page_" + notActivePage[2]).addClass("hidden");
    $(".page_" + notActivePage[3]).addClass("hidden");
    $(".p_" + notActivePage[0]).removeClass("active");
    $(".p_" + notActivePage[1]).removeClass("active");
    $(".p_" + notActivePage[2]).removeClass("active");
    $(".p_" + notActivePage[3]).removeClass("active");

    $(".page_" + activePage).removeClass("hidden");
    $(".p_" + activePage).addClass("active");
}

function expandQuery() {
    var relevant = [];
    $('input.form-check-input:checkbox:checked').each(function () {
        relevant.push($(this)[0].id);
    });
    var query = $("#queryTerm").val();
    if (relevant.length > 0 && query != "") {
        $("#spinner").removeClass("hide");
        $.ajax({
            type: "POST",
            url: "?handler=ExpandQuery&query="+ query,
            beforeSend: function (xhr) {
                xhr.setRequestHeader("XSRF-TOKEN",
                    $('input:hidden[name="__RequestVerificationToken"]').val());
            },
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify(relevant)
        }).done(function (response) {
            $("#spinner").addClass("hide");
            $("#searchResult").html(response);
            $(".page_2").addClass("hidden");
            $(".page_3").addClass("hidden");
            $(".page_4").addClass("hidden");
            $(".page_5").addClass("hidden");

            $(".p_" + 1).addClass("active");
            $(".p_" + 2).removeClass("active");
            $(".p_" + 3).removeClass("active");
            $(".p_" + 4).removeClass("active");
            $(".p_" + 5).removeClass("active");
            $("#pagination_btm").removeClass("hidden");

            $("#expandQuery").removeClass("hide");
        }).fail(function (response) {
            alert("There was an error. Please try again!");
            console.log(response);
        });
    }
    else {
        alert("Please select a few relevant recipes!")
    }
}
