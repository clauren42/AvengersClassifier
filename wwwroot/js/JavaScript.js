
$(function () {
    $(".input-image").click(function (e) {
        $(".input-image").removeClass("active");
        $(this).addClass("active");

        var url = $(this).attr("src");

        $.ajax({
            url: "/api/ObjectDetection?url=" + url,
            type: "GET",
            success: function (result) {
                var data = result;
                $("#predictionDiv").html(data);
                
            },
            error: function (e) {
                var x = e;
            }
        });
    });
});

