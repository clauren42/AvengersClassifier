using System.IO;
using Microsoft.AspNetCore.Mvc;
using WebApplication2.ModelScorers;
using System.Linq;

namespace WebApplication2.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ObjectDetectionController : ControllerBase
    {
        
        private readonly IModelScorer _modelScorer;

        public ObjectDetectionController(IModelScorer modelScorer) //When using DI/IoC (IImageFileWriter imageWriter)
        {
            //Get injected dependencies
            _modelScorer = modelScorer;
            
        }

        public class Result
        {
            public string PredictedLabel { get; set; }
            //public decimal Probability { get; set; }
        }

        [HttpGet()]
        public string Get([FromQuery]string url)
        {
            string imageFileRelativePath = @"../../../" + url;
            string imageFilePath = ModelHelpers.GetAbsolutePath(imageFileRelativePath);

            var img = Path.GetFullPath(imageFilePath);
            var prediction = _modelScorer.GetPredictions(img);
            string score = prediction.Score.Max().ToString("p");
            string response = $"{prediction.PredictedLabelValue.ToString()} with {score} confidence";
            return response;
        }
    }
}
