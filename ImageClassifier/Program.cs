using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

namespace ImageClassifier
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length > 0)
            {
                string pathtoImages = args[0].ToString();
                Console.WriteLine($"Classifying images at {pathtoImages}");
                ClassifyImages(pathtoImages);
            };
        }
        static void ClassifyImages(string pathtoImages)
        {
            string trainedModelLocation = Path.GetFullPath("../../../../ImageClassification.Train/imageClassifier.zip");
            
            TrainedModel model = new TrainedModel(trainedModelLocation);
            List<ImageData> imageList = DataHelper.ReadFromFolder(pathtoImages);

            ImagePrediction prediction = new ImagePrediction();
            List<PredictionResult> results = new List<PredictionResult>();

            foreach (ImageData image in imageList)
            {
                prediction = model.predictor.Predict(image);

                PredictionResult result = new PredictionResult(image.ImagePath.ToString(), prediction.PredictedLabelValue, prediction.Score.Max());
                results.Add(result);
            }

            var output = JsonConvert.SerializeObject(results, Formatting.Indented);
            Console.WriteLine(output);

        }


    }    
}
