using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using System.Linq;

namespace ImageClassifier
{
  
        public class TrainedModel
        {
            private static MLContext mlContext = new MLContext(seed: 1);
            public readonly PredictionEngine<ImageData, ImagePrediction> predictor;

        public TrainedModel(string trainedModelPath)
            {
                Console.WriteLine($"Loading model from {trainedModelPath}");
                ITransformer model = mlContext.Model.Load(trainedModelPath, out var modelInputSchema);
                this.predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            }
        }
    public class DataHelper
    {
        public static List<ImageData> ReadFromFolder(string folder, int MaxImagesperClass =-1)
        {
            Console.WriteLine($"Reading images from {folder}");
            List<ImageData> imageList = new List<ImageData>();
            int nestedFolders = 0;
            bool filterImages = false;
            if (MaxImagesperClass > 1) { filterImages = true; };
            nestedFolders = Directory.GetDirectories(folder).Count();

            if (nestedFolders > 0)
            {
                foreach (var name in Directory.EnumerateDirectories(folder))
                {
                    var label = Path.GetFileName(name);
                    
                    foreach (var f in Directory.EnumerateFiles(name, "*.jpg"))
                        if (!filterImages || filterImages && imageList.Count < MaxImagesperClass)
                        {
                            imageList.Add(new ImageData { ImagePath = Path.GetFullPath(f), Label = label });
                        }
                }
            }
            else
            {
                var label = Path.GetFileName(folder);
                foreach (var f in Directory.EnumerateFiles(folder, "*.jpg"))
                    if (!filterImages || filterImages && imageList.Count < MaxImagesperClass)
                    {
                        imageList.Add(new ImageData { ImagePath = Path.GetFullPath(f), Label = label });
                    }
            }
            return imageList;
        }
    }
    
    public class ImageData
        {
            public string ImagePath { get; set; }
            public string Label { get; set; }
        }

    public class ImagePrediction
    {
        public float[] Score { get; set; }

        public string PredictedLabelValue { get; set; }
    }

    public class PredictionResult
        {
            public string ImageName;
            public string ActualLabel;
            public string PredictedLabel;
            public float Score;

            public PredictionResult(string ImageName, string PredictedLabel, float Score, string ActualLabel = "")
            {
                this.ActualLabel = ActualLabel;
                this.PredictedLabel = PredictedLabel;
                this.ImageName = ImageName;
                this.Score = Score;
            }
        }
}
