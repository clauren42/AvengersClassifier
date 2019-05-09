using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using ImageClassifier;

namespace WebApplication2.ModelScorers
{
    public interface IModelScorer
    {
        ImagePrediction GetPredictions(string imagesFilePath);
   
    }

    public class ModelScorer : IModelScorer
    {
        private readonly string _imagesLocation;
        private readonly string _imagesTmpFolder;
        private readonly string _modelLocation;
        private readonly MLContext _mlContext;
        public readonly TrainedModel _trainedModel;

#pragma warning disable IDE0032
        public TrainedModel trainedModel
        {
            get => _trainedModel;
        }

        public ModelScorer()
        {
            var assetsPath = ModelHelpers.GetAbsolutePath(@"../../../Model");
            _imagesTmpFolder = ModelHelpers.GetAbsolutePath(@"/wwwroot/tempImages");
            _modelLocation = Path.Combine(assetsPath, "imageClassifier.zip");
            _mlContext = new MLContext();
            this._trainedModel = new TrainedModel(_modelLocation);

        }

        public ImagePrediction GetPredictions(string imagesFilePath)
        {
            var imageInputData = new ImageData { ImagePath = Path.GetFullPath(imagesFilePath) };
            var prediction = _trainedModel.predictor.Predict(imageInputData);
            
            return prediction;
        }
    }
}

