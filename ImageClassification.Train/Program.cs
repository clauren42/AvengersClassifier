using System;
using System.IO;
using System.Threading.Tasks;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using ImageClassifier;
using Newtonsoft.Json;


namespace ImageClassification
{
    public class Program
    {

        static void Main()
        {
            string assetsPath = Path.Combine(new FileInfo(typeof(Program).Assembly.Location).Directory.FullName, "assets");

            var trainingImages = Path.GetFullPath("../../../assets/inputs/train"); // "C:\\repo\\Image-classification-transfer-learning\\grocery\\train2\\";
            var testingImages = Path.GetFullPath("../../../../ImageClassification.Test/assets/test/thor");
            var featurizerModel = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb");
            string trainedModelLocation = Path.GetFullPath("../../../imageClassifier.zip");
           
            //Train model
            try
            { 
                var modelBuilder = new ModelBuilder(trainingImages, featurizerModel, trainedModelLocation);
                modelBuilder.Train();
            }
            catch (Exception ex)
            {
                ConsoleHelpers.ConsoleWriteException(ex.Message);
            }

            //Test model
            ConsoleHelpers.ConsoleWriteHeader("Test model with few sample images");
            try
            {
                ConsoleHelpers.ConsoleWriteHeader("Load saved model");
                TrainedModel model = new TrainedModel(trainedModelLocation);

                ImagePrediction prediction = new ImagePrediction();
                List<PredictionResult> results = new List<PredictionResult>();

                List<ImageData> imageList = DataHelper.ReadFromFolder(testingImages);
                foreach (ImageData image in imageList)
                {
                    prediction = model.predictor.Predict(image);

                    PredictionResult result = new PredictionResult(Path.GetFileName(image.ImagePath).ToString(), prediction.PredictedLabelValue, prediction.Score.Max());
                    results.Add(result);
                }

                var output = JsonConvert.SerializeObject(results, Formatting.Indented);
                Console.WriteLine(output);

            }
            catch (Exception ex)
            {
                ConsoleHelpers.ConsoleWriteException(ex.Message);
            }
        }

    }

    public class ModelBuilder
    {

        private readonly string imagesFolder;
        private readonly string inputModelLocation;
        private readonly string outputModelLocation;
        private readonly MLContext mlContext;
        private static string LabelTokey = nameof(LabelTokey);
        private static string ImageReal = nameof(ImageReal);
        private static string PredictedLabelValue = nameof(PredictedLabelValue);

        public ModelBuilder(string imagesFolder, string inputModelLocation, string outputModelLocation)
        {
            this.imagesFolder = imagesFolder;
            this.inputModelLocation = inputModelLocation;
            this.outputModelLocation = outputModelLocation;
            mlContext = new MLContext(seed: 1);
        }

        private struct ImageSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
            public const float mean = 117;
            public const float scale = 1;
            public const bool channelsLast = true;
        }

        public void Train()
        {
            var featurizerModelLocation = inputModelLocation;

            ConsoleHelpers.ConsoleWriteHeader("Read model");
            Console.WriteLine($"Model location: {featurizerModelLocation}");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Default parameters: image size=({ImageSettings.imageWidth},{ImageSettings.imageHeight}), image mean: {ImageSettings.mean}");

            // Get the training data sample images
            ConsoleHelpers.ConsoleWriteHeader("Collecting sample training data");

            var data = mlContext.Data.LoadFromEnumerable(DataHelper.ReadFromFolder(imagesFolder));
            Images.SummarizeTrainingData(imagesFolder);

            // Train the model
            ConsoleHelpers.ConsoleWriteHeader("Training classification model");

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelTokey, inputColumnName: "Label")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: "", inputColumnName: nameof(Images.ImagePath)))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: ImageSettings.imageWidth, imageHeight: ImageSettings.imageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: ImageSettings.channelsLast, offsetImage: ImageSettings.mean))
                .Append(mlContext.Model.LoadTensorFlowModel(inputModelLocation).
                        ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelTokey, featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            ITransformer model = pipeline.Fit(data);

            // Save the model to assets/outputs
            ConsoleHelpers.ConsoleWriteHeader("Save model to local file");

            mlContext.Model.Save(model, data.Schema, outputModelLocation);

            Console.WriteLine($"Model saved: {outputModelLocation}");

            // Get some performance metrics on the model
            var trainData = model.Transform(data);
            var classificationContext = mlContext.MulticlassClassification;

            ConsoleHelpers.ConsoleWriteHeader("Evaluating classification metrics");
            var metrics = classificationContext.Evaluate(trainData, labelColumnName: LabelTokey, predictedLabelColumnName: "PredictedLabel");
            Console.WriteLine($"Total Log Loss is: {metrics.LogLoss}");
            Console.WriteLine($"Per Class LogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");


            
        }

        
    }

    public class Images
        {
            public string ImagePath { get; set; }
            public string Label { get; set; }

            public static void SummarizeTrainingData(string folder)
            {
                foreach (var name in Directory.EnumerateDirectories(folder))
                {
                    ConsoleHelpers.ConsoleWriteClassStatistics(Path.GetFileName(name), Directory.EnumerateFiles(name, "*.jpg").Count());
                }
            }
        }


    public class ImageWithLabelPrediction : ImagePrediction
    {
        public ImageWithLabelPrediction(ImagePrediction pred, string label)
        {
            Label = label;
            Score = pred.Score;
            PredictedLabelValue = pred.PredictedLabelValue;
        }

        public string Label;
    }

    public static class ConsoleHelpers
    {
        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new String('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsolePressAnyKey()
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(" ");
            Console.WriteLine("Press any key to finish.");
            Console.ReadKey();
        }

        public static void ConsoleWriteException(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Red;
            const string exceptionTitle = "EXCEPTION";
            Console.WriteLine(" ");
            Console.WriteLine(exceptionTitle);
            Console.WriteLine(new String('#', exceptionTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
        }

        public static void ConsoleWriteImagePrediction(string ImagePath, string PredictedLabel, float Probability)
        {
            var defaultForeground = Console.ForegroundColor;
            var labelColor = ConsoleColor.Magenta;
            var probColor = ConsoleColor.Blue;

            Console.ForegroundColor = defaultForeground;
            Console.Write("ImagePath: ");
            Console.ForegroundColor = labelColor;
            Console.Write($"{Path.GetFileName(ImagePath)}");
            Console.ForegroundColor = defaultForeground;
            Console.Write(" predicted as ");
            Console.ForegroundColor = labelColor;
            Console.Write(PredictedLabel);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" with score ");
            Console.ForegroundColor = probColor;
            Console.Write(Probability);
            Console.ForegroundColor = defaultForeground;
            Console.WriteLine("");

        }

        public static void ConsoleWriteClassStatistics(string ClassName, int Samples)
        {
            var defaultForeground = Console.ForegroundColor;
            var labelColor = ConsoleColor.Magenta;
            var probColor = ConsoleColor.Blue;

            Console.ForegroundColor = defaultForeground;
            Console.Write("Class Name: ");
            Console.ForegroundColor = labelColor;
            Console.Write($"{ClassName}");
            Console.ForegroundColor = defaultForeground;
            Console.Write(" Samples: ");
            Console.ForegroundColor = probColor;
            Console.Write(Samples);
            Console.ForegroundColor = defaultForeground;
            Console.WriteLine("");
        }
    }
}
