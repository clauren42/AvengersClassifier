using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.IO;
using ImageClassifier;

namespace Tests
{
    public class Tests
    {


        [SetUp]
        public void Setup()
        {
        }

        [TestFixture]
        public class ImageClassificationTests
        {
            private static string assetsPath = Path.Combine(Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName, "..//assets");
            static readonly string testDirectory = Path.GetFullPath("../../../assets/test");
            static readonly string trainedModelLocation = Path.GetFullPath("../../../../ImageClassification.Train/imageClassifier.zip");


            readonly TrainedModel model = new TrainedModel(trainedModelLocation);
                                      
            public static List<TestCaseData> TestCases
            {
                get
                {
                    var TestCases = new List<TestCaseData>();
                    List<ImageData> imageList = DataHelper.ReadFromFolder(testDirectory);
                    foreach (ImageData image in imageList)
                    {
                        TestCases.Add(new TestCaseData(Path.GetFullPath(image.ImagePath), image.Label));
                    }

                    return TestCases;
                }
            }

            [TestCaseSource("TestCases")]
            public void TestImages(string imagePath, string expectedLabel)
            {
                try
                {
                    var TestCase = new ImageData();
                    TestCase.ImagePath = imagePath;
                    ImagePrediction prediction = model.predictor.Predict(TestCase);
                    Console.WriteLine($"Test case {TestCase.ImagePath} predicted as {prediction.PredictedLabelValue.ToLower()} should be {expectedLabel.ToLower()}");
                    Assert.That(prediction.PredictedLabelValue.ToLower(), Is.EqualTo(expectedLabel.ToLower()));
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }
        }

     
    }
}