using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using DetectionCore.YoloParser;
using DetectionCore.DataStructures;
using MLConsole.Helpers;
using DetectionCore.ONNX;
using DetectionCore.Helpers;

namespace MLConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                var relativePath = @"../../../assets";
                var filehelper = new FileHelper(relativePath);
                var images = ImageNetData.ReadFromFile(filehelper.ImagesFolder);

                var mlContext = new MLContext();

                // Загруэаем данные
                var imageDataView = mlContext.Data.LoadFromEnumerable(images);

                // Создаем экземпляр OnnxModelScorer и используем его для оценки загруженных данных
                var modelScorer = new OnnxModelScorer(filehelper.ImagesFolder, filehelper.ModelFilePath, mlContext);
                var probabilities = modelScorer.Score(imageDataView);

                // Создаем экземпляр YoloOutputParser и используем его для обработки выходных данных модели
                YoloOutputParser parser = new YoloOutputParser();

                var boundingBoxes = probabilities.Select(probability => parser.ParseOutputs(probability))
                    .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

                for (var i = 0; i < images.Count(); i++)
                {
                    string imageFileName = images.ElementAt(i).Label;
                    var detectedObjects = boundingBoxes.ElementAt(i);
                    var imageWithLabels = ImageHelper.DrawBoundingBox(filehelper.ImagesFolder, filehelper.OutputFolder, imageFileName, detectedObjects);
                    
                    if (!Directory.Exists(filehelper.OutputFolder))
                    {
                        Directory.CreateDirectory(filehelper.OutputFolder);
                    }

                    imageWithLabels.Save(Path.Combine(filehelper.OutputFolder, imageFileName));

                    LogDetectedObjects(imageFileName, detectedObjects);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("========= End of Process..Hit any Key ========");
            Console.ReadLine();
        }

        private static void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
        {
            Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

            foreach (var box in boundingBoxes)
            {
                Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
            }

            Console.WriteLine("");
        }
    }
}
