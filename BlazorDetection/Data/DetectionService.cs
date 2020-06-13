using DetectionCore.DataStructures;
using DetectionCore.Helpers;
using DetectionCore.ONNX;
using DetectionCore.YoloParser;

using Microsoft.ML;

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace BlazorDetection.Data
{
    public class DetectionService
    {
        public async IAsyncEnumerable<string> DetectAsync(IEnumerable<Bitmap> images)
        {
            //var imageArray = images.ToArray();
            var mlContext = new MLContext();
            // Создаем экземпляр OnnxModelScorer и используем его для оценки загруженных данных
            var currentPath = Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location);
            var modelPath = Path.Combine(currentPath, "Model", "tiny_yolov2", "Model.onnx");
            var modelScorer = new OnnxModelScorer(modelPath, mlContext);

            // Подготовка данных
            var tensorData = ImageData.GetTensorDataFromImages(images);

            // Загружаем данные
            var imageDataView = mlContext.Data.LoadFromEnumerable(tensorData);

            var probabilities = modelScorer.Score(imageDataView, tensorDataWiew: true);

            // Создаем экземпляр YoloOutputParser и используем его для обработки выходных данных модели
            YoloOutputParser parser = new YoloOutputParser();

            var boundingBoxes = probabilities.Select(probability => parser.ParseOutputs(probability))
                .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

            int idx = 0;

            foreach (var image in images)
            {
                var detectedObjects = boundingBoxes.ElementAt(idx);
                var imageWithLabels = ImageHelper.DrawBoundingBox(image, detectedObjects);

                LogHelper.LogDetectedObjects(image.Size.ToString(), detectedObjects);

                using (MemoryStream ms = new MemoryStream())
                {
                    imageWithLabels.Save(ms, ImageFormat.Png);
                    byte[] imageBytes = ms.ToArray();

                    // Convert byte[] to Base64 String
                    var base64String = Convert.ToBase64String(imageBytes);

                    yield return base64String;
                }

                idx++;
            }
        }
    }
}
