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
        public async Task<List<string>> DetectAsync(IEnumerable<Bitmap> images)
        {
            var imageArray = images.ToArray();
            var mlContext = new MLContext();
            // ������� ��������� OnnxModelScorer � ���������� ��� ��� ������ ����������� ������
            var currentPath = Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location);
            var modelPath = Path.Combine(currentPath, "Model", "tiny_yolov2", "Model.onnx");
            var modelScorer = new OnnxModelScorer(modelPath, mlContext);

            // ���������� ������
            // var tensorDataArray = TensorData.GetTensorDataFromImages(imageArray);
            var tensorDataArray = ImageData.GetTensorDataFromImages(images.ToArray());

            // ��������� ������
            var imageDataView = mlContext.Data.LoadFromEnumerable(tensorDataArray);

            var probabilities = modelScorer.Score(imageDataView, tensorDataWiew: true);

            // ������� ��������� YoloOutputParser � ���������� ��� ��� ��������� �������� ������ ������
            YoloOutputParser parser = new YoloOutputParser();

            var boundingBoxes = probabilities.Select(probability => parser.ParseOutputs(probability))
                .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

            var imagesCount = images.Count();
            var imageStrings = new List<string>(imagesCount);

            for (var i = 0; i < imagesCount; i++)
            {
                var detectedObjects = boundingBoxes.ElementAt(i);
                var imageWithLabels = ImageHelper.DrawBoundingBox(imageArray[i], detectedObjects);

                LogHelper.LogDetectedObjects(imageArray[i].Size.ToString(), detectedObjects);

                using (MemoryStream ms = new MemoryStream())
                {
                    imageWithLabels.Save(ms, ImageFormat.Png);
                    byte[] imageBytes = ms.ToArray();

                    // Convert byte[] to Base64 String
                    var base64String = Convert.ToBase64String(imageBytes);
                    imageStrings.Add(base64String);
                }
            }

           
            return imageStrings;
        }
    }
}
