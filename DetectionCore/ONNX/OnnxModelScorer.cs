using System;
using System.Collections.Generic;
using System.Linq;

using Microsoft.ML;
using Microsoft.ML.Data;

using DetectionCore.DataStructures;
using DetectionCore.YoloParser;

namespace DetectionCore.ONNX
{
    public class OnnxModelScorer
    {
        private readonly string ImagesFolder;
        private readonly string ModelLocation;
        private readonly MLContext MlContext;

        private IList<YoloBoundingBox> _boundingBoxes = new List<YoloBoundingBox>();

        public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
        {
            ImagesFolder = imagesFolder;
            ModelLocation = modelLocation;
            MlContext = mlContext;
        }

        public struct ImageNetSettings
        {
            public const int ImageHeight = 416;
            public const int ImageWidth = 416;
        }

        /// <summary>
        /// Наименование входных и выходных полей модели можно посмотреть в Netron
        /// </summary>
        public struct TinyYoloModelSettings
        {
            // input tensor name
            public const string ModelInput = "image";

            // output tensor name
            public const string ModelOutput = "grid";
        }

        /// <summary>
        /// Загружает модель
        /// </summary>
        /// <param name="modelLocation"></param>
        /// <returns></returns>
        private ITransformer LoadModel(string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.ImageWidth},{ImageNetSettings.ImageHeight})");

            var data = MlContext.Data.LoadFromEnumerable(new List<ImageNetData>());
            var pipeline = MlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(MlContext.Transforms.ResizeImages(
                    outputColumnName: "image",
                    imageWidth: ImageNetSettings.ImageWidth,
                    imageHeight: ImageNetSettings.ImageHeight,
                    inputColumnName: "image")
                )
                .Append(MlContext.Transforms.ExtractPixels(outputColumnName: "image"))
                .Append(MlContext.Transforms.ApplyOnnxModel(
                    modelFile: modelLocation,
                    outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput },
                    inputColumnNames: new[] { TinyYoloModelSettings.ModelInput })
                );

            var model = pipeline.Fit(data);
            return model;
        }

        /// <summary>
        /// Создание прогнозов
        /// </summary>
        /// <param name="testData"></param>
        /// <param name="model"></param>
        /// <returns></returns>
        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            Console.WriteLine($"Images location: {ImagesFolder}");
            Console.WriteLine("");
            Console.WriteLine("=====Identify the objects in the images=====");
            Console.WriteLine("");

            // оценка данных
            IDataView scoredData = model.Transform(testData);

            // Извлекаем прогнозируемые вероятности и возвращаем их для дополнительной обработки
            IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);

            return probabilities;
        }

        /// <summary>
        /// Оценка
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public IEnumerable<float[]> Score(IDataView data)
        {
            var model = LoadModel(ModelLocation);

            return PredictDataUsingModel(data, model);
        }
    }
}
