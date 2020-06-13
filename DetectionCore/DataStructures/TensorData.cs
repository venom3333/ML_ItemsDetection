using DetectionCore.Helpers;

using Microsoft.ML.Data;

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace DetectionCore.DataStructures
{
    /// <summary>
    /// A class to hold sample tensor data. 
    /// Member name should match the inputs that the model expects (in this
    /// case, input).
    /// </summary>
    public class TensorData
    {
        [VectorType(ImageNetSettings.ImageWidth, ImageNetSettings.ImageHeight, ImageNetSettings.NumChannels)]
        public float[] image { get; set; }

        /// <summary>
        /// Method to generate sample test data. Returns 2 sample rows.
        /// </summary>
        public static TensorData[] GetTensorDataFromImages(Image[] images)
        {
            TensorData[] tensorDataList = new TensorData[images.Length];

            for (int idx = 0; idx < images.Length; idx++)
            {
                // ресайз до размеров входного тензора
                var bitmap = ImageHelper.ResizeImage(images[idx], ImageNetSettings.ImageWidth, ImageNetSettings.ImageHeight);

                // наполнение тензора данными
                var tensorData = new TensorData { image = new float[ImageNetSettings.InputSize] };
                for (int i = 0; i < bitmap.Width; i++)
                {
                    for (int j = 0; j < bitmap.Height; j++)
                    {
                        Color pixel = bitmap.GetPixel(i, j);

                        tensorData.image[i * j * ImageNetSettings.NumChannels] = (float)pixel.R / 255;
                        tensorData.image[i * j * ImageNetSettings.NumChannels + 1] = (float)pixel.G / 255;
                        tensorData.image[i * j * ImageNetSettings.NumChannels + 2] = (float)pixel.B / 255;
                    }
                }
                tensorDataList[idx] = tensorData;
            }

            return tensorDataList;
        }
    }
}
