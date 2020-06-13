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
    public class ImageData
    {
        [VectorType(ImageNetSettings.NumChannels, ImageNetSettings.ImageWidth, ImageNetSettings.ImageHeight)]
        public Image image { get; set; }

        /// <summary>
        /// Method to generate sample test data. Returns 2 sample rows.
        /// </summary>
        public static ImageData[] GetTensorDataFromImages(Image[] images)
        {
            ImageData[] tensorDataList = new ImageData[images.Length];

            for (int idx = 0; idx < images.Length; idx++)
            {
                // ресайз до размеров входного тензора
                //var bitmap = ImageHelper.ResizeImage(images[idx], ImageNetSettings.ImageWidth, ImageNetSettings.ImageHeight);

                // наполнение тензора данными
                var tensorData = new ImageData { image = images[idx] };
                tensorDataList[idx] = tensorData;
            }

            return tensorDataList;
        }
    }
}
