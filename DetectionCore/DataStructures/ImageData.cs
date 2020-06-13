using DetectionCore.Helpers;

using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

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
        [ImageType(ImageNetSettings.ImageHeight, ImageNetSettings.ImageWidth)]
        //[VectorType(ImageNetSettings.NumChannels, ImageNetSettings.ImageWidth, ImageNetSettings.ImageHeight)]
        public Bitmap image { get; set; }

        /// <summary>
        /// Method to generate sample test data. Returns 2 sample rows.
        /// </summary>
        public static IEnumerable<ImageData> GetTensorDataFromImages(IEnumerable<Bitmap> images)
        {
            foreach (var image in images)
            {
                var tensorData = new ImageData { image = image };
                yield return tensorData;
            }
        }
    }
}
