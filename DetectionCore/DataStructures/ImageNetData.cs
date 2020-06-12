using System.Collections.Generic;
using System.IO;
using System.Linq;

using Microsoft.ML.Data;

namespace DetectionCore.DataStructures
{
    /// <summary>
    /// Класс входных данных изображения
    /// </summary>
    public class ImageNetData
    {
        /// <summary>
        /// Путь, по которому хранится изображение
        /// </summary>
        [LoadColumn(0)]
        public string ImagePath;

        /// <summary>
        /// Имя файла
        /// </summary>
        [LoadColumn(1)]
        public string Label;

        public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
        {
            return Directory
                .GetFiles(imageFolder)
                .Where(filePath => Path.GetExtension(filePath).ToLower() == ".jpg")
                .Select(filePath => new ImageNetData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
        }
    }
}
