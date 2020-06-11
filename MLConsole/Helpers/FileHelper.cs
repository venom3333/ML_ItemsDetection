using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MLConsole.Helpers
{
    /// <summary>
    /// Помощник работы с файлами
    /// </summary>
    public class FileHelper
    {
        public string AssetsRelativePath { get; set; }
        public string AssetsPath { get; set; }
        public string ModelFilePath { get; set; }
        public string ImagesFolder { get; set; }
        public string OutputFolder { get; set; }

        public FileHelper(string assetsRelativePath)
        {
            AssetsRelativePath = assetsRelativePath;
            AssetsPath = GetAbsolutePath(AssetsRelativePath);
            ModelFilePath = Path.Combine(AssetsPath, "Model", "tiny_yolov2", "Model.onnx");
            ImagesFolder = Path.Combine(AssetsPath, "images");
            OutputFolder = Path.Combine(AssetsPath, "images", "output");
        }

        /// <summary>
        /// Получение абсолютного пути файла от относительного
        /// </summary>
        /// <param name="relativePath"></param>
        /// <returns></returns>
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
