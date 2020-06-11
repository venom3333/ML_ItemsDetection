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
        string AssetsRelativePath { get; set; }
        string AssetsPath { get; set; }
        string ModelFilePath { get; set; }
        string ImagesFolder { get; set; }
        string OutputFolder { get; set; }

        public FileHelper(string assetsRelativePath)
        {
            AssetsRelativePath = assetsRelativePath;
            AssetsPath = GetAbsolutePath(AssetsRelativePath);
            ModelFilePath = Path.Combine(AssetsPath, "Model", "TinyYolo2_model.onnx");
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
