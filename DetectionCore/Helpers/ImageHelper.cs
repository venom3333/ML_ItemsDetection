using DetectionCore.ONNX;
using DetectionCore.YoloParser;

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Text;

namespace DetectionCore.Helpers
{
    public class ImageHelper
    {
        public static Image DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
            Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));
            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;

            foreach (var box in filteredBoundingBoxes)
            {
                // получаем размеры ограничивающего прямоугольника
                var x = (uint)Math.Max(box.Dimensions.X, 0);
                var y = (uint)Math.Max(box.Dimensions.Y, 0);
                var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
                var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

                // Поскольку размеры ограничивающего прямоугольника соответствуют входным данным модели 416 x 416,
                // масштабируем размеры ограничивающего прямоугольника в соответствии с фактическим размером изображения
                x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.ImageWidth;
                y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.ImageHeight;
                width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.ImageWidth;
                height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.ImageHeight;

                string text = $"{box.Label} ({box.Confidence * 100:0}%)";

                // Чтобы нарисовать что-то на изображении, преобразовываем его в объект Graphics.
                using Graphics thumbnailGraphic = Graphics.FromImage(image);

                // настраиваем параметры объекта Graphics
                thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                // задаем параметры шрифта и цвета для текста
                Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                SolidBrush fontBrush = new SolidBrush(Color.Black);
                Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                Pen pen = new Pen(box.BoxColor, 3.2f);
                SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                // Создаем и заполняем прямоугольник над ограничивающей рамкой, которая будет содержать текст
                // Это поможет выделить текст и улучшить удобочитаемость
                thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);

                // рисуем текст и прямоугольник на изображении
                thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);
                thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
            }

            return image;
        }
    }
}
