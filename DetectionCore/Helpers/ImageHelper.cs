using DetectionCore.DataStructures;
using DetectionCore.ONNX;
using DetectionCore.YoloParser;

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Text;

namespace DetectionCore.Helpers
{
    public class ImageHelper
    {
        public static Image DrawBoundingBox(Stream stream, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
            Image image = Image.FromStream(stream);
            return DrawBoundingBox(image, filteredBoundingBoxes);
        }

        public static Image DrawBoundingBox(string inputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
            Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));
            return DrawBoundingBox(image, filteredBoundingBoxes);
        }

        public static Image DrawBoundingBox(Image image, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
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
                x = (uint)originalImageWidth * x / ImageNetSettings.ImageWidth;
                y = (uint)originalImageHeight * y / ImageNetSettings.ImageHeight;
                width = (uint)originalImageWidth * width / ImageNetSettings.ImageWidth;
                height = (uint)originalImageHeight * height / ImageNetSettings.ImageHeight;

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

        /// <summary>
        /// Resize the image to the specified width and height.
        /// </summary>
        /// <param name="image">The image to resize.</param>
        /// <param name="width">The width to resize to.</param>
        /// <param name="height">The height to resize to.</param>
        /// <returns>The resized image.</returns>
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using var graphics = Graphics.FromImage(destImage);

            graphics.CompositingMode = CompositingMode.SourceCopy;
            graphics.CompositingQuality = CompositingQuality.HighQuality;
            graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
            graphics.SmoothingMode = SmoothingMode.HighQuality;
            graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

            using var wrapMode = new ImageAttributes();
            wrapMode.SetWrapMode(WrapMode.TileFlipXY);
            graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);

            return destImage;
        }
    }
}
