using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;

namespace MLConsole.YoloParser
{
    public class YoloBoundingBox
    {
        /// <summary>
        /// Размеры ограничивающего прямоугольника
        /// </summary>
        public BoundingBoxDimensions Dimensions { get; set; }

        /// <summary>
        /// Класс объекта, обнаруженного в ограничивающем прямоугольнике
        /// </summary>
        public string Label { get; set; }

        /// <summary>
        /// Достоверность класса
        /// </summary>
        public float Confidence { get; set; }

        /// <summary>
        /// Прямоугольное представление измерений ограничивающего прямоугольника
        /// </summary>
        public RectangleF Rect
        {
            get { return new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
        }

        /// <summary>
        /// Цвет, связанный с соответствующим классом, который используется для рисования изображения
        /// </summary>
        public Color BoxColor { get; set; }
    }
}
