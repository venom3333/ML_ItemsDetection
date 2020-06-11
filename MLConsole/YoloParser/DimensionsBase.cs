using System;
using System.Collections.Generic;
using System.Text;

namespace MLConsole.YoloParser
{
    public class DimensionsBase
    {
        /// <summary>
        /// Расположение объекта вдоль оси X
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// Расположение объекта вдоль оси Y
        /// </summary>
        public float Y { get; set; }

        /// <summary>
        /// Высота объекта
        /// </summary>
        public float Height { get; set; }

        /// <summary>
        /// Ширина объекта
        /// </summary>
        public float Width { get; set; }
    }
}
