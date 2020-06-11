using Microsoft.ML.Data;

namespace MLConsole.DataStructures
{
    /// <summary>
    /// Класс прогноза данных
    /// </summary>
    public class ImageNetPrediction
    {
        /// <summary>
        /// Содержит измерения, оценку объекта и вероятности класса для каждого ограничивающего прямоугольника, обнаруженного в изображении
        /// </summary>
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
}
