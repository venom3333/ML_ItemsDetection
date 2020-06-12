using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace DetectionCore.YoloParser
{
    public class YoloOutputParser
    {
        /// <summary>
        /// Число строк в сетке (grid), на которые делится изображение
        /// </summary>
        public const int ROW_COUNT = 13;

        /// <summary>
        /// Число столбцов в сетке (grid), на которые делится изображение
        /// </summary>
        public const int COL_COUNT = 13;

        /// <summary>
        /// Общее число значений, содержащихся в одной ячейке сетки (grid)
        /// </summary>
        public const int CHANNEL_COUNT = 125;

        /// <summary>
        /// Число ограничивающих прямоугольников (bounding boxes) в ячейке
        /// </summary>
        public const int BOXES_PER_CELL = 5;

        /// <summary>
        /// Число компонентов, содержащихся в ограничивающих прямоугольниках (bounding boxes) (X, Y, высота, ширина, достоверность)
        /// </summary>
        public const int BOX_INFO_FEATURE_COUNT = 5;

        /// <summary>
        /// Число прогнозов класса, содержащихся в каждом ограничивающем прямоугольнике (bounding box)
        /// </summary>
        public const int CLASS_COUNT = 20;

        /// <summary>
        /// Ширина одной ячейки в сетке (grid) изображения
        /// </summary>
        public const float CELL_WIDTH = 32;

        /// <summary>
        /// Высота одной ячейки в сетке изображения
        /// </summary>
        public const float CELL_HEIGHT = 32;

        /// <summary>
        /// Начальная координата текущей ячейки в сетке (grid)
        /// </summary>
        private int channelStride = ROW_COUNT * COL_COUNT;

        /// <summary>
        /// Список привязок для всех пяти ограничивающих прямоугольников (bounding boxes)
        /// </summary>
        private float[] anchors = new float[]
        {
            1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
        };

        /// <summary>
        /// Наименования классов
        /// </summary>
        private string[] labels = new string[]
        {
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };

        /// <summary>
        /// С каждым из классов связаны цвета.
        /// </summary>
        private static Color[] classColors = new Color[]
        {
            Color.Khaki,
            Color.Fuchsia,
            Color.Silver,
            Color.RoyalBlue,
            Color.Green,
            Color.DarkOrange,
            Color.Purple,
            Color.Gold,
            Color.Red,
            Color.Aquamarine,
            Color.Lime,
            Color.AliceBlue,
            Color.Sienna,
            Color.Orchid,
            Color.Tan,
            Color.LightPink,
            Color.Yellow,
            Color.HotPink,
            Color.OliveDrab,
            Color.SandyBrown,
            Color.DarkTurquoise
        };

        /// <summary>
        /// Функция-сигмоида, которая выводит число от 0 до 1.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        private float Sigmoid(float value)
        {
            var k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        /// <summary>
        /// Нормализует входной вектор в распределение вероятности
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        private float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        /// <summary>
        /// Сопоставляет элементы в выходных данных одномерной модели с соответствующей позицией в тензоре 125 x 13 x 13.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="channel"></param>
        /// <returns></returns>
        private int GetOffset(int x, int y, int channel)
        {
            // YOLO outputs a tensor that has a shape of 125x13x13, which 
            // WinML flattens into a 1D array.  To access a specific channel 
            // for a given (x,y) cell position, we need to calculate an offset
            // into the array

            // YOLO выводит тензор, имеющий форму 125x13x13,
            // который WinML превращает в одномерный массив.
            // Для доступа к определенному каналу (channel) для данной x, y позиции в ячейке
            // нам нужно вычислить смещение в массиве
            return (channel * this.channelStride) + (y * COL_COUNT) + x;
        }

        /// <summary>
        /// Извлекает измерения ограничивающего прямоугольника с помощью метода GetOffset из выходных данных модели
        /// </summary>
        /// <param name="modelOutput"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="channel"></param>
        /// <returns></returns>
        private BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
        {
            return new BoundingBoxDimensions
            {
                X = modelOutput[GetOffset(x, y, channel)],
                Y = modelOutput[GetOffset(x, y, channel + 1)],
                Width = modelOutput[GetOffset(x, y, channel + 2)],
                Height = modelOutput[GetOffset(x, y, channel + 3)]
            };
        }

        /// <summary>
        /// Извлекает значение достоверности того, что модель обнаружила объект, и использует функцию Sigmoid, чтобы преобразовать ее в процент
        /// </summary>
        /// <param name="modelOutput"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="channel"></param>
        /// <returns></returns>
        private float GetConfidence(float[] modelOutput, int x, int y, int channel)
        {
            return Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]);
        }

        /// <summary>
        /// Использует измерения ограничивающего прямоугольника и сопоставляет их с соответствующей ячейкой на изображении
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="box"></param>
        /// <param name="boxDimensions"></param>
        /// <returns></returns>
        private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
        {
            return new CellDimensions
            {
                X = ((float)x + Sigmoid(boxDimensions.X)) * CELL_WIDTH,
                Y = ((float)y + Sigmoid(boxDimensions.Y)) * CELL_HEIGHT,
                Width = (float)Math.Exp(boxDimensions.Width) * CELL_WIDTH * anchors[box * 2],
                Height = (float)Math.Exp(boxDimensions.Height) * CELL_HEIGHT * anchors[box * 2 + 1],
            };
        }

        /// <summary>
        /// Извлекает прогнозы класса для ограничивающего прямоугольника из выходных данных модели с помощью метода GetOffset
        /// и превращает их в распределение вероятности с помощью метода Softmax
        /// </summary>
        /// <param name="modelOutput"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="channel"></param>
        /// <returns></returns>
        public float[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
        {
            float[] predictedClasses = new float[CLASS_COUNT];
            int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;
            for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
            }
            return Softmax(predictedClasses);
        }

        /// <summary>
        /// Выбирает из списка прогнозируемых классов класс с наибольшей вероятностью
        /// </summary>
        /// <param name="predictedClasses"></param>
        /// <returns></returns>
        private ValueTuple<int, float> GetTopResult(float[] predictedClasses)
        {
            return predictedClasses
                .Select((predictedClass, index) => (Index: index, Value: predictedClass))
                .OrderByDescending(result => result.Value)
                .First();
        }

        /// <summary>
        /// Фильтрует перекрывающиеся ограничивающие прямоугольники с более низкими вероятностями
        /// </summary>
        /// <param name="boundingBoxA"></param>
        /// <param name="boundingBoxB"></param>
        /// <returns></returns>
        private float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
        {
            var areaA = boundingBoxA.Width * boundingBoxA.Height;

            if (areaA <= 0)
                return 0;

            var areaB = boundingBoxB.Width * boundingBoxB.Height;

            if (areaB <= 0)
                return 0;

            var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
            var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
            var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
            var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }

        /// <summary>
        /// Обработка выходных данных, создаваемых моделью
        /// </summary>
        /// <param name="yoloModelOutputs"></param>
        /// <param name="threshold"></param>
        /// <returns></returns>
        public IList<YoloBoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .3F)
        {
            var boxes = new List<YoloBoundingBox>();

            for (int row = 0; row < ROW_COUNT; row++)
            {
                for (int column = 0; column < COL_COUNT; column++)
                {
                    for (int box = 0; box < BOXES_PER_CELL; box++)
                    {
                        // вычисляем начальную позицию текущего bounding box в выходных данных одномерной модели
                        var channel = (box * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT));

                        // получаем размеры текущего bounding box
                        var boundingBoxDimensions = ExtractBoundingBoxDimensions(yoloModelOutputs, row, column, channel);

                        // получаем достоверность для текущего bounding box
                        var confidence = GetConfidence(yoloModelOutputs, row, column, channel);

                        // проходит ли достоверность по заданному порогу
                        if (confidence < threshold)
                            continue;

                        // сопоставляем текущий bounding box с текущей обрабатываемой ячейкой
                        var mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxDimensions);

                        // получение вероятности распределения прогнозируемых классов для текущего bounding box
                        float[] predictedClasses = ExtractClasses(yoloModelOutputs, row, column, channel);

                        // получаем значение и индекс класса с наибольшей вероятностью для текущего bounding box и вычисляем его "оценку"
                        var (topResultIndex, topResultScore) = GetTopResult(predictedClasses);
                        var topScore = topResultScore * confidence;

                        // проходит ли "оценка" по заданному порогу
                        if (topScore < threshold)
                            continue;

                        // добавляем новый bounding box в список
                        var boundingBox = new YoloBoundingBox
                        {
                            Dimensions = new BoundingBoxDimensions
                            {
                                X = (mappedBoundingBox.X - mappedBoundingBox.Width / 2),
                                Y = (mappedBoundingBox.Y - mappedBoundingBox.Height / 2),
                                Width = mappedBoundingBox.Width,
                                Height = mappedBoundingBox.Height
                            },
                            Confidence = topScore,
                            Label = labels[topResultIndex],
                            BoxColor = classColors[topResultIndex]
                        };

                        boxes.Add(boundingBox);
                    }
                }
            }

            return boxes;
        }

        /// <summary>
        /// Удаляет перекрывающиеся изображения (boxes)
        /// </summary>
        /// <param name="boxes"></param>
        /// <param name="limit"></param>
        /// <param name="threshold"></param>
        /// <returns></returns>
        public IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
        {
            // начальное количество активных bounding boxes
            var activeCount = boxes.Count;

            // массив с размером = количеству обнаруженных boxes
            var isActiveBoxes = new bool[boxes.Count];

            // помечаем все как активные
            for (int i = 0; i < isActiveBoxes.Length; i++)
                isActiveBoxes[i] = true;

            // сортируем список в порядке убывания (с самой большой вероятностью - первый)
            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                .OrderByDescending(b => b.Box.Confidence)
                .ToList();

            // список для хранения отфильтрованных результатов
            var results = new List<YoloBoundingBox>();

            // обрабатываем кахдый отсортированный box
            for (int i = 0; i < boxes.Count; i++)
            {
                // обрабатываем ли текущий
                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    // если результатов больше указанного лимита, выходим из цикла
                    if (results.Count >= limit)
                        break;

                    // иначе обрабатываем следующие
                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        // аналогично внешнему циклу, если box активен и готов к обработке
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;

                            if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                            {
                                // деактивируем как обработанный
                                isActiveBoxes[j] = false;

                                // уменьшаем количество активных
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }

                    if (activeCount <= 0)
                        break;
                }
            }

            return results;
        }
    }
}
