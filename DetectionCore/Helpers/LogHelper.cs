using DetectionCore.YoloParser;
using System;
using System.Collections.Generic;
using System.Text;

namespace DetectionCore.Helpers
{
    static public class LogHelper
    {
        public static void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
        {
            Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

            foreach (var box in boundingBoxes)
            {
                Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
            }

            Console.WriteLine("");
        }
    }
}
