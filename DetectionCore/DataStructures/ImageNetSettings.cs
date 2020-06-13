using System;
using System.Collections.Generic;
using System.Text;

namespace DetectionCore.DataStructures
{
    public struct ImageNetSettings
    {
        public const int ImageHeight = 416;
        public const int ImageWidth = 416;
        public const int NumChannels = 3;
        public const int InputSize = ImageHeight * ImageWidth * NumChannels;
    }
}
