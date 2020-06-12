using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace BlazorInterface.Data
{
    public class DetectionService
    {
        public async Task DetectAsync()
        {
            await Task.Run(() => Thread.Sleep(200));
            return;
        }
    }
}
