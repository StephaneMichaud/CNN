using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace CNN.Networks
{
    class PoolingLayer
    {
        public int Stride { get; private set; }
        public int KernelSize { get; private set; }

        public PoolingLayer(FileStream fileReader)
        {

        }

        public PoolingLayer()
        {

        }
    }
}
