using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN
{
    public class PN: SubNetwork
    {
        public int Stride { get; private set; }
        public int KernelSize { get; private set; }

        public PN(int stride, int kernelSize, int inputWidth, int inputHeight, int inputDepth)
            :base(SubNetwork.TYPE.P,inputDepth,inputHeight,inputWidth)
        {
            Stride = stride;
            KernelSize = kernelSize;
        }
        protected override void ComputeOutputVolumeParameters()
        {
            OutputWidth = (InputWidth - KernelSize) / Stride + 1;
            OutputHeight = (InputHeight - KernelSize) / Stride + 1;
            OutputDepth = InputDepth;
        }

        public override void FeedForward(Matrix[] inputVolume)
        {
            Matrix filter;
            double[,] output = new double[OutputHeight, OutputWidth];
            for(int k = 0; k < InputDepth; ++k)//The max pooling operation is applied on each depth layer.
            {
                for (int i = 0; i < OutputHeight; ++i)
                {
                    for (int j = 0; j < OutputWidth; ++j)
                    {
                        filter = Matrix.SubMatrix(inputVolume[k], i * Stride, j * Stride, KernelSize, KernelSize);
                        output[i, j] = FindMax(filter);
                    }
                }
                OutputVolume[k] = new Matrix(output);
            }
        }
        private double FindMax(Matrix filter)
        {
            double max = filter[0, 0];
            for (int i = 0; i < filter.M; ++i)
                for (int j = 0; j < filter.N; ++j)
                    max = Math.Max(max, filter[i, j]);
            return max;
        }
        protected override void BackwardPass(Matrix[] deltaVolume)
        {
            throw new NotImplementedException();
        }

        public override void Save(FileStream fileStream)
        {
            throw new NotImplementedException();
        }
    }
}
