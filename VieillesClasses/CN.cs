using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN
{
    public class CN : SubNetwork
    {
        Matrix[,] Kernels { get; set; }//[i,j]: i kernel#, j layer of the kernel
        double[] Biases { get; set; }//1 per kernel: therefore, the index represents the depth
        public int Stride { get; private set; }
        public int KernelSize { get; private set; }
        public int NbKernels { get; private set; }
        public int ZeroPadding { get; private set; }
        public CN(int stride, int kernelSize, int inputWidth, int inputHeight, int inputDepth, int zeroPadding, int nbKernels)
            : base(SubNetwork.TYPE.C, inputDepth, inputHeight, inputWidth)
        {
            Stride = stride;
            KernelSize = kernelSize;
            ZeroPadding = zeroPadding;
            NbKernels = nbKernels;
            InitializeParameters();
        }

        private void InitializeParameters()
        {
            InitializeKernels();
            InitializeBiases();
        }
        private void InitializeKernels()
        {
            Kernels = new Matrix[NbKernels,InputDepth];
            for (int i = 0; i < NbKernels; i++)
                for (int j = 0; j < InputDepth; j++)
                    Kernels[i, j] = new Matrix(new double[KernelSize,KernelSize], new Random());//LINE TO CHANGE AND MAKE RANDOM OBJECT AVAILIBLE FOR ALL SUBNETWORKS
        }
        private void InitializeBiases()
        {
            Biases = new double[NbKernels];
            for (int i = 0; i < NbKernels; i++)
            {
                Biases[i] = (new Random()).NextDouble();//LINE TO CHANGE AND MAKE RANDOM OBJECT AVAILIBLE FOR ALL SUBNETWORKS
            }
        }
        protected override void BackwardPass(Matrix[] deltaVolume)
        {

        }

        protected override void ComputeOutputVolumeParameters()
        {
            OutputWidth = (InputWidth - KernelSize + 2 * ZeroPadding) / Stride + 1;
            OutputHeight = (InputHeight - KernelSize + 2 * ZeroPadding) / Stride + 1;
            OutputDepth = NbKernels;
        }

        public override void FeedForward(Matrix[] inputVolume)
        {
            double[,] output = new double[OutputHeight, OutputWidth];
            double result;
            for (int l = 0; l < NbKernels; ++l)//Foreach kernel...
            {
                for (int i = 0; i < OutputHeight; ++i)//On the height...
                {
                    for (int j = 0; j < OutputWidth; ++j)//On the width...
                    {
                        result = 0;
                        for (int k = 0; k < InputDepth; ++k)//On the depth...
                        {
                            result += Matrix.SubMatrix(inputVolume[k], i * Stride, j * Stride, KernelSize, KernelSize) % Kernels[l, k];
                        }
                        output[i, j] = result + Biases[l];
                    }
                }
                OutputVolume[l] = new Matrix(output);
            }
        }

        public override void Save(FileStream fileStream)
        {
            throw new NotImplementedException();
        }

        public static CN Load(FileStream fileStream)
        {
            return null;
        }

        private static void LoadList(List<Matrix> parameters, ref BinaryReader binaryFileReader)
        {

        }
    }
}
