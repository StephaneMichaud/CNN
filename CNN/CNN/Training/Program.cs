using System;
using MathNet.Numerics.LinearAlgebra;
using CNN.Networks;

namespace CNN
{
    class Program
    {
        static void Main(string[] args)
        {
            ConvolutionnalLayer testCNN = new ConvolutionnalLayer(1080, 1920, 3, 24, 5);
            Matrix<double>[] input = new Matrix<double>[3];
            //double[] allo = new double[3];
            input[0] = Matrix<double>.Build.Dense(1080, 1920, (x,y)=> RandomGenerator.GenerateRandomDouble(0, 1));
            input[1] = Matrix<double>.Build.Dense(1080, 1920, (x, y) => RandomGenerator.GenerateRandomDouble(0, 1));
            input[2] = Matrix<double>.Build.Dense(1080, 1920, (x, y) => RandomGenerator.GenerateRandomDouble(0, 1));
            Matrix<double>[] output=testCNN.FeedForward(input);
            
            for (int j = 0; j < output.Length; ++j)
                Console.WriteLine(output[j]);
            Console.ReadKey();

        }
    }
}
