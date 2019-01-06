using System;
using MathNet.Numerics.LinearAlgebra;
using CNN.Networks;

namespace CNN
{
    class Program
    {
        static void Main(string[] args)
        {
            const int IMAGE_HEIGHT = 224;
            const int IMAGE_WIDTH = 224;
            ConvolutionnalLayer testCNN = new ConvolutionnalLayer(IMAGE_HEIGHT, IMAGE_WIDTH, 3, 3, 3);
            Matrix<double>[] input = new Matrix<double>[3];
            input[0] = Matrix<double>.Build.Dense(IMAGE_HEIGHT, IMAGE_WIDTH, (x,y)=> (double)RandomGenerator.GenerateRandomDouble(0, 1));
            input[1] = Matrix<double>.Build.Dense(IMAGE_HEIGHT, IMAGE_WIDTH, (x, y) => (double)RandomGenerator.GenerateRandomDouble(0, 1));
            input[2] = Matrix<double>.Build.Dense(IMAGE_HEIGHT, IMAGE_WIDTH, (x, y) => (double)RandomGenerator.GenerateRandomDouble(0, 1));
            Matrix<double>[] output=testCNN.FeedForward(input);
            //https://www.cs.toronto.edu/~kriz/cifar.html 
            //for (int j = 0; j < output.Length; ++j)
            //    Console.WriteLine(output[j]);
            Console.ReadKey();

        }
    }
}
