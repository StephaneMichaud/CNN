using System;
using MathNet.Numerics.LinearAlgebra;
using CNN.Networks;

namespace CNN
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            int i = 0;
            Matrix<double> test = Matrix<double>.Build.Dense(3, 3,(x,y)=> i++);
            Console.WriteLine(test);
            double[] colMajorArray = test.ToColumnMajorArray();
            Console.WriteLine();
            ConvolutionnalLayer testCNN = new ConvolutionnalLayer(3, 3, 3, 2, 2);
            Matrix<double>[] input = new Matrix<double>[3];
            //double[] allo = new double[3];
            double[,] layer1 = { { 1, 2, 8 }, { 4, 3, 23 }, { 6, 5, 10 } };
            input[0] = Matrix<double>.Build.DenseOfArray(layer1);
            Console.WriteLine(input[0].SubMatrix(0, 2, 0, 2));
            input[1] = Matrix<double>.Build.Dense(3, 3, 4);
            input[2] = Matrix<double>.Build.Dense(3, 3, 8);
            testCNN.FeedForward(input);
            
            for (int j = 0; j < colMajorArray.Length; ++j)
                Console.WriteLine(colMajorArray[j]);
            Console.ReadKey();

        }
    }
}
