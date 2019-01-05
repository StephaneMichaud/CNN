using System;
using MathNet.Numerics.LinearAlgebra;

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
            for (int j = 0; j < colMajorArray.Length; ++j)
                Console.WriteLine(colMajorArray[j]);
            Console.ReadKey();

        }
    }
}
