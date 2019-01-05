using System;
using System.Collections.Generic;
using System.Text;

namespace CNN.Networks
{
    static class RandomGenerator
    {
        static private Random generator = new Random();

        static public double GenerateRandomDouble(double min, double max)
        {
            return generator.NextDouble()*(max-min) + min;
        }
    }
}
