using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN
{
    class Program
    {
        static void Main(string[] args)
        {
            Random generator = new Random();
            
            PN pooling = new PN(2, 2, 28, 28, 3);
            CN convolutionnal = new CN(2, 2, 28, 28, 3, 1, 4);
            FCN reLuLayer = new FCN(new int[] { },FCN.ReLU,FCN.ReLUPrime,generator);
            SubNetwork[] architecture = new SubNetwork[] { convolutionnal,pooling,convolutionnal,pooling,convolutionnal,pooling,reLuLayer};
            CNN network = new CNN(architecture, generator);

            network.FeedForward(new Matrix[0]);
        }
    }
}
