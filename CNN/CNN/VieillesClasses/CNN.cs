using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN
{
    public class CNN
    {
        Random Generator { get; set; }
        SubNetwork[] Layers { get; set; }
        public CNN(SubNetwork[] layers, Random generator)
        {
            Generator = generator;
            Layers = layers;//BRIS D'ENCAPSULATION?!
        }
        public void FeedForward(Matrix[] inputVolume)
        {
            Layers[0].FeedForward(inputVolume);
            for (int i = 1; i < Layers.Length; ++i)
                Layers[i].FeedForward(Layers[i - 1].GetOutputVolume());
        }
        private void Backpropagate(Matrix[] inputVolume, Matrix[] desiredOutputVolume)
        {
            FeedForward(inputVolume);
            //Layers[Layers.Length - 1].BackwardPass();
            //for(int i = Layers.Length - 2; i >= 0; --i)
            //    Layers[i].BackwardPass();
        }
        public void Train(List<Matrix[]> inputs, List<Matrix[]> outputs, int miniBatchSize)
        {
        }

    }
}
