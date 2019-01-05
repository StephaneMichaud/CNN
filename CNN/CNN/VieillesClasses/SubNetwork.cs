using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN
{
    public abstract class SubNetwork
    {
        public enum TYPE {F,C,P};
        public TYPE NetworkType { get; private set; }
        protected int InputDepth { get;  set; }
        protected int InputWidth { get;  set; }
        protected int InputHeight { get;  set; }
        protected int OutputDepth { get;  set; }
        protected int OutputWidth { get;  set; }
        protected int OutputHeight { get;  set; }
        protected Matrix[] OutputVolume { get; set; }
        public Matrix[] GetOutputVolume()
        {
            Matrix[] volume = new Matrix[OutputDepth];
            for (int i = 0; i < OutputDepth; i++)
                volume[i] = new Matrix(OutputVolume[i].GetTable());
            return volume;
        }
        public SubNetwork(TYPE networkType, int inputDepth, int inputHeight, int inputWidth)
        {
            NetworkType = networkType;
            InputDepth = inputDepth;
            InputWidth = inputWidth;
            InputHeight = inputHeight;
            ComputeOutputVolumeParameters();
        }
        protected abstract void BackwardPass(Matrix[] deltaVolume);
        protected abstract void ComputeOutputVolumeParameters();
        public abstract void FeedForward(Matrix[] inputVolume);
        public abstract void Save(FileStream fileReader);
    }
}
