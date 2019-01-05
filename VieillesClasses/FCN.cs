using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN
{
    public class FCN:SubNetwork
    {
        #region Constants and properties.
        const double LEARNING_CONSTANT_WEIGHTS = -0.001f;
        const double LEARNING_CONSTANT_BIASES = -0.0015f;
        Func<double, double> DEFAULT_ACTIVATION = ReLU;
        Func<double, double> DEFAULT_ACTIVATION_PRIME = ReLUPrime;

        Random Generator { get; set; }
        List<Matrix> Weights { get; set; }
        List<Matrix> Biases { get; set; }
        List<Matrix> Activity { get; set; }
        List<Matrix> WeightedInput { get; set; }
        List<Matrix> DeltaMatrices { get; set; }
        List<Matrix> Gradients { get; set; }
        List<Matrix> MiniBatchDeltaMatrices { get; set; }
        List<Matrix> MiniBatchGradients { get; set; }
        Func<double, double> Activation { get; set; }
        Func<double, double> ActivationPrime { get; set; }
        public double[,] GetResult()
        {
            return Activity[Activity.Count - 1].GetTable();
        }
        #endregion

        #region Initialization
        public FCN(string fileName)
        {
            int M = 0;
            int N = 0;
            double[,] table;
            Activity = new List<Matrix>();
            WeightedInput = new List<Matrix>();
            DeltaMatrices = new List<Matrix>();
            Weights = new List<Matrix>();
            Biases = new List<Matrix>();

            FileStream fileStream = new FileStream(fileName, FileMode.Open);
            BinaryReader binaryFileReader = new BinaryReader(fileStream);

            Weights.Capacity = binaryFileReader.ReadInt32();
            for (int i = 0; i < Weights.Capacity; ++i)
            {
                M = binaryFileReader.ReadInt32();
                N = binaryFileReader.ReadInt32();
                table = new double[M, N];
                for (int j = 0; j < M; ++j)
                    for (int k = 0; k < N; ++k)
                        table[j, k] = binaryFileReader.ReadDouble();
                Weights.Add(new Matrix(table));
            }

            Biases.Capacity = binaryFileReader.ReadInt32();
            for (int i = 0; i < Biases.Capacity; ++i)
            {
                M = binaryFileReader.ReadInt32();
                N = binaryFileReader.ReadInt32();
                table = new double[M, N];
                for (int j = 0; j < M; ++j)
                    for (int k = 0; k < N; ++k)
                        table[j, k] = binaryFileReader.ReadDouble();
                Biases.Add(new Matrix(table));
            }

            Activation = DEFAULT_ACTIVATION;
            ActivationPrime = DEFAULT_ACTIVATION_PRIME;

            binaryFileReader.Close();
            fileStream.Close();
        }
        public FCN(TYPE networkType,int inputDepth,int inputHeight,int inputWidth,int[] architecture, Func<double, double> activation, Func<double, double> activationPrime, Random generator)
            :base(networkType,inputDepth,inputHeight,inputWidth)
        {
            Generator = generator;
            Activation = activation;
            ActivationPrime = activationPrime;
            InitializeWeights(architecture);
            InitializeBiases(architecture);
            Activity = new List<Matrix>();
            WeightedInput = new List<Matrix>();
            DeltaMatrices = new List<Matrix>();
        }

        private void InitializeBiases(int[] architecure)
        {
            Biases = new List<Matrix>();
            for (int i = 1; i < architecure.GetLength(0); ++i)
                Biases.Add(new Matrix(new double[architecure[i], 1], Generator));
        }

        private void InitializeWeights(int[] architecture)
        {
            Weights = new List<Matrix>();
            for (int i = 1; i < architecture.GetLength(0); ++i)
                Weights.Add(new Matrix(new double[architecture[i], architecture[i - 1]], Generator));
        }
        #endregion

        #region Activation Functions.
        public static double Sigmoid(double x)
        {
            return (double)(1 / (1 + Math.Exp(-x)));
        }

        public static double SigmoidPrime(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        public static double Tanh(double x)
        {
            return 2 * Sigmoid(2 * x) - 1;
        }

        public static double TanhPrime(double x)
        {
            return 1 - Tanh(x) * Tanh(x);
        }

        public static double ReLU(double x)
        {
            return Math.Max(x, 0);
        }

        public static double ReLUPrime(double x)
        {
            if (x > 0)
                return 1;
            else
                return 0;
        }

        public static double[,] SoftMax(double[,] x)
        {
            double sum = 0;
            double[,] s = new double[x.GetLength(0), 1];
            for (int i = 0; i < x.GetLength(0); ++i)
            {
                for (int j = 0; j < x.GetLength(0); ++j)
                    sum += Math.Exp(x[j, 0]);
                s[i, 0] = Math.Exp(x[i, 0]) / sum;
                sum = 0;
            }
            return s;
        }

        public static double[,] SoftMaxPrime(double[,] x)
        {
            double sum = 0;
            double[,] sPrime = new double[x.GetLength(0), 1];
            for (int j = 0; j < x.GetLength(0); ++j)
                sum += Math.Exp(x[j, 0]);
            for (int i = 0; i < x.GetLength(0); ++i)
                sPrime[i, 0] = (Math.Exp(x[i, 0])*sum- Math.Exp(x[i, 0])* Math.Exp(x[i, 0])) /(Math.Pow(sum,2));
            return sPrime;
        }
        #endregion

        #region Batch-Learning loop
        public void FeedForward(Matrix input)
        {
            WeightedInput = new List<Matrix> { Weights[0] * input + Biases[0] };
            Activity = new List<Matrix> { input, Matrix.Function(WeightedInput[0], Activation) };

            for (int i = 1; i < Weights.Count - 1; ++i)
            {
                WeightedInput.Add(Weights[i] * Activity[i] + Biases[i]);
                Activity.Add(Matrix.Function(WeightedInput[i], Activation));
            }

            WeightedInput.Add(Weights[Weights.Count - 1] * Activity[Weights.Count - 1] + Biases[Weights.Count - 1]);
            Activity.Add(new Matrix(SoftMax(WeightedInput[Weights.Count - 1].GetTable())));
        }

        private void Backpropagate(Matrix input, Matrix desiredOutput)
        {
            Activity.Clear();
            WeightedInput.Clear();
            FeedForward(input);

            DeltaMatrices = new List<Matrix> { Matrix.Schur((Activity[Activity.Count - 1] - desiredOutput), new Matrix(SoftMaxPrime(WeightedInput[WeightedInput.Count - 1].GetTable()))) };
            for (int i = 1; i < Weights.Count; ++i)
            {
                DeltaMatrices.Add(Matrix.Schur(Matrix.Transpose(Weights[Weights.Count - i]) * DeltaMatrices[i - 1], Matrix.Function(WeightedInput[WeightedInput.Count - i - 1], ActivationPrime)));
            }
            DeltaMatrices.Reverse();


            Matrix gradient;
            double[,] table;
            Gradients = new List<Matrix>();

            for (int i = 0; i < Weights.Count; ++i)
            {
                table = new double[Weights[i].M, Weights[i].N];
                for (int k = 0; k < table.GetLength(1); ++k)
                {
                    for (int j = 0; j < table.GetLength(0); ++j)
                    {
                        table[j, k] = Activity[i][k, 0] * DeltaMatrices[i][j, 0];
                    }
                }
                gradient = new Matrix(table);
                Gradients.Add(gradient);
            }
        }

        private void MiniBatch(List<Matrix> inputs, List<Matrix> outputs)
        {
            Backpropagate(inputs[0], outputs[0]);
            MiniBatchGradients = Gradients.ToList();
            MiniBatchDeltaMatrices = DeltaMatrices.ToList();
            for (int i = 1; i < inputs.Count; ++i)
            {
                Backpropagate(inputs[i], outputs[i]);
                for (int j = 0; j < MiniBatchGradients.Count; ++j)
                {
                    MiniBatchGradients[j] += Gradients[j];
                    MiniBatchDeltaMatrices[j] += DeltaMatrices[j];
                }
            }
        }

        private void UpdateParameters()
        {
            for (int i = 0; i < MiniBatchDeltaMatrices.Count; ++i)
            {
                Biases[i] += LEARNING_CONSTANT_BIASES * MiniBatchDeltaMatrices[i];
                Weights[i] += LEARNING_CONSTANT_WEIGHTS * MiniBatchGradients[i];
            }
        }

        public void Train(List<Matrix> inputs, List<Matrix> outputs, int miniBatchSize)
        {
            List<Matrix> miniBatchInputs = new List<Matrix>();
            List<Matrix> miniBatchOutputs = new List<Matrix>();
            for(int i = 0; i < inputs.Count / miniBatchSize; ++i)
            {
                Console.Write("Training progress: " + Math.Round(100*(float)(i+1) /(inputs.Count / miniBatchSize),0) + "%");
                for (int j = i * miniBatchSize; j < i*miniBatchSize + miniBatchSize; ++j)
                {
                    miniBatchInputs.Add(inputs[j]);
                    miniBatchOutputs.Add(outputs[j]);
                }

                MiniBatch(miniBatchInputs, miniBatchOutputs);
                UpdateParameters();

                MiniBatchDeltaMatrices.Clear();
                MiniBatchGradients.Clear();
                miniBatchOutputs.Clear();
                miniBatchInputs.Clear();
            }
        }
        #endregion

        #region Methods to save to external files
        public override void Save(FileStream fileStream)
        {
            //FileStream fileStream = new FileStream(fileName, FileMode.Create);
            BinaryWriter binaryFileWriter = new BinaryWriter(fileStream);

            binaryFileWriter.Write(Weights.Count);
            SaveList(Weights, ref binaryFileWriter);

            binaryFileWriter.Write(Biases.Count);
            SaveList(Biases, ref binaryFileWriter);

            binaryFileWriter.Close();
            fileStream.Close();
        }
        private void SaveList(List<Matrix> parameters, ref BinaryWriter binaryFileWriter)
        {
            for (int i = 0; i < parameters.Count; ++i)
            {
                binaryFileWriter.Write(parameters[i].M);
                binaryFileWriter.Write(parameters[i].N);
                for (int j = 0; j < parameters[i].M; ++j)
                    for (int k = 0; k < parameters[i].N; ++k)
                        binaryFileWriter.Write(parameters[i][j, k]);
            }
        }

        public static FCN Load(FileStream fileStream)
        {
            BinaryReader binaryFileReader = new BinaryReader(fileStream);

            binaryFileReader.Read(Weights.Count);//read the number of weights
            LoadList(Weights, ref binaryFileReader);//read the weights themselves

            binaryFileReader.Read(Biases.Count);//read the number of biases
            LoadList(Biases, ref binaryFileReader);//read the biases themselves

            binaryFileReader.Close();
            fileStream.Close();
            return null;
        }

        private static void LoadList(List<Matrix> parameters, ref BinaryReader binaryFileReader)
        {

        }
        #endregion

        #region Other methods
        public override string ToString()
        {
            string architecture = Weights[0].N.ToString() + " ";
            foreach (Matrix w in Weights)
                architecture += w.M + " ";
            return "Activation function: " + Activation.Method.Name + "\n" + architecture;
        }

        public override void FeedForward(Matrix[] inputVolume)
        {
            throw new NotImplementedException();
        }

        protected override void BackwardPass(Matrix[] deltaVolume)
        {
            throw new NotImplementedException();
        }

        protected override void ComputeOutputVolumeParameters()
        {
            throw new NotImplementedException();
        }
        #endregion

    }
}
