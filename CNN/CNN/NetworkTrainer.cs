using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NEURAL
{
    class NetworkTrainer
    {
        /// <summary>
        /// 
        /// </summary>
        Vector<double>[] VectorDelta { get; set; }
        /// <summary>
        /// 
        /// </summary>
        Matrix<double>[] Gradiant { get; set; }

        public double LearningRate { get; private set; }
        public FCNeuralNetwork Subject { get; private set; }
        public int NbLayer{get{ return Subject.NbLayer; } }

        public NetworkTrainer(FCNeuralNetwork subject, double learningRate = 0.01)
        {
            Subject = subject;
            LearningRate = learningRate;

        }

        public void TrainSubject(List<double[]> trainingInputs, List<double[]> trainingOutput,int batchSize)
        {
            if (Subject == null)
                return;
            //FAIRE VERIFICATION TAILLLE INPUTS/OUTPUTS
            int trainingDataSize = trainingInputs.Count;
            for (int i = 0; i < trainingDataSize-batchSize; i+=batchSize)
            {
                Console.WriteLine(i+"/"+trainingDataSize);
                MiniBatchTrain(trainingInputs, trainingOutput, i,i+batchSize);
                VectorDelta = null;
                Gradiant = null;
            }

        }
        private void MiniBatchTrain(List<double[]> trainingInputs, List<double[]> trainingOutput, int batchBegin,int batchEnd)
        {
            var b = Vector<double>.Build;
            for (int x = batchBegin; x < batchEnd; x++)
            {
                BackPropagate(b.DenseOfArray(trainingInputs[x]), b.DenseOfArray(trainingOutput[x]));
            }
            UpdateValues();
        }

        public void TrainSubject(List<Vector<double>> trainingInputs, List<Vector<double>> trainingOutput, int batchSize)
        {
            if (Subject == null)
                return;
            //FAIRE VERIFICATION TAILLLE INPUTS/OUTPUTS
            int trainingDataSize = trainingInputs.Count;
            Console.WriteLine("Training of {0} training data initiated",trainingDataSize);
            for (int i = 0; i < trainingDataSize - batchSize; i += batchSize)
            {
                VectorDelta = null;
                Gradiant = null;
                //Console.WriteLine(i + "/" + trainingDataSize);
                Console.Write("\rTraining complété à: :{0}%   ", Math.Round((100*(float)i)/trainingDataSize,2));
                MiniBatchTrain(trainingInputs, trainingOutput, i, i + batchSize);
            }

        }
        private void MiniBatchTrain(List<Vector<double>> trainingInputs, List<Vector<double>> trainingOutput, int batchBegin, int batchEnd)
        {
            var b = Vector<double>.Build;
            for (int x = batchBegin; x < batchEnd; x++)
            {
                BackPropagate(trainingInputs[x], trainingOutput[x]);
            }
            UpdateValues();
        }

        private void BackPropagate(Vector<double> input, Vector<double> output)
        {
            Subject.FeedForward(input);
            Vector<double>[] deltatemp = new Vector<double>[NbLayer];
            Matrix<double>[] gradiantTemp = new Matrix<double>[NbLayer];

            //difference entre desired output et obtained output
            deltatemp[NbLayer - 1] = Vector<double>.op_DotMultiply((Subject.ActivationValues[NbLayer-1] - output),
                Subject.ActivationEquations[NbLayer-1].DerivativeOperation(Subject.WeightedInput[NbLayer-1]));
            //On calcule le reste des erreurs pour les deltas
            for (int l = NbLayer-2; l>=0; --l)
            {
                deltatemp[l] = Vector<double>.op_DotMultiply(Subject.Weights[l+1].TransposeThisAndMultiply(deltatemp[l+1]) ,
                    Subject.ActivationEquations[l].DerivativeOperation( Subject.WeightedInput[l]) );
            }
            //On calcule les gradiants pour chaque weights en commencant par la premiere matrices en utilisant les inputs comme
            //remplacement a la couche d'activation
            var builder = Matrix<double>.Build;
            gradiantTemp[0] = (builder.Dense(Subject.Weights[0].RowCount, Subject.Weights[0].ColumnCount,
                   (r, c) => input[c] * deltatemp[0][r]));
            for (int i = 1; i < NbLayer; ++i)//FOREACH WEIGHT MATRICES
            {
                gradiantTemp[i]=(builder.Dense(Subject.Weights[i].RowCount, Subject.Weights[i].ColumnCount,
                    (r, c) => Subject.ActivationValues[i-1][c] * deltatemp[i][r]));
            }
            //Si premiere fois
            if(VectorDelta==null)
            {
                VectorDelta = deltatemp;
                Gradiant = gradiantTemp;
            }
            else//sinon
            { //On additionne les nouveaux gradiants aux anciens
                for (int i = 0; i < NbLayer; i++)
                {
                    VectorDelta[i] += deltatemp[i];
                    Gradiant[i] += gradiantTemp[i];
                }
            }
        }
        void UpdateValues()
        {
            //pour toutes les donnes dans un minibatch, on ajuste les connexions selon l'erreur
            for (int i = 0; i < VectorDelta.Length; ++i)
            {
                Subject.Biais[i] -= LearningRate * VectorDelta[i];
                Subject.Weights[i] -= LearningRate * Gradiant[i];
            }
        }
    }
}
