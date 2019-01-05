using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.IO;

namespace NEURAL
{
    /// <summary>
    /// Class qui devrait etre plus performante pour les neurals networks
    /// </summary>
    class FCNeuralNetwork
    {
        /// <summary>
        /// 
        /// </summary>
        Random RandomGenerateur { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public ActivationEquation[] ActivationEquations { get; private set; }
        /// <summary>
        /// Valeur des neurones apres etre passer dans la fonction d'activation. Notation A l'activation
        /// </summary>
        public Vector<double>[] ActivationValues { get; private set; }

        /// <summary>
        /// Valeurs des neurones avant etre passer dans la fonction d'activation. Notation Z soit le weighted input
        /// </summary>
        public Vector<double>[] WeightedInput { get;private set; }
        /// <summary>
        /// Tableau de matrices contenant les poids des connexions des neurones avec l'etage precedent. La matrice indice 0 
        /// correspond au connexion du premier hidden layer avec les donnes inputs donne au network dans le feedforward
        /// </summary>
        public Matrix<double>[] Weights { get; set; }

        public Vector<double>[] Biais { get; set; }

        /// <summary>
        /// Nombre de valeurs nécéssaire pour compute le network
        /// </summary>
        public int NbNecessaryInputs { get; private set; }

        /// <summary>
        /// Nombre de layer du neuralNetwork. Les inputs ne compte pas.
        /// </summary>
        public int NbLayer { get { return WeightedInput.Length; } }

        


        public FCNeuralNetwork(int nbInputs, int[] structure)
        {
            RandomGenerateur = new Random();
            NbNecessaryInputs = nbInputs;
            InstancierNodes(structure);
            Instancierweights();

        }
        public FCNeuralNetwork(string pathName)
        {
            RandomGenerateur = new Random();
            BinaryReader bsReader = new BinaryReader(new FileStream(pathName, FileMode.Open));

            NbNecessaryInputs = bsReader.ReadInt32();
            int[] structure =new int[bsReader.ReadInt32()];
            for (int i = 0; i < structure.Length; i++)
                structure[i] = bsReader.ReadInt32();

            InstancierNodes(structure);
            Instancierweights();

            //On lis les biais de chaque neurons pour chque layer
            for (int i = 0; i < NbLayer; i++)
                for (int j = 0; j < structure[i]; j++)
                    Biais[i][j] = bsReader.ReadDouble();

            //On lis les poids des connexions de chaque neurons, pour tout les neurons precedent, de chaque layer
            for (int i = 0; i < NbLayer; i++)
                for (int r = 0; r < Weights[i].RowCount; r++)
                    for (int c = 0; c < Weights[i].ColumnCount; c++)
                        Weights[i][r, c] = bsReader.ReadDouble();

            bsReader.Close();
        }

        public void InstancierNodes(int[] structure)
        {
            //On instancie les tableau de vecteurs
            ActivationValues = new Vector<double>[structure.Length];
            WeightedInput = new Vector<double>[structure.Length];
            Biais = new Vector<double>[structure.Length];
            ActivationEquations = new ActivationEquation[structure.Length];


            //On definie la taille des vecteurs et on les remplis de 0
            ReLuActivation tempActivationFunction = new ReLuActivation();
            for (int i = 0; i < structure.Length; i++)
            {
                ActivationEquations[i] = tempActivationFunction;
                WeightedInput[i] = Vector<double>.Build.Dense(structure[i], 0);                                
                ActivationValues[i] = Vector<double>.Build.Dense(structure[i], 0);
                //Instancie les biais avec des valeurs entre 0 et -1
                Biais[i] = Vector<double>.Build.Dense(structure[i], v=>0/*(2 * RandomGenerateur.NextDouble() - 1)/100*/);
            }
            ActivationEquations[NbLayer-1] = new SoftMaxActivation();
        }
        public void Instancierweights()
        {
            Weights = new Matrix<double>[NbLayer];

            //Forme des matrices de weights ou chaque ligne correpond au connexion d'un neurone. 
            //On multiplie cette matrices avec un vecteur de donnes pour avoir les valeurs de l'etage suivant
            
            //Creer la premiere matrice de weights. Valeurs sont instancier entre -1 et 1
            Weights[0] = Matrix<double>.Build.Dense(WeightedInput[0].Count, NbNecessaryInputs, (row, col) => getGaussianNumber(0,Math.Sqrt(1.0/(NbNecessaryInputs)))  /* Math.Sqrt(2/NbNecessaryInputs)/*getRandomNegOrPos()*(Math.Sqrt (12.0/(0+ActivationValues[0].Count)))*/);
            //Creer le reste des matrices de weights selon le nombre de neurones dans l'etage correspondant
            for (int i = 1; i < NbLayer; i++)
            {
                Weights[i] = Matrix<double>.Build.Dense(WeightedInput[i].Count, ActivationValues[i-1].Count, (row, col) => getGaussianNumber(0, Math.Sqrt(1.0/ (ActivationValues[i - 1].Count)))/* Math.Sqrt(2 / (ActivationValues[i-1].Count))/*(2 * RandomGenerateur.NextDouble() - 1) / 100/* getRandomNegOrPos() * (Math.Sqrt(12.0 / (ActivationValues[i-1].Count + ActivationValues[i].Count)))*/);
            }
        }
        double getGaussianNumber(double mean, double stdDev)
        {
            double u1 = 1.0 - RandomGenerateur.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - RandomGenerateur.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal =
                         mean + stdDev * randStdNormal;
            return randNormal;
        }

        public double[] FeedForward(double[] inputs)
        {
            //EST CE QUE CEST EFFICACE DE TJRS CONVERTIR LES TABLEAU DE DOUBLE EN MATRICES?

            if (inputs.Length != NbNecessaryInputs)
                throw new Exception("Le nombre de inputs entre n'est pas valide");
            //A CHANGER PEUT ETRE
            //On calcule les valeurs du premier layer selon les inputs donnes avec une equation matriciel
            WeightedInput[0] = Weights[0] * Vector<double>.Build.DenseOfArray(inputs) + Biais[0];
            ActivationValues[0] = ActivationEquations[0].Operation( WeightedInput[0]);

            //On calculs les valeurs selon l'etage precedent avec equations matricielles
            for (int layer = 1; layer < NbLayer; layer++)
            {
                WeightedInput[layer] = (Weights[layer] * ActivationValues[layer-1]) + Biais[layer];
                //A CHECKER SI ON POURRAIT OPTIMISER, pas sure si build dense est efficace, checker autre methode
                //Faire verification pour dernier layer si RELU pour utiliser softmax
                ActivationValues[layer] = ActivationEquations[layer].Operation(WeightedInput[layer]);
            }
            //On retourne les valeurs d'activation du dernier layer
            return ActivationValues[NbLayer-1].ToArray();
        }
        public double[] FeedForward(Vector<double> inputs)
        {
            //EST CE QUE CEST EFFICACE DE TJRS CONVERTIR LES TABLEAU DE DOUBLE EN MATRICES?

            if (inputs.Count != NbNecessaryInputs)
                throw new Exception("Le nombre de inputs entre n'est pas valide");
            //A CHANGER PEUT ETRE
            //On calcule les valeurs du premier layer selon les inputs donnes avec une equation matriciel
            WeightedInput[0] = (Weights[0] * inputs) + Biais[0];
            ActivationValues[0] = ActivationEquations[0].Operation( WeightedInput[0]);

            //On calculs les valeurs selon l'etage precedent avec equations matricielles
            for (int layer = 1; layer < NbLayer; layer++)
            {
                WeightedInput[layer] = (Weights[layer] * ActivationValues[layer - 1]) + Biais[layer];
                //A CHECKER SI ON POURRAIT OPTIMISER, pas sure si build dense est efficace, checker autre methode
                //Faire verification pour dernier layer si RELU pour utiliser softmax
                ActivationValues[layer] = ActivationEquations[layer].Operation(WeightedInput[layer]);
            }
            //On retourne les valeurs d'activation du dernier layer
            return ActivationValues[NbLayer - 1].ToArray();
        }

        /// <summary>
        /// Permet de sauvegarder la valeurs du neural network dans un fichier binaire
        /// </summary>
        /// <param name="pathName">Chemin et nom du fichier qui sera creer</param>
        public void SaveAsBinaryFile(string pathName)
        {
            BinaryWriter bsWriter = new BinaryWriter(new FileStream(pathName, FileMode.Create));

            bsWriter.Write(NbNecessaryInputs);
            bsWriter.Write(WeightedInput.Length);
            for (int i = 0; i < WeightedInput.Length; i++)
            {
                bsWriter.Write(WeightedInput[i].Count);
            }

            for (int i = 0; i < NbLayer; i++)
            {
                for (int b = 0; b < Biais[i].Count; b++)
                {
                    bsWriter.Write(Biais[i][b]);
                }
            }

            for (int i = 0; i < Weights.Length; i++)
            {
                for (int r = 0; r < Weights[i].RowCount; r++)
                {
                    for (int c = 0; c < Weights[i].ColumnCount; c++)
                    {
                        bsWriter.Write(Weights[i][r,c]);
                    }
                }
            }
            bsWriter.Close();
        }

    }

    #region EquationActivation

    public abstract class ActivationEquation
    {
        public abstract Vector<double> Operation(Vector<double> input);
        public abstract Vector<double> DerivativeOperation(Vector<double> input);
    }

    public class SigmoidActivation:ActivationEquation
    {

        public override Vector<double> Operation(Vector<double> input)
        {
            Vector<double> output = input.Clone();
            for (int i = 0; i < output.Count; i++)
            {
                output[i] = SingleSigmoid(input[i]);
            }
            return output;
        }
        static double SingleSigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public override Vector<double> DerivativeOperation(Vector<double> input)
        {
            Vector<double> output = input.Clone();
            for (int i = 0; i < output.Count; i++)
            {
                output[i] = SingleSigmoid(input[i]) * (1 - SingleSigmoid(input[i]));
            }
            return output;
        }

    }
    public class HyperbolicTanActivation 
    {
        public double DerivativeOperation(double x)
        {
            return 1 - Math.Pow(Math.Tanh(x), 2);
        }

        public double Operation(double x)
        {
            return Math.Tanh(x);
        }

    }
    public class ReLuActivation: ActivationEquation
    {
        public override Vector<double> DerivativeOperation(Vector<double> input)
        {
            Vector<double> output = Vector<double>.Build.Dense(input.Count);
            for (int i = 0; i < output.Count; i++)
            {
                output[i] = input[i] > 0 ? 1 : 0;

            }
            return output;
        }

        public override Vector<double> Operation(Vector<double> input)
        {
            Vector<double> output = Vector<double>.Build.Dense(input.Count);
            for (int i = 0; i < output.Count; i++)
            {
                output[i] = Math.Max(0.001, input[i]);
                
            }
            return output;
        }

    }
    public class SoftMaxActivation:ActivationEquation
    {
        public override Vector<double> Operation(Vector<double> input)
        {
            Vector<double> output = input.Clone();
            double somme = 0;
            for (int i = 0; i < output.Count; i++)
            {
                output[i] = Math.Exp(input[i]);
                somme += output[i];
            }
            output /= somme;
            return output;
        }
        public override Vector<double> DerivativeOperation(Vector<double> input)
        {
            double somme = 0;
            Vector<double> sPrime = Vector<double>.Build.Dense(input.Count);
            for (int j = 0; j < sPrime.Count; ++j)
                somme += Math.Exp(input[j]);
            double sommeCarre = Math.Pow(somme, 2);
            for (int i = 0; i < input.Count; i++)
            {
                sPrime[i] = (Math.Exp(input[i]) * somme - Math.Pow(Math.Exp(input[i]),2))
                    / sommeCarre;
            }
            return sPrime;
        }

    }
    #endregion
}
