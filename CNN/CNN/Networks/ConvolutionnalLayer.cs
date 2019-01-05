using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace CNN.Networks
{
    class ConvolutionnalLayer
    {
        /// <summary>
        /// Filter(Kernels) qui seront applique sur les matrices d'inputs. Filtres[i][j] où
        ///i represente un filtre en particulier et j la profondeur du filtre.
        ///Les matrices contiennent les connexions applique sur chaque donne de l'input
        /// </summary>
        Matrix<double>[][] Filters { get; set; }
        /// <summary>
        /// Biaises assoicie pour les filtres. Un biais par filtre.
        /// </summary>
        double[] Biases { get; set; }
        public int Stride { get; private set; }
        public int FilterSize { get; private set; }
        public int HeightSizePadding { get; private set; }
        public int WidthSizePadding { get; private set; }

        //Servent pour dimension des inputs
        public int NbRowInput { get; private set; }
        public int NbColumnsInput { get; private set; }
        public int Depth { get; private set; }

        public int DimensionOfFilter { get; private set; }

        public int NbFilters{ get { return Filters.Length; } }
        public Tuple<int, int> getFilterSize(int depth, int id)
        {
            if (!(depth < NbFilters))
                return null;
            if (!(id < Filters[depth].Length))
                return null;
            return new Tuple<int, int>(Filters[depth][id].RowCount, Filters[depth][id].ColumnCount);
        }
        public ConvolutionnalLayer(FileStream fileReader)
        {

        }

        public ConvolutionnalLayer(int inputsNbRows,int inputsNbColumns,int depthInput,int nbFilters,int dimensionFilter,int stride=1)
        {
            NbRowInput = inputsNbRows;
            NbColumnsInput = inputsNbColumns;
            Depth = depthInput;
            Stride = stride;
            DimensionOfFilter = dimensionFilter;
            WidthSizePadding = DimensionOfFilter - Stride;
            InstancierFilters(nbFilters);
            InstancierBiases();
            SetPaddingSize();
        }

        private void InstancierFilters(int nbFilters)
        {
            Filters = new Matrix<double>[nbFilters][];

            var matrixBuilder = Matrix<double>.Build;
            //Pour chaque filtre que l'on doit creer
            for (int i = 0; i < nbFilters; i++)
            {
                //On créer le tableau de matrices où la taille du tableau est la profondeur
                Filters[i] = new Matrix<double>[Depth];
                //Pour chaque profondeur (donc l'etage d'un filtre), on crée une matrice de bonne taille où ses valeurs sont initialiser aleatoirement
                for (int j = 0; j <Depth; j++)
                    Filters[i][j] = matrixBuilder.Dense(DimensionOfFilter, DimensionOfFilter,
                        (x,y)=> RandomGenerator.GenerateRandomDouble(-1,1));
            }
        }

        private void InstancierBiases()
        {
            Biases = new double[NbFilters];
        }
        private void SetPaddingSize()
        {
            WidthSizePadding = 0;
            for(int i=0;true;i++) //W−F + 2P
                if ((NbColumnsInput-FilterSize+2*i)%Stride==0)
                {
                    WidthSizePadding = i;
                    break;
                }
            HeightSizePadding = 0;
            for (int i = 0; true; i++) //W−F + 2P
                if ((NbRowInput - FilterSize + 2 * i) % Stride == 0)
                {
                    HeightSizePadding = i;
                    break;
                }
        }

        #region Old FeedForward
        #endregion

        public Matrix<double>[] FeedForward(Matrix<double> input)
        {
            //On cree la matrice avec la padding
            Matrix<double> bonneMatrice = Matrix<double>.Build.Dense(NbRowInput + 2 * HeightSizePadding, NbColumnsInput + 2 * WidthSizePadding, 0);
            //On set les valeurs
            bonneMatrice.SetSubMatrix(HeightSizePadding, WidthSizePadding, input);


            //double[,] output = new double[bonneMatrice.RowCount, bonneMatrice.ColumnCount];
            Matrix<double>[] outputVolume = new Matrix<double>[NbFilters];
           
            double result;
            for (int l = 0; l < NbFilters; ++l)//Foreach kernel...
            {
                Matrix<double> output = Matrix<double>.Build.Dense(bonneMatrice.RowCount, bonneMatrice.ColumnCount);
                for (int i = 0; i*Stride < bonneMatrice.RowCount; ++i)//On the height...
                {
                    for (int j = 0; j*Stride < bonneMatrice.ColumnCount; ++j)//On the width...
                    {
                        result = 0;
                        for (int k = 0; k < Depth; ++k)//On the depth...
                        {
                            result += Matrix<double>.op_DotMultiply(bonneMatrice.SubMatrix(i * Stride, j * Stride, FilterSize, FilterSize),
                                Filters[l][k]).ColumnSums().Sum();
                        }
                        output[i, j] = result + Biases[l];
                    }
                }
                outputVolume[l] = output;
            }

            return outputVolume;
        }
    }
}
