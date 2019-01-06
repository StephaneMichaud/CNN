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
        Matrix<double> Filters { get; set; }

        Matrix<double> LastInputs { get; set; }
        /// <summary>
        /// Biaises assoicie pour les filtres. Un biais par filtre.
        /// </summary>
        double[] Biases { get; set; }
        public int Stride { get; private set; }
        public int NbRowPadding { get; private set; }
        public int NbColumnsPadding { get; private set; }

        //Servent pour dimension des inputs
        public int NbRowInput { get; private set; }
        public int NbColumnsInput { get; private set; }
        public int NbRowOutput { get; private set; }
        public int NbColumnsOutput { get; private set; }
        public int Depth { get; private set; }

        public int DimensionOfFilter { get; private set; }

        public int NbFilters{ get { return Filters.RowCount; } }

        public ConvolutionnalLayer(FileStream fileReader)
        {

        }

        public ConvolutionnalLayer(int inputsNbRows,int inputsNbColumns,int depthInput,int nbFilters,int dimensionOfFilters)
        {
            NbRowInput = inputsNbRows;
            NbColumnsInput = inputsNbColumns;
            Depth = depthInput;
            Stride = 1;
            DimensionOfFilter = dimensionOfFilters;
            InstancierFilters(nbFilters);
            InstancierBiases();
            // SetPaddingSize();
            NbRowPadding = (DimensionOfFilter - 1) / 2;
            NbColumnsPadding = (DimensionOfFilter - 1) / 2;

            NbRowOutput = ((NbRowInput+2*NbRowPadding - DimensionOfFilter) / Stride) + 1;
            NbColumnsOutput = ((NbColumnsInput + 2 * NbColumnsPadding - DimensionOfFilter) / Stride) + 1;

        }

        private void InstancierFilters(int nbFilters)
        {
            int i = 0;
            Filters = Matrix<double>.Build.Dense(nbFilters,DimensionOfFilter*DimensionOfFilter*Depth,
                (x,y)=> RandomGenerator.GenerateRandomDouble(-1, 1));
        }

        private void InstancierBiases()
        {
            Biases = new double[NbFilters];
        }

        public Matrix<double>[] FeedForward(Matrix<double>[] input)
        {
            //On cree la matrice avec la padding
            Matrix<double>[] image = new Matrix<double>[Depth];
            Matrix<double>[] outputVolume = new Matrix<double>[NbFilters];

            for (int d = 0; d < Depth; ++d)
            { 
                image[d] = Matrix<double>.Build.Dense(NbRowInput + 2 * NbRowPadding, NbColumnsInput + 2 * NbColumnsPadding, 0);
                //On set les valeurs
                image[d].SetSubMatrix(NbRowPadding, NbColumnsPadding, input[d]);
            }
            FillLastInputs(image);

            Matrix<double> result= Filters*LastInputs;
            
            for (int i = 0; i < NbFilters; ++i)
            {
                outputVolume[i] = RowToMatrix(result.Row(i), i);
            }
            
            return outputVolume;
        }
        private Matrix<double> RowToMatrix(Vector<double> row,int index)
        {//x = 
            return Matrix<double>.Build.Dense(NbRowOutput,NbColumnsOutput,(x,y)=>Math.Max(row[y*NbRowOutput+x]+Biases[index],0));
        }

        private void FillLastInputs(Matrix<double>[] image)
        {
            //On instancie LastInputs sous la forme d'une matrice de colonnes ou chque colonnes seront un vecteur
            //representant les valeurs d'une sous section de toutes les profondeurs une apres l'autre
            LastInputs = Matrix<double>.Build.Dense(DimensionOfFilter * DimensionOfFilter * Depth, NbRowOutput * NbColumnsOutput);
            //Image sous forme d'un vecteur

            int cmptStrideHor = 0;
            int cmptStrideVertical = 0;
            for (int col = 0; col < LastInputs.ColumnCount; ++col)//Remplir le last input avec les bonnes valeurs
            {
                Vector<double> colTemp = Vector<double>.Build.Dense(DimensionOfFilter * DimensionOfFilter * Depth);
                for (int d = 0; d < Depth; ++d)
                {
                    colTemp.SetSubVector(d * DimensionOfFilter * DimensionOfFilter, DimensionOfFilter * DimensionOfFilter,
                        Vector<double>.Build.DenseOfArray(image[d].SubMatrix(cmptStrideVertical * Stride, DimensionOfFilter, cmptStrideHor * Stride, DimensionOfFilter).ToColumnMajorArray()));
                }
                //On met la colonne dans lastInputs
                LastInputs.SetColumn(col, colTemp);
                cmptStrideHor++;
                if (cmptStrideHor == (((image[0].ColumnCount - DimensionOfFilter) / Stride) + 1))
                {
                    cmptStrideHor = 0;
                    cmptStrideVertical++;
                }
            }
        }

        
        #region Old FeedForward
        #endregion

        /*public Matrix<double>[] FeedForward(Matrix<double> input)
        {
            //On cree la matrice avec la padding
            Matrix<double> bonneMatrice = Matrix<double>.Build.Dense(NbRowInput + 2 * NbRowPaddingPadding, NbColumnsInput + 2 * NbColumnsPadding, 0);
            //On set les valeurs
            bonneMatrice.SetSubMatrix(NbRowPaddingPadding, NbColumnsPadding, input);


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
                            result += Matrix<double>.op_DotMultiply(bonneMatrice.SubMatrix(i * Stride, j * Stride, DimensionOfFilter, DimensionOfFilter),
                                Filters[l][k]).ColumnSums().Sum();
                        }
                        output[i, j] = result + Biases[l];
                    }
                }
                outputVolume[l] = output;
            }

            return outputVolume;
        }*/
    }
}
