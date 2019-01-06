using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace CNN.Networks
{
    /// <summary>
    /// 
    /// </summary>
    class ConvolutionnalLayer
    {

        /// <summary>
        /// Filtres qui seront iterer sur les inputs donnes. Sont sous la forme tel que Filtres[i][j]
        /// où i(row) représente un filtre et j(col) une valeur d'une connexion dans ce filtre. Les valeurs du filtre(j) sont en ordre 
        /// de profondeur tel que si un filtre a une profondeur double, les données sont mis dans l'ordre:
        /// j11,j12,j13,j14...pour la premiere profondeur suivi par j21,j22,j23,j24... qui sont les valeurs
        /// de la deuxième profondeur. Cette matrice sera multiplié avec la matrice LastInputs tel que Filters*LastInputs
        /// afin de calculer une matrice contenant tout les weighted values des outputs.
        /// </summary>
        Matrix<double> Filters { get; set; }
        /// <summary>
        /// Valeurs de l'image la plus récente où les sous sections de l'image visité par les filtres ont été mis sous forme de vecteurs.
        /// Ces vecteur colonnes contiennes les informations des inputs de la sous-section tel que LastInputs[i][j] ou j(col) represente une 
        /// sous-section particuliere et i(row) une donné dans la sous-section. Les données de sous-section sont arrangées en ordre de profondeur 
        /// tels que: i11,i12,i12...i21,i22,i23...où le premier index de i correspond à la profondeur et le deuxième à une donnée dans la sous-section. 
        /// Les données d'une profondeur sont mis en ordre top-bottom donc les élément d'une colonne une après l'autre. Cette matrice sera multiplié 
        /// avec la matrice Filter tel que :Filter*LastInputs afin de calculer une matrice contenant tout les weighted values des outputs. LastInputs
        /// est gardé en mémoire afin de l'utiliser lors du backpropagate.
        /// </summary>
        Matrix<double> LastInputs { get; set; }
        /// <summary>
        /// Biaises assoicie pour les filtres. Un biais par filtre. Chaque biais[i] est associé au filtre[i][]
        /// </summary>
        double[] Biases { get; set; }
        /// <summary>
        /// Représente de combien d'espace sera bouger le filtre sur l'image entre chaque itération. Est normalement de valeur 1 ou 2.
        /// </summary>
        public int Stride { get; private set; }
        /// <summary>
        /// Nombre de lignes(row) de padding qui seront ajoute d'un coté de la matrice des inputs.(P)
        /// </summary>
        public int NbRowPadding { get; private set; }
        /// <summary>
        /// Nombre de colonnes(columns) de padding qui seront ajoute d'un coté de la matrice des inputs.(P)
        /// </summary>
        public int NbColumnsPadding { get; private set; }


        //Servent pour dimension des inputs
        /// <summary>
        /// Taille en y de volume de l'input. Sera le nombre de lignes des differentes matrices de profondeurs
        /// </summary>
        public int NbRowInput { get; private set; }
        /// <summary>
        /// Taille en x du volume de l'input. Sera le nombre de colonnes des differentes matrices de profondeurs
        /// </summary>
        public int NbColumnsInput { get; private set; }
        /// <summary>
        /// Taille en z du volume de l'input. Sera le nombre de matrices de profondeurs.
        /// </summary>
        public int Depth { get; private set; }
        /// <summary>
        /// Le nombre de rangées dans le volume de sortie.
        /// </summary>
        public int NbRowOutput { get; private set; }
        /// <summary>
        /// Le nombre de colonnes dans le volume de sortie.
        /// </summary>
        public int NbColumnsOutput { get; private set; }
        /// <summary>
        /// Nombre de filtres individuels dans cette couche. Correspond aussi abstract a la taille en Z du output volume.
        /// </summary>
        public int NbFilters{ get { return Filters.RowCount; } }
        /// <summary>
        /// Dimensions (carrée) de tout les filtres appartenant à cette couche.
        /// </summary>
        public int DimensionOfFilter { get; private set; }


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

            //Apres ce point, l'ordre des ces affectations est important
            InstancierFilters(nbFilters);
            InstancierBiases();

            //Instancie le padding
            NbRowPadding = (DimensionOfFilter - 1) / 2;
            NbColumnsPadding = (DimensionOfFilter - 1) / 2;

            //Instancie les valeurs de la taille du volume de l'output afin d'éviter a tout le temps les recalculés.
            NbRowOutput = ((NbRowInput + 2*NbRowPadding - DimensionOfFilter) / Stride) + 1;
            NbColumnsOutput = ((NbColumnsInput + 2*NbColumnsPadding - DimensionOfFilter) / Stride) + 1;

        }
        /// <summary>
        /// Instancie tout les filtres de cette couche avec leur nombre. Leur dimension est obtenue à l'aide d'un attribut
        /// </summary>
        /// <param name="nbFilters">Nombre de filtre devant être créés</param>
        private void InstancierFilters(int nbFilters)
        {
            Filters = Matrix<double>.Build.Dense(nbFilters,DimensionOfFilter*DimensionOfFilter*Depth,
                (x,y)=> RandomGenerator.GenerateRandomDouble(-1, 1));
        }
        /// <summary>
        /// Instancie tout les biais de cette couche avec le nombre de filtres. Leur valeurs sont initialement 0.
        /// </summary>
        private void InstancierBiases()
        {
            Biases = new double[NbFilters];
        }
        /// <summary>
        /// Prends le volume d'entrée (généralement une image) et applique les convolutions des filtres, les biais et la fonction d'activation (ReLu)
        /// puis retourne le volume de sortie.
        /// </summary>
        /// <param name="input">Volume d'entré. Doit correspondre aux dimensions données au constructeur</param>
        /// <returns>Le volume de sortie.</returns>
        public Matrix<double>[] FeedForward(Matrix<double>[] input)
        {
            //FAIRE VÉRIFICATION DES DIMENSIONS DE INPUT!!!!!!!!!!!!

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
            //LA MAGIE SE PASSE SUR CETTE LIGNE IMPORTANT!!!!!
            Matrix<double> result= Filters*LastInputs;
            //On remet nos outputs selon un tableau de matrices (ptete mettre dans fonction)
            for (int i = 0; i < NbFilters; ++i)
            {
                outputVolume[i] = RowToMatrix(result.Row(i), i);
            }
            return outputVolume;
        }
        /// <summary>
        /// Prend un vecteur ligne du résultat de la multiplication matricielle Filters*LastInputs et le tranforme en matrice 2d
        ///tout en appliquant les biais et la fonction d'acttivation (Relu) aux valeurs de cette matrice.
        /// </summary>
        /// <param name="row">Vecteur ligne qui sera transformé en matrice 2D pour le volume de sortie</param>
        /// <param name="index">Index utilisé pour additioné le biais correspondant aux donnés de la matrice 2D.</param>
        /// <returns></returns>
        private Matrix<double> RowToMatrix(Vector<double> row,int index)
        {
            return Matrix<double>.Build.Dense(NbRowOutput,NbColumnsOutput,(x,y)=>Math.Max(row[y*NbRowOutput+x]+Biases[index],0));
        }
        /// <summary>
        /// Pour faire l'application des convolutions, l'implémentation utilise deux matrices construites à partir de l'image et des filtres.
        /// La matrice LastInputs correspond à 'écraser' chacune des sections (endroits où sont appliqués les filtres) en colonnes.
        /// </summary>
        /// <param name="image">Le volume d'entré avec padding ajouté.</param>
        private void FillLastInputs(Matrix<double>[] image)
        {
            //On instancie LastInputs sous la forme d'une matrice de colonnes ou chaque colonnes seront un vecteur
            //representant les valeurs d'une sous section de toutes les profondeurs une apres l'autre
            //Est 4D lol, imagine toi un e/s\pace 4D bro lol
            //HAS SCIENCE GONE TOO FAR/?!/!  \
            //ILLUMINAUGHTRY CUN=TFIRMD?/ ''' \ 
            //                         /   0   \
            //                        /_________\
            LastInputs = Matrix<double>.Build.Dense(DimensionOfFilter * DimensionOfFilter * Depth, NbRowOutput * NbColumnsOutput);
            //Image sous forme d'un vecteur
            //Compteur utilisé pour savoir dans quelle sous-section de l'image nous sommes.
            int cmptStrideHor = 0;
            int cmptStrideVertical = 0;
            for (int col = 0; col < LastInputs.ColumnCount; ++col)//Remplir le last input avec les bonnes valeurs
            {
                Vector<double> colTemp = Vector<double>.Build.Dense(DimensionOfFilter * DimensionOfFilter * Depth);
                for (int d = 0; d < Depth; ++d)//remplir la colonne temporaire avec une sous-section de l'image 
                {
                    //Chaque dimension de profondeur sont ajouté l'une après l'autre dans le vecteur colonne.
                    colTemp.SetSubVector(d * DimensionOfFilter * DimensionOfFilter, DimensionOfFilter * DimensionOfFilter,
                        Vector<double>.Build.DenseOfArray(image[d].SubMatrix(cmptStrideVertical * Stride, DimensionOfFilter, cmptStrideHor * Stride, DimensionOfFilter).ToColumnMajorArray()));
                }
                //On met la colonne dans lastInputs
                LastInputs.SetColumn(col, colTemp);
                //On incrémente les compteurs de sous-sections
                cmptStrideHor++;
                //Si on arrive a la fin d'une itération horizontale, on descend et revient au début a la gauche.
                if (cmptStrideHor == NbColumnsOutput)
                {
                    cmptStrideHor = 0;
                    cmptStrideVertical++;
                }
            }
        }
    }
}
