using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN
{
    /// <summary>
    /// This class will be created with the philosophy of a reuse in future projects in ML, numerical analysis
    /// or even mathematics.
    /// </summary>
    public class Matrix
    {
        #region Basics.
        private double[,] Table { get; set; }
        public int M { get; private set; }
        public int N { get; private set; }
        public Matrix(double[,] table)
        {
            Table = table;
            M = table.GetLength(0);
            N = table.GetLength(1);
        }
        public Matrix(double[,] table, Random generator)
        {
            Table = table;
            M = table.GetLength(0);
            N = table.GetLength(1);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    Table[i, j] = (double)generator.Next(-100,101)/10000f;
        }
        #endregion

        #region Operators.
        /// <summary>
        /// The inverse of the matrix. If it D.N.E., returns null.
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public static Matrix operator!(Matrix A)
        {
            return SubMatrix(GaussJordan(Augment(A, Identity(A.N))),0,A.N,A.N,A.N);
        }
        public static Matrix operator *(Matrix A, Matrix B)
        {
            double[,] table = new double[A.M,B.N];
            for (int i = 0; i < B.N; ++i)
                for (int j = 0; j < A.M; ++j)
                    table[j, i] = Transpose(SubMatrix(A,j,0,1,A.N)) % SubMatrix(B,0,i,B.M,1);
            return new Matrix(table);
        }
        public static Matrix operator *(double c, Matrix A)
        {
            double[,] table = A.GetTable();
            for (int i = 0; i < A.M; ++i)
                for (int j = 0; j < A.N; ++j)
                    table[i,j] *= c;
            return new Matrix(table);
        }
        public static Matrix operator *(Matrix A, double c)
        {
            return c * A;
        }
        public static Matrix operator +(Matrix A, Matrix B)
        {
            double[,] table = A.GetTable();
            for (int i = 0; i < A.M; ++i)
                for (int j = 0; j < A.N; ++j)
                    table[i, j] = A[i, j] + B[i, j];
            return new Matrix(table);
        }
        public static Matrix operator -(Matrix A, Matrix B)
        {
            return A + -1 * B;
        }
        public static Matrix operator ^(Matrix A, int c)
        {
            for (int i = 0; i < c - 1; ++i)
                A *= A;
            return A;
        }

        public static double operator %(Matrix A, Matrix B)
        {
            double dotProd = 0;
            A = Matrix.Schur(A, B);
            for (int i = 0; i < A.M; ++i)
                for(int j = 0; j < A.N; ++j)
                    dotProd += A[i, j];
            return dotProd;
        }

        //public static double operator %(Matrix A, Matrix B)
        //{
        //    double dotProd = 0;
        //    for (int i = 0; i < A.M; ++i)
        //        dotProd += A[i,0] * B[i,0];
        //    return dotProd;
        //}
        #endregion

        #region Complex functions.
        /// <summary>
        /// The determinant of A.
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public static double Determinant(Matrix A)
        {
            int nbLineSwitches = 0;
            double determinant = 1;
            A = Gauss(A, ref nbLineSwitches);
            for(int i = 0; i < A.N; ++i)
                determinant *= A[i, i];
            if(nbLineSwitches % 2 == 1)
                determinant *= -1;
            return determinant;
        }

        /// <summary>
        /// The LU factorisation of the matrix if it exists.
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public static Matrix[] LU(Matrix A)
        {
            return new Matrix[0];
        }
        /// <summary>
        /// The standard Gaussian elimination with line permutations allowed. Returns the form with maximum pivots.
        /// </summary>
        /// <param name="A">The augmented matrix.</param>
        /// <returns></returns>
        public static Matrix Gauss(Matrix A)
        {
            double coef = 0;
            int nbLineSwitch = 0;
            double[,] table = A.GetTable();

            for (int j = 0; j < A.N; ++j)
            {
                for (int i = j + 1; i < A.M; ++i)
                {
                    while (table[j, j] == 0)
                    {
                        if (nbLineSwitch > A.M - j)
                            return new Matrix(table);
                        table = SwitchLines(j, j + nbLineSwitch++, table);
                    }
                    coef = -table[i, j] / table[j, j];
                    Matrix ligne = coef * SubMatrix(new Matrix(table), j, 0, 1, A.N) + SubMatrix(new Matrix(table), i, 0, 1, A.N);
                    for (int k = 0; k < A.N; ++k)
                        table[i, k] = ligne[0, k];
                }
            }
            return new Matrix(table);
        }
        /// <summary>
        /// The standard Gauss-Jordan elimination. Stops the method and returns null if a pivot is 0.
        /// Submethod to find the inverse.
        /// </summary>
        /// <param name="A">The augmented Matrix.</param>
        /// <returns></returns>
        public static Matrix GaussJordan(Matrix A)
        {
            double coef = 0;
            double[,] table = Gauss(A).GetTable();

            for(int j = 0; j < A.M; ++j)
            {
                for(int i = j - 1; i >= 0; --i)
                {
                    coef = -table[i, j]/table[j,j];
                    Matrix ligne = coef*SubMatrix(new Matrix(table),j,0,1,A.N) + SubMatrix(new Matrix(table),i,0,1,A.N);
                    for (int k = 0; k < A.N; ++k)
                        table[i,k] = ligne[0,k];
                }
            }

            for(int i = 0; i < A.M; ++i)
            {
                coef = table[i, i];
                for(int j = 0; j < A.N; ++j)
                {
                    table[i, j] /= coef;
                }
            }

            return new Matrix(table);
        }
        private static Matrix Gauss(Matrix A, ref int nbLineSwitches)
        {
            double coef = 0;
            double[,] table = A.GetTable();

            for (int j = 0; j < A.N; ++j)
            {
                for (int i = j + 1; i < A.M; ++i)
                {
                    while (table[j, j] == 0)
                    {
                        if (nbLineSwitches > A.M - j)
                            return new Matrix(table);
                        table = SwitchLines(j, j + nbLineSwitches++, table);
                    }
                    coef = -table[i, j] / table[j, j];
                    Matrix ligne = coef * SubMatrix(new Matrix(table), j, 0, 1, A.N) + SubMatrix(new Matrix(table), i, 0, 1, A.N);
                    for (int k = 0; k < A.N; ++k)
                        table[i, k] = ligne[0, k];
                }
            }
            return new Matrix(table);
        }
        /// <summary>
        /// Element wise multiplication. Also known as the Hadamard product. A and B must have the same dimensions.
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static Matrix Schur(Matrix A, Matrix B)
        {
            double[,] table = A.GetTable();
            for (int i = 0; i < A.M; ++i)
                for (int j = 0; j < A.N; ++j)
                    table[i, j] = A[i, j] * B[i, j];
            return new Matrix(table);
        }
        public static Matrix Transpose(Matrix A)
        {
            double[,] table = new double[A.N,A.M];
            for (int i = 0; i < A.N; ++i)
                for (int j = 0; j < A.M; ++j)
                    table[i, j] = A[j, i];
            return new Matrix(table);
        }
        public static Matrix Function(Matrix A, Func<double, double> function)
        {
            double[,] table = A.GetTable();
            for (int i = 0; i < A.M; ++i)
                for (int j = 0; j < A.N; ++j)
                    table[i, j] = function(A[i, j]);
            return new Matrix(table);
        }
        /// <summary>
        /// Returns the projection of b onto the column space of A.
        /// Only works if the columns of A are linearly independent.
        /// </summary>
        /// <param name="A"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix Projection(Matrix A, Matrix b)
        {
            return A*!(Transpose(A)*A)*(Transpose(A))*(b);//May crash due to the priority of the operators being undefined. To remake in multiple lines if so.
        }
        /// <summary>
        /// Return the best solution for Ax = b, x^.
        /// </summary>
        /// <param name="A"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix LeastSquares(Matrix A, Matrix b)
        {
            return !(Transpose(A)*A)*Transpose(A)*b;//May crash due to the priority of the operators being undefined. To remake in multiple lines if so.
        }
        public static double[,] ParametricCurveFitting(double[,] points, int degree)
        {
            return new double[0,0];
        }

        public static Matrix GramSchmidt(Matrix A)
        {
            double[,] table = ReplaceColumn(A,SubMatrix(A,0,0,A.M,1)).GetTable();
            for(int j = 1; j < A.N; ++j)
            {
                for(int k = j -1; k >= 0; --k)
                {
                    table = ReplaceColumn(new Matrix(table), SubMatrix(new Matrix(table), 0, j, A.M, 1)).GetTable();
                }
                table = ReplaceColumn(new Matrix(table), Normalize(SubMatrix(new Matrix(table),0,j,A.M,1))).GetTable();
            }
            return new Matrix(table);
        }
        #endregion

        #region Client code interaction functions.
        public double this [int i, int j] { get { return Table[i, j]; } }
        public double[,] GetTable()
        {
            double[,] t = new double[M, N];
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    t[i, j] = Table[i, j];
            return t;
        }
        public void Show()
        {
            for (int i = 0; i < M; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    Console.Write(Table[i, j] + "   ");
                }
                Console.WriteLine();
            }
        }
        static public Matrix SubMatrix(Matrix A, int rowOrigin, int columnOrigin, int rowDimension, int columnDimension)
        {
            double[,] table = new double[rowDimension, columnDimension];
            for (int i = 0; i < rowDimension; ++i)
                for (int j = 0; j < columnDimension; ++j)
                    table[i, j] = A[i + rowOrigin, j + columnOrigin];
            return new Matrix(table);
        }
        static public Matrix Identity(int n)
        {
            double[,] table = new double[n, n];
            for (int i = 0; i < n; ++i)
                table[i, i] = 1;
            return new Matrix(table);
        }
        static public Matrix Augment(Matrix A, Matrix B)
        {
            double[,] table = new double[A.M, A.N + B.N];
            for(int i = 0; i < A.M; ++i)
            {
                for(int j = 0; j < A.N+B.N; ++j)
                {
                    if (j >= A.N)
                        table[i, j] = B[i, j - A.N];
                    else
                        table[i, j] = A[i, j];
                }
            }
            return new Matrix(table);
        }
        public static double[,] SwitchLines(int line1, int line2, double[,] table)
        {
            double buffer = 0;
            for (int i = 0; i < table.GetLength(1); ++i)
            {
                buffer = table[i, line1];
                table[i, line1] = table[i, line2];
                table[i, line2] = buffer;
            }
            return table;
        }
        public static double Norm(Matrix vector)
        {
            double norm = 0;
            for(int i = 0; i < vector.M;++i)
                norm += (double)Math.Pow(vector[i,0], 2);
            return (double)Math.Sqrt(norm);
        }
        public static Matrix Normalize(Matrix vector)
        {
            double norm = Norm(vector);
            double[,] table = vector.GetTable();
            for (int i = 0; i < vector.M; ++i)
                table[i, 0] /= norm;
            return new Matrix(table);
        }
        public static Matrix ReplaceLine(Matrix A, Matrix line)
        {
            return new Matrix(new double[0, 0]);
        }
        public static Matrix ReplaceColumn(Matrix A, Matrix column)
        {
            return new Matrix(new double[0, 0]);
        }
        #endregion
    }
}
