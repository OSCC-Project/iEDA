/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the class library                   */
/*       SoPlex --- the Sequential object-oriented simPlex.                  */
/*                                                                           */
/*  Copyright 1996-2022 Zuse Institute Berlin                                */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*      http://www.apache.org/licenses/LICENSE-2.0                           */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*  You should have received a copy of the Apache-2.0 license                */
/*  along with SoPlex; see the file LICENSE. If not email to soplex@zib.de.  */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file  clufactor.h
 * @brief Implementation of sparse LU factorization.
 */
#ifndef _CLUFACTOR_H_
#define _CLUFACTOR_H_

#include "soplex/spxdefines.h"
#include "soplex/slinsolver.h"
#include "soplex/timer.h"
#include "soplex/svector.h"

#include "vector"

#define WITH_L_ROWS 1

namespace soplex
{
/**@brief   Implementation of sparse LU factorization.
 * @ingroup Algo
 *
 * This class implements a sparse LU factorization with either
 * FOREST-TOMLIN or ETA updates, using dynamic Markowitz pivoting.
 */
template <class R>
class CLUFactor
{
public:

   //----------------------------------------
   /**@name Public types */
   ///@{
   /** Doubly linked ring structure for garbage collection of column or
    *  row file in working matrix.
    */
   struct Dring
   {
      Dring* next;
      Dring* prev;
      int    idx;
   };

   /// Pivot Ring
   class Pring
   {
   public:
      Pring* next;                ///<
      Pring* prev;                ///<
      int    idx;                 ///< index of pivot row
      int    pos;                 ///< position of pivot column in row
      int    mkwtz;               ///< markowitz number of pivot

      Pring() : next(0), prev(0)  ///< constructor
      {
         mkwtz = -1;
         idx = -1;
         pos = -1;
      }

   private:
      Pring(const Pring&);             ///< blocked copy constructor
      Pring& operator= (const Pring&); ///< blocked assignment operator
   };
   ///@}

protected:

   //----------------------------------------
   /**@name Protected types */
   ///@{
   /// Temporary data structures.
   class Temp
   {
   public:
      int*    s_mark;       ///< marker
      std::vector<R>   s_max;        ///< maximum absolute value per row (or -1)
      int*    s_cact;       ///< lengths of columns of active submatrix
      int     stage;        ///< stage of the structure
      Pring   pivots;       ///< ring of selected pivot rows
      Pring*  pivot_col;    ///< column index handlers for R linked list
      Pring*  pivot_colNZ;  ///< lists for columns to number of nonzeros
      Pring*  pivot_row;    ///< row index handlers for R linked list
      Pring*  pivot_rowNZ;  ///< lists for rows to number of nonzeros

      Temp();               ///< constructor
      ~Temp();              ///< destructor
      void init(int p_dim); ///< initialization
      void clear();         ///< clears the structure

   private:
      Temp(const Temp&);               ///< blocked copy constructor
      Temp& operator= (const Temp&);   ///< blocked assignment operator
   };

   /// Data structures for saving the row and column permutations.
   struct Perm
   {
      int* orig;          ///< orig[p] original index from p
      int* perm;          ///< perm[i] permuted index from i
   };

   /// Data structures for saving the working matrix and U factor.
   struct U
   {
      ///
      struct Row
      {
         Dring list;         /*!< \brief Double linked ringlist of VectorBase<R>
                               indices in the order they appear
                               in the row file                      */
         Dring* elem;        ///< %Array of ring elements.
         int    size;        ///< size of arrays val and idx
         int    used;        ///< used entries of arrays idx and val
         std::vector<R>  val;         ///< hold nonzero values
         int*   idx;         ///< hold column indices of nonzeros
         int*   start;       ///< starting positions in val and idx
         int*   len;         ///< used nonzeros per row vectors
         int*   max;         /*!< \brief maximum available nonzeros per row:
                               start[i] + max[i] == start[elem[i].next->idx]
                               len[i] <= max[i].                    */
      } row;

      ///
      struct Col
      {
         Dring list;         /*!< \brief Double linked ringlist of VectorBase<R>
                                indices in the order they appear
                                in the column file                  */
         Dring* elem;        ///< %Array of ring elements.
         int size;           ///< size of array idx
         int used;           ///< used entries of array idx
         int* idx;           ///< hold row indices of nonzeros
         std::vector<R> val;          /*!< \brief hold nonzero values: this is only initialized
                                in the end of the factorization with DEFAULT
                                updates.                            */
         int* start;         ///< starting positions in val and idx
         int* len;           ///< used nonzeros per column vector
         int* max;           /*!< \brief maximum available nonzeros per colunn:
                               start[i] + max[i] == start[elem[i].next->idx]
                               len[i] <= max[i].                    */
      } col;
   };


   /// Data structures for saving the working matrix and L factor.
   struct L
   {
      int  size;           ///< size of arrays val and idx
      std::vector<R> val;           ///< values of L vectors
      int*  idx;           ///< indices of L vectors
      int  startSize;      ///< size of array start
      int  firstUpdate;    ///< number of first update L vector
      int  firstUnused;    ///< number of first unused L vector
      int*  start;         ///< starting positions in val and idx
      int*  row;           ///< column indices of L vectors
      int  updateType;     ///< type of updates to be used.

      /* The following arrays have length |firstUpdate|, since they keep
       * rows of the L-vectors occuring during the factorization (without
       * updates), only:
       */
      std::vector<R> rval;          ///< values of rows of L
      int*  ridx;          ///< indices of rows of L
      int*  rbeg;          ///< start of rows in rval and ridx
      int*  rorig;         ///< original row permutation
      int*  rperm;         ///< original row permutation
   };
   ///@}

   //----------------------------------------
   /**@name Protected data */
   ///@{
   typename SLinSolver<R>::Status stat;   ///< Status indicator.

   int     thedim;            ///< dimension of factorized matrix
   int     nzCnt;             ///< number of nonzeros in U
   R    initMaxabs;        ///< maximum abs number in initail Matrix
   R    maxabs;            ///< maximum abs number in L and U

   R    rowMemMult;        ///< factor of minimum Memory * number of nonzeros
   R    colMemMult;        ///< factor of minimum Memory * number of nonzeros
   R    lMemMult;          ///< factor of minimum Memory * number of nonzeros

   Perm    row;               ///< row permutation matrices
   Perm    col;               ///< column permutation matrices

   L       l;                 ///< L matrix
   std::vector<R>   diag;              ///< Array of pivot elements
   U       u;                 ///< U matrix

   R*   work;              ///< Working array: must always be left as 0!

   Timer*  factorTime;        ///< Time spent in factorizations
   int     factorCount;       ///< Number of factorizations
   int     hugeValues;        ///< number of times huge values occurred during solve (only used in debug mode)
   ///@}

private:

   //----------------------------------------
   /**@name Private data */
   ///@{
   Temp    temp;              ///< Temporary storage
   ///@}

   //----------------------------------------
   /**@name Solving
      These helper methods are used during the factorization process.
      The solve*-methods solve lower and upper triangular systems from
      the left or from the right, respectively  The methods with '2' in
      the end solve two systems at the same time.  The methods with
      "Eps" in the end consider elements smaller then the passed epsilon
      as zero.
   */
   ///@{
   ///
   void solveUright(R* wrk, R* vec) const;
   ///
   int  solveUrightEps(R* vec, int* nonz, R eps, R* rhs);
   ///
   void solveUright2(R* work1, R* vec1, R* work2, R* vec2);
   ///
   int  solveUright2eps(R* work1, R* vec1, R* work2, R* vec2, int* nonz, R eps);
   ///
   void solveLright2(R* vec1, R* vec2);
   ///
   void solveUpdateRight(R* vec);
   ///
   void solveUpdateRight2(R* vec1, R* vec2);
   ///
   void solveUleft(R* work, R* vec);
   ///
   void solveUleft2(R* work1, R* vec1, R* work2, R* vec2);
   ///
   int solveLleft2forest(R* vec1, int* /* nonz */, R* vec2, R /* eps */);
   ///
   void solveLleft2(R* vec1, int* /* nonz */, R* vec2, R /* eps */);
   ///
   int solveLleftForest(R* vec, int* /* nonz */, R /* eps */);
   ///
   void solveLleft(R* vec) const;
   ///
   int solveLleftEps(R* vec, int* nonz, R eps);
   ///
   void solveUpdateLeft(R* vec);
   ///
   void solveUpdateLeft2(R* vec1, R* vec2);

   void inline updateSolutionVectorLright(R change, int j, R& vec, int* idx, int& nnz);
   ///
   void vSolveLright(R* vec, int* ridx, int& rn, R eps);
   ///
   void vSolveLright2(R* vec, int* ridx, int& rn, R eps,
                      R* vec2, int* ridx2, int& rn2, R eps2);
   ///
   void vSolveLright3(R* vec, int* ridx, int& rn, R eps,
                      R* vec2, int* ridx2, int& rn2, R eps2,
                      R* vec3, int* ridx3, int& rn3, R eps3);
   ///
   int vSolveUright(R* vec, int* vidx, R* rhs, int* ridx, int rn, R eps);
   ///
   void vSolveUrightNoNZ(R* vec, R* rhs, int* ridx, int rn, R eps);
   ///
   int vSolveUright2(R* vec, int* vidx, R* rhs, int* ridx, int rn, R eps,
                     R* vec2, R* rhs2, int* ridx2, int rn2, R eps2);
   ///
   int vSolveUpdateRight(R* vec, int* ridx, int n, R eps);
   ///
   void vSolveUpdateRightNoNZ(R* vec, R /*eps*/);
   ///
   int solveUleft(R eps, R* vec, int* vecidx, R* rhs, int* rhsidx, int rhsn);
   ///
   void solveUleftNoNZ(R eps, R* vec, R* rhs, int* rhsidx, int rhsn);
   ///
   int solveLleftForest(R eps, R* vec, int* nonz, int n);
   ///
   void solveLleftForestNoNZ(R* vec);
   ///
   int solveLleft(R eps, R* vec, int* nonz, int rn);
   ///
   void solveLleftNoNZ(R* vec);
   ///
   int solveUpdateLeft(R eps, R* vec, int* nonz, int n);

   ///
   void forestPackColumns();
   ///
   void forestMinColMem(int size);
   ///
   void forestReMaxCol(int col, int len);

   ///
   void initPerm();
   ///
   void initFactorMatrix(const SVectorBase<R>** vec, const R eps);
   ///
   void minLMem(int size);
   ///
   void setPivot(const int p_stage, const int p_col, const int p_row, const R val);
   ///
   void colSingletons();
   ///
   void rowSingletons();

   ///
   void initFactorRings();
   ///
   void freeFactorRings();

   ///
   int setupColVals();
   ///
   void setupRowVals();

   ///
   void eliminateRowSingletons();
   ///
   void eliminateColSingletons();
   ///
   void selectPivots(R threshold);
   ///
   int updateRow(int r, int lv, int prow, int pcol, R pval, R eps);

   ///
   void eliminatePivot(int prow, int pos, R eps);
   ///
   void eliminateNucleus(const R eps, const R threshold);
   ///
   void minRowMem(int size);
   ///
   void minColMem(int size);
   ///
   void remaxCol(int p_col, int len);
   ///
   void packRows();
   ///
   void packColumns();
   ///
   void remaxRow(int p_row, int len);
   ///
   int makeLvec(int p_len, int p_row);
   ///@}

protected:

   //----------------------------------------
   /**@name Solver methods */
   ///@{
   ///
   void solveLright(R* vec);
   ///
   int  solveRight4update(R* vec, int* nonz, R eps, R* rhs,
                          R* forest, int* forestNum, int* forestIdx);
   ///
   void solveRight(R* vec, R* rhs);
   ///
   int  solveRight2update(R* vec1, R* vec2, R* rhs1,
                          R* rhs2, int* nonz, R eps, R* forest, int* forestNum, int* forestIdx);
   ///
   void solveRight2(R* vec1, R* vec2, R* rhs1, R* rhs2);
   ///
   void solveLeft(R* vec, R* rhs);
   ///
   int solveLeftEps(R* vec, R* rhs, int* nonz, R eps);
   ///
   int solveLeft2(R* vec1, int* nonz, R* vec2, R eps, R* rhs1, R* rhs2);

   ///
   int vSolveRight4update(R eps,
                          R* vec, int* idx,               /* result       */
                          R* rhs, int* ridx, int rn,      /* rhs & Forest */
                          R* forest, int* forestNum, int* forestIdx);
   ///
   int vSolveRight4update2(R eps,
                           R* vec, int* idx,              /* result1 */
                           R* rhs, int* ridx, int rn,     /* rhs1    */
                           R* vec2, R eps2,            /* result2 */
                           R* rhs2, int* ridx2, int rn2,  /* rhs2    */
                           R* forest, int* forestNum, int* forestIdx);
   /// sparse version of above method
   void vSolveRight4update2sparse(
      R eps, R* vec, int* idx,    /* result1 */
      R* rhs, int* ridx, int& rn,    /* rhs1    */
      R eps2, R* vec2, int* idx2, /* result2 */
      R* rhs2, int* ridx2, int& rn2, /* rhs2    */
      R* forest, int* forestNum, int* forestIdx);
   ///
   int vSolveRight4update3(R eps,
                           R* vec, int* idx,              /* result1 */
                           R* rhs, int* ridx, int rn,     /* rhs1    */
                           R* vec2, R eps2,            /* result2 */
                           R* rhs2, int* ridx2, int rn2,  /* rhs2    */
                           R* vec3, R eps3,            /* result3 */
                           R* rhs3, int* ridx3, int rn3,  /* rhs3    */
                           R* forest, int* forestNum, int* forestIdx);
   /// sparse version of above method
   void vSolveRight4update3sparse(
      R eps, R* vec, int* idx,    /* result1 */
      R* rhs, int* ridx, int& rn,    /* rhs1    */
      R eps2, R* vec2, int* idx2, /* result2 */
      R* rhs2, int* ridx2, int& rn2, /* rhs2    */
      R eps3, R* vec3, int* idx3, /* result3 */
      R* rhs3, int* ridx3, int& rn3, /* rhs3    */
      R* forest, int* forestNum, int* forestIdx);
   ///
   void vSolveRightNoNZ(R* vec, R eps,    /* result */
                        R* rhs, int* ridx, int rn);            /* rhs    */
   ///
   int vSolveLeft(R eps,
                  R* vec, int* idx,                      /* result */
                  R* rhs, int* ridx, int rn);            /* rhs    */
   ///
   void vSolveLeftNoNZ(R eps,
                       R* vec,                           /* result */
                       R* rhs, int* ridx, int rn);       /* rhs    */
   ///
   int vSolveLeft2(R eps,
                   R* vec, int* idx,                     /* result */
                   R* rhs, int* ridx, int rn,            /* rhs    */
                   R* vec2,                              /* result2 */
                   R* rhs2, int* ridx2, int rn2);        /* rhs2    */
   /// sparse version of solving 2 systems of equations
   void vSolveLeft2sparse(R eps,
                          R* vec, int* idx,                     /* result */
                          R* rhs, int* ridx, int& rn,           /* rhs    */
                          R* vec2, int* idx2,                   /* result2 */
                          R* rhs2, int* ridx2, int& rn2);       /* rhs2    */
   ///
   int vSolveLeft3(R eps,
                   R* vec, int* idx,                     /* result */
                   R* rhs, int* ridx, int rn,            /* rhs    */
                   R* vec2,                              /* result2 */
                   R* rhs2, int* ridx2, int rn2,         /* rhs2    */
                   R* vec3,                              /* result3 */
                   R* rhs3, int* ridx3, int rn3);        /* rhs3    */
   /// sparse version of solving 3 systems of equations
   void vSolveLeft3sparse(R eps,
                          R* vec, int* idx,                     /* result */
                          R* rhs, int* ridx, int& rn,           /* rhs    */
                          R* vec2, int* idx2,                   /* result2 */
                          R* rhs2, int* ridx2, int& rn2,        /* rhs2    */
                          R* vec3, int* idx3,                   /* result2 */
                          R* rhs3, int* ridx3, int& rn3);       /* rhs2    */

   void forestUpdate(int col, R* work, int num, int* nonz);

   void update(int p_col, R* p_work, const int* p_idx, int num);
   void updateNoClear(int p_col, const R* p_work, const int* p_idx, int num);

   ///
   void factor(const SVectorBase<R>** vec,   ///< Array of column VectorBase<R> pointers
               R threshold,    ///< pivoting threshold
               R eps);         ///< epsilon for zero detection
   ///@}

   //----------------------------------------
   /**@name Debugging */
   ///@{
   ///
   void dump() const;

   ///
   bool isConsistent() const;
   ///@}
};

} // namespace soplex

// For general templated functions
#include "clufactor.hpp"

#endif // _CLUFACTOR_H_
