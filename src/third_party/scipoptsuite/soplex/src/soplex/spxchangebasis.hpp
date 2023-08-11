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

#include <iostream>

#include "soplex/spxdefines.h"
#include "soplex/spxbasis.h"
#include "soplex/spxsolver.h"
#include "soplex/spxout.h"

namespace soplex
{
template <class R>
void SPxBasisBase<R>::reDim()
{

   assert(theLP != 0);

   MSG_DEBUG(std::cout << "DCHBAS01 SPxBasisBase<R>::reDim():"
             << " matrixIsSetup=" << matrixIsSetup
             << " fatorized=" << factorized
             << std::endl;)

   thedesc.reSize(theLP->nRows(), theLP->nCols());

   if(theLP->dim() != matrix.size())
   {
      MSG_INFO3((*this->spxout), (*this->spxout) <<
                "ICHBAS02 basis redimensioning invalidates factorization"
                << std::endl;)

      matrix.reSize(theLP->dim());
      theBaseId.reSize(theLP->dim());
      matrixIsSetup = false;
      factorized    = false;
   }

   MSG_DEBUG(std::cout << "DCHBAS03 SPxBasisBase<R>::reDim(): -->"
             << " matrixIsSetup=" << matrixIsSetup
             << " fatorized=" << factorized
             << std::endl;)

   assert(matrix.size()    >= theLP->dim());
   assert(theBaseId.size() >= theLP->dim());
}

/* adapt basis and basis descriptor to added rows */
template <class R>
void SPxBasisBase<R>::addedRows(int n)
{
   assert(theLP != 0);

   if(n > 0)
   {
      reDim();

      if(theLP->rep() == SPxSolverBase<R>::COLUMN)
      {
         /* after adding rows in column representation, reDim() should set these bools to false. */
         assert(!matrixIsSetup && !factorized);

         for(int i = theLP->nRows() - n; i < theLP->nRows(); ++i)
         {
            thedesc.rowStatus(i) = dualRowStatus(i);
            baseId(i) = theLP->SPxLPBase<R>::rId(i);
         }
      }
      else
      {
         assert(theLP->rep() == SPxSolverBase<R>::ROW);

         for(int i = theLP->nRows() - n; i < theLP->nRows(); ++i)
            thedesc.rowStatus(i) = dualRowStatus(i);
      }

      /* If matrix was set up, load new basis vectors to the matrix.
       * In the row case, the basis is not effected by adding rows. However,
       * since @c matrix stores references to the rows in the LP (SPxLPBase<R>), a realloc
       * in SPxLPBase<R> (e.g. due to space requirements) might invalidate these references.
       * We therefore have to "reload" the matrix if it is set up. Note that reDim()
       * leaves @c matrixIsSetup untouched if only row have been added, since the basis
       * matrix already has the correct size. */
      if(status() > NO_PROBLEM && matrixIsSetup)
         loadMatrixVecs();

      /* update basis status */
      switch(status())
      {
      case PRIMAL:
      case UNBOUNDED:
         setStatus(REGULAR);
         break;

      case OPTIMAL:
      case INFEASIBLE:
         setStatus(DUAL);
         break;

      case NO_PROBLEM:
      case SINGULAR:
      case REGULAR:
      case DUAL:
         break;

      default:
         MSG_ERROR(std::cerr << "ECHBAS04 Unknown basis status!" << std::endl;)
         throw SPxInternalCodeException("XCHBAS01 This should never happen.");
      }
   }
}

template <class R>
void SPxBasisBase<R>::removedRow(int i)
{

   assert(status() >  NO_PROBLEM);
   assert(theLP    != 0);

   if(theLP->rep() == SPxSolverBase<R>::ROW)
   {
      if(theLP->isBasic(thedesc.rowStatus(i)))
      {
         setStatus(NO_PROBLEM);
         factorized = false;

         MSG_DEBUG(std::cout << "DCHBAS05 Warning: deleting basic row!\n";)
      }
   }
   else
   {
      assert(theLP->rep() == SPxSolverBase<R>::COLUMN);
      factorized = false;

      if(!theLP->isBasic(thedesc.rowStatus(i)))
      {
         setStatus(NO_PROBLEM);
         MSG_DEBUG(std::cout << "DCHBAS06 Warning: deleting nonbasic row!\n";)
      }
      else if(status() > NO_PROBLEM && matrixIsSetup)
      {
         for(int j = theLP->dim(); j >= 0; --j)
         {
            SPxId id = baseId(j);

            if(id.isSPxRowId() && !theLP->has(SPxRowId(id)))
            {
               baseId(j) = baseId(theLP->dim());

               if(j < theLP->dim())
                  matrix[j] = &theLP->vector(baseId(j));

               break;
            }
         }
      }
   }

   thedesc.rowStatus(i) = thedesc.rowStatus(theLP->nRows());
   reDim();
}

template <class R>
void SPxBasisBase<R>::removedRows(const int perm[])
{
   assert(status() > NO_PROBLEM);
   assert(theLP != 0);

   int i;
   int n = thedesc.nRows();

   if(theLP->rep() == SPxSolverBase<R>::ROW)
   {
      for(i = 0; i < n; ++i)
      {
         if(perm[i] != i)
         {
            if(perm[i] < 0)                // row got removed
            {
               if(theLP->isBasic(thedesc.rowStatus(i)))
               {
                  setStatus(NO_PROBLEM);
                  factorized = matrixIsSetup = false;
                  MSG_DEBUG(std::cout << "DCHBAS07 Warning: deleting basic row!\n";)
               }
            }
            else                            // row was moved
               thedesc.rowStatus(perm[i]) = thedesc.rowStatus(i);
         }
      }
   }
   else
   {
      assert(theLP->rep() == SPxSolverBase<R>::COLUMN);

      factorized    = false;
      matrixIsSetup = false;

      for(i = 0; i < n; ++i)
      {
         if(perm[i] != i)
         {
            if(perm[i] < 0)                // row got removed
            {
               if(!theLP->isBasic(thedesc.rowStatus(i)))
                  setStatus(NO_PROBLEM);
            }
            else                            // row was moved
               thedesc.rowStatus(perm[i]) = thedesc.rowStatus(i);
         }
      }
   }

   reDim();
}

template <class R>
static typename SPxBasisBase<R>::Desc::Status
primalColStatus(int i, const SPxLPBase<R>* theLP)
{
   assert(theLP != 0);

   if(theLP->upper(i) < R(infinity))
   {
      if(theLP->lower(i) > R(-infinity))
      {
         if(theLP->lower(i) == theLP->SPxLPBase<R>::upper(i))
            return SPxBasisBase<R>::Desc::P_FIXED;
         /*
           else
           return (-theLP->lower(i) < theLP->upper(i))
           ? SPxBasisBase<R>::Desc::P_ON_LOWER
           : SPxBasisBase<R>::Desc::P_ON_UPPER;
         */
         else if(theLP->maxObj(i) == 0)
            return (-theLP->lower(i) < theLP->upper(i))
                   ? SPxBasisBase<R>::Desc::P_ON_LOWER
                   : SPxBasisBase<R>::Desc::P_ON_UPPER;
         else
            return (theLP->maxObj(i) < 0)
                   ? SPxBasisBase<R>::Desc::P_ON_LOWER
                   : SPxBasisBase<R>::Desc::P_ON_UPPER;
      }
      else
         return SPxBasisBase<R>::Desc::P_ON_UPPER;
   }
   else if(theLP->lower(i) > R(-infinity))
      return SPxBasisBase<R>::Desc::P_ON_LOWER;
   else
      return SPxBasisBase<R>::Desc::P_FREE;
}


/* adapt basis and basis descriptor to added columns */
template <class R>
void SPxBasisBase<R>::addedCols(int n)
{
   assert(theLP != 0);

   if(n > 0)
   {
      reDim();

      if(theLP->rep() == SPxSolverBase<R>::ROW)
      {
         /* after adding columns in row representation, reDim() should set these bools to false. */
         assert(!matrixIsSetup && !factorized);

         for(int i = theLP->nCols() - n; i < theLP->nCols(); ++i)
         {
            thedesc.colStatus(i) = primalColStatus(i, theLP);
            baseId(i) = theLP->SPxLPBase<R>::cId(i);
         }
      }
      else
      {
         assert(theLP->rep() == SPxSolverBase<R>::COLUMN);

         for(int i = theLP->nCols() - n; i < theLP->nCols(); ++i)
            thedesc.colStatus(i) = primalColStatus(i, theLP);
      }

      /* If matrix was set up, load new basis vectors to the matrix
       * In the column case, the basis is not effected by adding columns. However,
       * since @c matrix stores references to the columns in the LP (SPxLPBase<R>), a realloc
       * in SPxLPBase<R> (e.g. due to space requirements) might invalidate these references.
       * We therefore have to "reload" the matrix if it is set up. Note that reDim()
       * leaves @c matrixIsSetup untouched if only columns have been added, since the
       * basis matrix already has the correct size. */
      if(status() > NO_PROBLEM && matrixIsSetup)
         loadMatrixVecs();

      switch(status())
      {
      case DUAL:
      case INFEASIBLE:
         setStatus(REGULAR);
         break;

      case OPTIMAL:
      case UNBOUNDED:
         setStatus(PRIMAL);
         break;

      case NO_PROBLEM:
      case SINGULAR:
      case REGULAR:
      case PRIMAL:
         break;

      default:
         MSG_ERROR(std::cerr << "ECHBAS08 Unknown basis status!" << std::endl;)
         throw SPxInternalCodeException("XCHBAS02 This should never happen.");
      }
   }
}

template <class R>
void SPxBasisBase<R>::removedCol(int i)
{
   assert(status() > NO_PROBLEM);
   assert(theLP != 0);

   if(theLP->rep() == SPxSolverBase<R>::COLUMN)
   {
      if(theLP->isBasic(thedesc.colStatus(i)))
         setStatus(NO_PROBLEM);
   }
   else
   {
      assert(theLP->rep() == SPxSolverBase<R>::ROW);
      factorized = false;

      if(!theLP->isBasic(thedesc.colStatus(i)))
         setStatus(NO_PROBLEM);
      else if(status() > NO_PROBLEM)
      {
         for(int j = theLP->dim(); j >= 0; --j)
         {
            SPxId id = baseId(j);

            if(id.isSPxColId() && !theLP->has(SPxColId(id)))
            {
               baseId(j) = baseId(theLP->dim());

               if(matrixIsSetup &&
                     j < theLP->dim())
                  matrix[j] = &theLP->vector(baseId(j));

               break;
            }
         }
      }
   }

   thedesc.colStatus(i) = thedesc.colStatus(theLP->nCols());
   reDim();
}

template <class R>
void SPxBasisBase<R>::removedCols(const int perm[])
{
   assert(status() > NO_PROBLEM);
   assert(theLP != 0);

   int i;
   int n = thedesc.nCols();

   if(theLP->rep() == SPxSolverBase<R>::COLUMN)
   {
      for(i = 0; i < n; ++i)
      {
         if(perm[i] < 0)            // column got removed
         {
            if(theLP->isBasic(thedesc.colStatus(i)))
               setStatus(NO_PROBLEM);
         }
         else                        // column was potentially moved
            thedesc.colStatus(perm[i]) = thedesc.colStatus(i);
      }
   }
   else
   {
      assert(theLP->rep() == SPxSolverBase<R>::ROW);
      factorized = matrixIsSetup = false;

      for(i = 0; i < n; ++i)
      {
         if(perm[i] != i)
         {
            if(perm[i] < 0)                // column got removed
            {
               if(!theLP->isBasic(thedesc.colStatus(i)))
                  setStatus(NO_PROBLEM);
            }
            else                            // column was moved
               thedesc.colStatus(perm[i]) = thedesc.colStatus(i);
         }
      }
   }

   reDim();
}


/**
 * mark the basis as not factorized
 */
template <class R>
void SPxBasisBase<R>::invalidate()
{
   if(factorized || matrixIsSetup)
   {
      MSG_INFO3((*this->spxout), (*this->spxout) << "ICHBAS09 explicit invalidation of factorization" <<
                std::endl;)
   }

   factorized    = false;
   matrixIsSetup = false;
}

/**
 * Create the initial slack basis descriptor and set up the basis matrix accordingly.
 * This code has been adapted from SPxBasisBase<R>::addedRows() and SPxBasisBase<R>::addedCols().
 */
template <class R>
void SPxBasisBase<R>::restoreInitialBasis()
{
   assert(!factorized);

   MSG_INFO3((*this->spxout), (*this->spxout) << "ICHBAS10 setup slack basis" << std::endl;)

   if(theLP->rep() == SPxSolverBase<R>::COLUMN)
   {
      for(int i = 0; i < theLP->nRows(); ++i)
      {
         thedesc.rowStatus(i) = dualRowStatus(i);
         baseId(i) = theLP->SPxLPBase<R>::rId(i);
      }

      for(int i = 0; i < theLP->nCols(); ++i)
         thedesc.colStatus(i) = primalColStatus(i, theLP);
   }
   else
   {
      assert(theLP->rep() == SPxSolverBase<R>::ROW);

      for(int i = 0; i < theLP->nRows(); ++i)
         thedesc.rowStatus(i) = dualRowStatus(i);

      for(int i = 0; i < theLP->nCols(); ++i)
      {
         thedesc.colStatus(i) = primalColStatus(i, theLP);
         baseId(i) = theLP->SPxLPBase<R>::cId(i);
      }
   }

   /* if matrix was set up, load new basis vectors to the matrix */
   if(status() > NO_PROBLEM && matrixIsSetup)
      loadMatrixVecs();

   /* update basis status */
   setStatus(REGULAR);
}

/**
 * @todo Implement changedRow(), changedCol(), changedElement() in a more clever
 * way. For instance, the basis won't be singular (but maybe infeasible) if the
 * change doesn't affect the basis rows/columns.
 *
 * The following methods (changedRow(), changedCol(), changedElement()) radically
 * change the current basis to the original (slack) basis also present after
 * loading the LP. The reason is that through the changes, the current basis may
 * become singular. Going back to the initial basis is quite inefficient, but
 * correct.
 */

/**@todo is this correctly implemented?
 */
template <class R>
void SPxBasisBase<R>::changedRow(int /*row*/)
{
   invalidate();
   restoreInitialBasis();
}

/**@todo is this correctly implemented?
 */
template <class R>
void SPxBasisBase<R>::changedCol(int /*col*/)
{
   invalidate();
   restoreInitialBasis();
}

/**@todo is this correctly implemented?
 */
template <class R>
void SPxBasisBase<R>::changedElement(int /*row*/, int /*col*/)
{
   invalidate();
   restoreInitialBasis();
}
} // namespace soplex
