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

#include <assert.h>
#include <iostream>

#include "soplex/spxdefines.h"
#include "soplex/spxsolver.h"
#include "soplex/spxpricer.h"
#include "soplex/spxratiotester.h"
#include "soplex/exceptions.h"

namespace soplex
{

template <class R>
void SPxSolverBase<R>::addedRows(int n)
{

   if(n > 0)
   {
      SPxLPBase<R>::addedRows(n);

      unInit();
      reDim();

      if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
         SPxBasisBase<R>::addedRows(n);
   }

   /* we must not assert consistency here, since addedCols() might be still necessary to obtain a consistent basis */
}

template <class R>
void SPxSolverBase<R>::addedCols(int n)
{

   if(n > 0)
   {
      SPxLPBase<R>::addedCols(n);

      unInit();
      reDim();

      if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
         SPxBasisBase<R>::addedCols(n);
   }

   /* we must not assert consistency here, since addedRows() might be still necessary to obtain a consistent basis */
}

template <class R>
void SPxSolverBase<R>::doRemoveRow(int i)
{

   SPxLPBase<R>::doRemoveRow(i);

   unInit();

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
   {
      this->removedRow(i);

      switch(SPxBasisBase<R>::status())
      {
      case SPxBasisBase<R>::DUAL:
      case SPxBasisBase<R>::INFEASIBLE:
         setBasisStatus(SPxBasisBase<R>::REGULAR);
         break;

      case SPxBasisBase<R>::OPTIMAL:
         setBasisStatus(SPxBasisBase<R>::PRIMAL);
         break;

      default:
         break;
      }
   }
}

template <class R>
void SPxSolverBase<R>::doRemoveRows(int perm[])
{

   SPxLPBase<R>::doRemoveRows(perm);

   unInit();

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
   {
      this->removedRows(perm);

      switch(SPxBasisBase<R>::status())
      {
      case SPxBasisBase<R>::DUAL:
      case SPxBasisBase<R>::INFEASIBLE:
         setBasisStatus(SPxBasisBase<R>::REGULAR);
         break;

      case SPxBasisBase<R>::OPTIMAL:
         setBasisStatus(SPxBasisBase<R>::PRIMAL);
         break;

      default:
         break;
      }
   }
}

template <class R>
void SPxSolverBase<R>::doRemoveCol(int i)
{
   forceRecompNonbasicValue();

   SPxLPBase<R>::doRemoveCol(i);

   unInit();

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
   {
      this->removedCol(i);

      switch(SPxBasisBase<R>::status())
      {
      case SPxBasisBase<R>::PRIMAL:
      case SPxBasisBase<R>::UNBOUNDED:
         setBasisStatus(SPxBasisBase<R>::REGULAR);
         break;

      case SPxBasisBase<R>::OPTIMAL:
         setBasisStatus(SPxBasisBase<R>::DUAL);
         break;

      default:
         break;
      }
   }
}

template <class R>
void SPxSolverBase<R>::doRemoveCols(int perm[])
{
   forceRecompNonbasicValue();

   SPxLPBase<R>::doRemoveCols(perm);

   unInit();

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
   {
      this->removedCols(perm);

      switch(SPxBasisBase<R>::status())
      {
      case SPxBasisBase<R>::PRIMAL:
      case SPxBasisBase<R>::UNBOUNDED:
         setBasisStatus(SPxBasisBase<R>::REGULAR);
         break;

      case SPxBasisBase<R>::OPTIMAL:
         setBasisStatus(SPxBasisBase<R>::DUAL);
         break;

      default:
         break;
      }
   }
}

template <class R>
void SPxSolverBase<R>::changeObj(const VectorBase<R>& newObj, bool scale)
{
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeObj(newObj, scale);

   /**@todo Factorization remains valid, we do not need a reDim()
    *       pricing vectors should be recomputed.
    */
   unInit();
}

template <class R>
void SPxSolverBase<R>::changeObj(int i, const R& newVal, bool scale)
{
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeObj(i, newVal, scale);


   /**@todo Factorization remains valid, we do not need a reDim()
    *       pricing vectors should be recomputed.
    */
   unInit();
}

template <class R>
void SPxSolverBase<R>::changeMaxObj(const VectorBase<R>& newObj, bool scale)
{
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeMaxObj(newObj, scale);

   /**@todo Factorization remains valid, we do not need a reDim()
    *       pricing vectors should be recomputed.
    */
   unInit();
}

template <class R>
void SPxSolverBase<R>::changeMaxObj(int i, const R& newVal, bool scale)
{
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeMaxObj(i, newVal, scale);

   /**@todo Factorization remains valid, we do not need a reDim()
    *       pricing vectors should be recomputed.
    */
   unInit();
}

template <class R>
void SPxSolverBase<R>::changeRowObj(const VectorBase<R>& newObj, bool scale)
{
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeRowObj(newObj, scale);

   /**@todo Factorization remains valid, we do not need a reDim()
    *       pricing vectors should be recomputed.
    */
   unInit();
}

template <class R>
void SPxSolverBase<R>::changeRowObj(int i, const R& newVal, bool scale)
{
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeRowObj(i, newVal, scale);

   /**@todo Factorization remains valid, we do not need a reDim()
    *       pricing vectors should be recomputed.
    */
   unInit();
}

template <class R>
void SPxSolverBase<R>::changeLowerStatus(int i, R newLower, R oldLower)
{
   typename SPxBasisBase<R>::Desc::Status& stat      = this->desc().colStatus(i);
   R                    currUpper = this->upper(i);
   R                    objChange = 0.0;

   MSG_DEBUG(std::cout << "DCHANG01 changeLowerStatus(): col " << i
             << "[" << newLower << ":" << currUpper << "] " << stat;)

   switch(stat)
   {
   case SPxBasisBase<R>::Desc::P_ON_LOWER:
      if(newLower <= R(-infinity))
      {
         if(currUpper >= R(infinity))
         {
            stat = SPxBasisBase<R>::Desc::P_FREE;

            if(m_nonbasicValueUpToDate && rep() == COLUMN)
               objChange = -theLCbound[i] * oldLower;
         }
         else
         {
            stat = SPxBasisBase<R>::Desc::P_ON_UPPER;

            if(m_nonbasicValueUpToDate && rep() == COLUMN)
               objChange = (theUCbound[i] * currUpper) - (theLCbound[i] * oldLower);
         }
      }
      else if(EQ(newLower, currUpper))
      {
         stat = SPxBasisBase<R>::Desc::P_FIXED;

         if(m_nonbasicValueUpToDate && rep() == COLUMN)
            objChange = this->maxObj(i) * (newLower - oldLower);
      }
      else if(m_nonbasicValueUpToDate && rep() == COLUMN)
         objChange = theLCbound[i] * (newLower - oldLower);

      break;

   case SPxBasisBase<R>::Desc::P_ON_UPPER:
      if(EQ(newLower, currUpper))
         stat = SPxBasisBase<R>::Desc::P_FIXED;

      break;

   case SPxBasisBase<R>::Desc::P_FREE:
      if(newLower > R(-infinity))
      {
         stat = SPxBasisBase<R>::Desc::P_ON_LOWER;

         if(m_nonbasicValueUpToDate && rep() == COLUMN)
            objChange = theLCbound[i] * newLower;
      }

      break;

   case SPxBasisBase<R>::Desc::P_FIXED:
      if(NE(newLower, currUpper))
      {
         stat = SPxBasisBase<R>::Desc::P_ON_UPPER;

         if(isInitialized())
            theUCbound[i] = this->maxObj(i);
      }

      break;

   case SPxBasisBase<R>::Desc::D_FREE:
   case SPxBasisBase<R>::Desc::D_ON_UPPER:
   case SPxBasisBase<R>::Desc::D_ON_LOWER:
   case SPxBasisBase<R>::Desc::D_ON_BOTH:
   case SPxBasisBase<R>::Desc::D_UNDEFINED:
      if(rep() == ROW && theShift > 0.0)
         forceRecompNonbasicValue();

      stat = this->dualColStatus(i);
      break;

   default:
      throw SPxInternalCodeException("XCHANG01 This should never happen.");
   }

   MSG_DEBUG(std::cout << " -> " << stat << std::endl;)

   // we only need to update the nonbasic value in column representation (see nonbasicValue() for comparison/explanation)
   if(rep() == COLUMN)
      updateNonbasicValue(objChange);
}

template <class R>
void SPxSolverBase<R>::changeLower(const VectorBase<R>& newLower, bool scale)
{
   // we better recompute the nonbasic value when changing all lower bounds
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeLower(newLower, scale);

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
   {
      for(int i = 0; i < newLower.dim(); ++i)
         changeLowerStatus(i, this->lower(i));

      unInit();
   }
}

template <class R>
void SPxSolverBase<R>::changeLower(int i, const R& newLower, bool scale)
{
   if(newLower != (scale ? this->lowerUnscaled(i) : this->lower(i)))
   {
      forceRecompNonbasicValue();

      R oldLower = this->lower(i);
      // This has to be done before calling changeLowerStatus() because that is calling
      // basis.dualColStatus() which calls lower() and needs the changed value.
      SPxLPBase<R>::changeLower(i, newLower, scale);

      if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
      {
         changeLowerStatus(i, this->lower(i), oldLower);
         unInit();
      }
   }
}

template <class R>
void SPxSolverBase<R>::changeUpperStatus(int i, R newUpper, R oldUpper)
{
   typename SPxBasisBase<R>::Desc::Status& stat      = this->desc().colStatus(i);
   R                    currLower = this->lower(i);
   R                    objChange = 0.0;

   MSG_DEBUG(std::cout << "DCHANG02 changeUpperStatus(): col " << i
             << "[" << currLower << ":" << newUpper << "] " << stat;)

   switch(stat)
   {
   case SPxBasisBase<R>::Desc::P_ON_LOWER:
      if(newUpper == currLower)
         stat = SPxBasisBase<R>::Desc::P_FIXED;

      break;

   case SPxBasisBase<R>::Desc::P_ON_UPPER:
      if(newUpper >= R(infinity))
      {
         if(currLower <= R(-infinity))
         {
            stat = SPxBasisBase<R>::Desc::P_FREE;

            if(m_nonbasicValueUpToDate && rep() == COLUMN)
               objChange = -theUCbound[i] * oldUpper;
         }
         else
         {
            stat = SPxBasisBase<R>::Desc::P_ON_LOWER;

            if(m_nonbasicValueUpToDate && rep() == COLUMN)
               objChange = (theLCbound[i] * currLower) - (theUCbound[i] * oldUpper);
         }
      }
      else if(EQ(newUpper, currLower))
      {
         stat = SPxBasisBase<R>::Desc::P_FIXED;

         if(m_nonbasicValueUpToDate && rep() == COLUMN)
            objChange = this->maxObj(i) * (newUpper - oldUpper);
      }
      else if(m_nonbasicValueUpToDate && rep() == COLUMN)
         objChange = theUCbound[i] * (newUpper - oldUpper);

      break;

   case SPxBasisBase<R>::Desc::P_FREE:
      if(newUpper < R(infinity))
      {
         stat = SPxBasisBase<R>::Desc::P_ON_UPPER;

         if(m_nonbasicValueUpToDate && rep() == COLUMN)
            objChange = theUCbound[i] * newUpper;
      }

      break;

   case SPxBasisBase<R>::Desc::P_FIXED:
      if(NE(newUpper, currLower))
      {
         stat = SPxBasisBase<R>::Desc::P_ON_LOWER;

         if(isInitialized())
            theLCbound[i] = this->maxObj(i);
      }

      break;

   case SPxBasisBase<R>::Desc::D_FREE:
   case SPxBasisBase<R>::Desc::D_ON_UPPER:
   case SPxBasisBase<R>::Desc::D_ON_LOWER:
   case SPxBasisBase<R>::Desc::D_ON_BOTH:
   case SPxBasisBase<R>::Desc::D_UNDEFINED:
      if(rep() == ROW && theShift > 0.0)
         forceRecompNonbasicValue();

      stat = this->dualColStatus(i);
      break;

   default:
      throw SPxInternalCodeException("XCHANG02 This should never happen.");
   }

   MSG_DEBUG(std::cout << " -> " << stat << std::endl;);

   // we only need to update the nonbasic value in column representation (see nonbasicValue() for comparison/explanation)
   if(rep() == COLUMN)
      updateNonbasicValue(objChange);
}

template <class R>
void SPxSolverBase<R>::changeUpper(const VectorBase<R>& newUpper, bool scale)
{
   // we better recompute the nonbasic value when changing all upper bounds
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeUpper(newUpper, scale);

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
   {
      for(int i = 0; i < newUpper.dim(); ++i)
         changeUpperStatus(i, this->upper(i));

      unInit();
   }
}

template <class R>
void SPxSolverBase<R>::changeUpper(int i, const R& newUpper, bool scale)
{
   if(newUpper != (scale ? this->upperUnscaled(i) : this->upper(i)))
   {
      forceRecompNonbasicValue();

      R oldUpper = this->upper(i);
      SPxLPBase<R>::changeUpper(i, newUpper, scale);

      if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
      {
         changeUpperStatus(i, this->upper(i), oldUpper);
         unInit();
      }
   }
}

template <class R>
void SPxSolverBase<R>::changeBounds(const VectorBase<R>& newLower, const VectorBase<R>& newUpper,
                                    bool scale)
{
   changeLower(newLower, scale);
   changeUpper(newUpper, scale);
}

template <class R>
void SPxSolverBase<R>::changeBounds(int i, const R& newLower, const R& newUpper, bool scale)
{
   changeLower(i, newLower, scale);

   if(EQ(newLower, newUpper))
      changeUpper(i, newLower, scale);
   else
      changeUpper(i, newUpper, scale);

}

template <class R>
void SPxSolverBase<R>::changeLhsStatus(int i, R newLhs, R oldLhs)
{
   typename SPxBasisBase<R>::Desc::Status& stat      = this->desc().rowStatus(i);
   R                    currRhs   = this->rhs(i);
   R                    objChange = 0.0;

   MSG_DEBUG(std::cout << "DCHANG03 changeLhsStatus()  : row " << i
             << ": " << stat;)

   switch(stat)
   {
   case SPxBasisBase<R>::Desc::P_ON_LOWER:
      if(newLhs <= R(-infinity))
      {
         if(currRhs >= R(infinity))
         {
            stat = SPxBasisBase<R>::Desc::P_FREE;

            if(m_nonbasicValueUpToDate && rep() == COLUMN)
               objChange = -theURbound[i] * oldLhs;
         }
         else
         {
            stat = SPxBasisBase<R>::Desc::P_ON_UPPER;

            if(m_nonbasicValueUpToDate && rep() == COLUMN)
               objChange = (theLRbound[i] * currRhs) - (theURbound[i] * oldLhs);
         }
      }
      else if(EQ(newLhs, currRhs))
      {
         stat = SPxBasisBase<R>::Desc::P_FIXED;

         if(m_nonbasicValueUpToDate && rep() == COLUMN)
            objChange = this->maxRowObj(i) * (newLhs - oldLhs);
      }
      else if(m_nonbasicValueUpToDate && rep() == COLUMN)
         objChange = theURbound[i] * (newLhs - oldLhs);

      break;

   case SPxBasisBase<R>::Desc::P_ON_UPPER:
      if(EQ(newLhs, currRhs))
         stat = SPxBasisBase<R>::Desc::P_FIXED;

      break;

   case SPxBasisBase<R>::Desc::P_FREE:
      if(newLhs > R(-infinity))
      {
         stat = SPxBasisBase<R>::Desc::P_ON_LOWER;

         if(m_nonbasicValueUpToDate && rep() == COLUMN)
            objChange = theURbound[i] * newLhs;
      }

      break;

   case SPxBasisBase<R>::Desc::P_FIXED:
      if(NE(newLhs, currRhs))
      {
         stat = SPxBasisBase<R>::Desc::P_ON_UPPER;

         if(isInitialized())
            theLRbound[i] = this->maxRowObj(i);
      }

      break;

   case SPxBasisBase<R>::Desc::D_FREE:
   case SPxBasisBase<R>::Desc::D_ON_UPPER:
   case SPxBasisBase<R>::Desc::D_ON_LOWER:
   case SPxBasisBase<R>::Desc::D_ON_BOTH:
   case SPxBasisBase<R>::Desc::D_UNDEFINED:
      if(rep() == ROW && theShift > 0.0)
         forceRecompNonbasicValue();

      stat = this->dualRowStatus(i);
      break;

   default:
      throw SPxInternalCodeException("XCHANG03 This should never happen.");
   }

   MSG_DEBUG(std::cout << " -> " << stat << std::endl;)

   // we only need to update the nonbasic value in column representation (see nonbasicValue() for comparison/explanation)
   if(rep() == COLUMN)
      updateNonbasicValue(objChange);
}

template <class R>
void SPxSolverBase<R>::changeLhs(const VectorBase<R>& newLhs, bool scale)
{
   // we better recompute the nonbasic value when changing all lhs
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeLhs(newLhs, scale);

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
   {
      for(int i = 0; i < this->nRows(); ++i)
         changeLhsStatus(i, this->lhs(i));

      unInit();
   }
}

template <class R>
void SPxSolverBase<R>::changeLhs(int i, const R& newLhs, bool scale)
{
   if(newLhs != (scale ? this->lhsUnscaled(i) : this->lhs(i)))
   {
      forceRecompNonbasicValue();

      R oldLhs = this->lhs(i);
      SPxLPBase<R>::changeLhs(i, newLhs, scale);

      if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
      {
         changeLhsStatus(i, this->lhs(i), oldLhs);
         unInit();
      }
   }
}

template <class R>
void SPxSolverBase<R>::changeRhsStatus(int i, R newRhs, R oldRhs)
{
   typename SPxBasisBase<R>::Desc::Status& stat      = this->desc().rowStatus(i);
   R                    currLhs   = this->lhs(i);
   R                    objChange = 0.0;

   MSG_DEBUG(std::cout << "DCHANG04 changeRhsStatus()  : row " << i
             << ": " << stat;)

   switch(stat)
   {
   case SPxBasisBase<R>::Desc::P_ON_UPPER:
      if(newRhs >= R(infinity))
      {
         if(currLhs <= R(-infinity))
         {
            stat = SPxBasisBase<R>::Desc::P_FREE;

            if(m_nonbasicValueUpToDate && rep() == COLUMN)
               objChange = -theLRbound[i] * oldRhs;
         }
         else
         {
            stat = SPxBasisBase<R>::Desc::P_ON_LOWER;

            if(m_nonbasicValueUpToDate && rep() == COLUMN)
               objChange = (theURbound[i] * currLhs) - (theLRbound[i] * oldRhs);
         }
      }
      else if(EQ(newRhs, currLhs))
      {
         stat = SPxBasisBase<R>::Desc::P_FIXED;

         if(m_nonbasicValueUpToDate && rep() == COLUMN)
            objChange = this->maxRowObj(i) * (newRhs - oldRhs);
      }
      else if(m_nonbasicValueUpToDate && rep() == COLUMN)
         objChange = theLRbound[i] * (newRhs - oldRhs);

      break;

   case SPxBasisBase<R>::Desc::P_ON_LOWER:
      if(EQ(newRhs, currLhs))
         stat = SPxBasisBase<R>::Desc::P_FIXED;

      break;

   case SPxBasisBase<R>::Desc::P_FREE:
      if(newRhs < R(infinity))
      {
         stat = SPxBasisBase<R>::Desc::P_ON_UPPER;

         if(m_nonbasicValueUpToDate && rep() == COLUMN)
            objChange = theLRbound[i] * newRhs;
      }

      break;

   case SPxBasisBase<R>::Desc::P_FIXED:
      if(NE(newRhs, currLhs))
      {
         stat = SPxBasisBase<R>::Desc::P_ON_LOWER;

         if(isInitialized())
            theURbound[i] = this->maxRowObj(i);
      }

      break;

   case SPxBasisBase<R>::Desc::D_FREE:
   case SPxBasisBase<R>::Desc::D_ON_UPPER:
   case SPxBasisBase<R>::Desc::D_ON_LOWER:
   case SPxBasisBase<R>::Desc::D_ON_BOTH:
   case SPxBasisBase<R>::Desc::D_UNDEFINED:
      if(rep() == ROW && theShift > 0.0)
         forceRecompNonbasicValue();

      stat = this->dualRowStatus(i);
      break;

   default:
      throw SPxInternalCodeException("XCHANG04 This should never happen.");
   }

   MSG_DEBUG(std::cout << " -> " << stat << std::endl;)

   // we only need to update the nonbasic value in column representation (see nonbasicValue() for comparison/explanation)
   if(rep() == COLUMN)
      updateNonbasicValue(objChange);
}


template <class R>
void SPxSolverBase<R>::changeRhs(const VectorBase<R>& newRhs, bool scale)
{
   // we better recompute the nonbasic value when changing all rhs
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeRhs(newRhs, scale);

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
   {
      for(int i = 0; i < this->nRows(); ++i)
         changeRhsStatus(i, this->rhs(i));

      unInit();
   }
}

template <class R>
void SPxSolverBase<R>::changeRhs(int i, const R& newRhs, bool scale)
{
   if(newRhs != (scale ? this->rhsUnscaled(i) : this->rhs(i)))
   {
      forceRecompNonbasicValue();

      R oldRhs = this->rhs(i);
      SPxLPBase<R>::changeRhs(i, newRhs, scale);

      if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
      {
         changeRhsStatus(i, this->rhs(i), oldRhs);
         unInit();
      }
   }
}

template <class R>
void SPxSolverBase<R>::changeRange(const VectorBase<R>& newLhs, const VectorBase<R>& newRhs,
                                   bool scale)
{
   // we better recompute the nonbasic value when changing all ranges
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeLhs(newLhs, scale);
   SPxLPBase<R>::changeRhs(newRhs, scale);

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
   {
      for(int i = this->nRows() - 1; i >= 0; --i)
      {
         changeLhsStatus(i, this->lhs(i));
         changeRhsStatus(i, this->rhs(i));
      }

      unInit();
   }
}

template <class R>
void SPxSolverBase<R>::changeRange(int i, const R& newLhs, const R& newRhs, bool scale)
{
   R oldLhs = this->lhs(i);
   R oldRhs = this->rhs(i);

   SPxLPBase<R>::changeLhs(i, newLhs, scale);

   if(EQ(newLhs, newRhs))
      SPxLPBase<R>::changeRhs(i, newLhs, scale);
   else
      SPxLPBase<R>::changeRhs(i, newRhs, scale);

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
   {
      changeLhsStatus(i, this->lhs(i), oldLhs);
      changeRhsStatus(i, this->rhs(i), oldRhs);
      unInit();
   }
}

template <class R>
void SPxSolverBase<R>::changeRow(int i, const LPRowBase<R>& newRow, bool scale)
{
   forceRecompNonbasicValue();

   SPxLPBase<R>::changeRow(i, newRow, scale);

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
      SPxBasisBase<R>::changedRow(i);

   unInit();
}

template <class R>
void SPxSolverBase<R>::changeCol(int i, const LPColBase<R>& newCol, bool scale)
{
   if(i < 0)
      return;

   forceRecompNonbasicValue();

   SPxLPBase<R>::changeCol(i, newCol, scale);

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
      SPxBasisBase<R>::changedCol(i);

   unInit();
}

template <class R>
void SPxSolverBase<R>::changeElement(int i, int j, const R& val, bool scale)
{
   if(i < 0 || j < 0)
      return;

   forceRecompNonbasicValue();

   SPxLPBase<R>::changeElement(i, j, val, scale);

   if(SPxBasisBase<R>::status() > SPxBasisBase<R>::NO_PROBLEM)
      SPxBasisBase<R>::changedElement(i, j);

   unInit();
}

template <class R>
void SPxSolverBase<R>::changeSense(typename SPxLPBase<R>::SPxSense sns)
{

   SPxLPBase<R>::changeSense(sns);
   unInit();
}
} // namespace soplex
