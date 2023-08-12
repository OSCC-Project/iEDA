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

/**@file  solbase.h
 * @brief Class for storing a primal-dual solution with basis information
 */
#ifndef _SOLBASE_H_
#define _SOLBASE_H_

/* undefine SOPLEX_DEBUG flag from including files; if SOPLEX_DEBUG should be defined in this file, do so below */
#ifdef SOPLEX_DEBUG
#define SOPLEX_DEBUG_SOLBASE
#undef SOPLEX_DEBUG
#endif

#include <assert.h>
#include <string.h>
#include <math.h>
#include <iostream>

#include "soplex/basevectors.h"
#include "soplex/spxsolver.h" // needed for basis information

namespace soplex
{
/**@class   SolBase
 * @brief   Class for storing a primal-dual solution with basis information
 * @ingroup Algo
 */
template <class R>
class SolBase
{
   template <class T> friend class SoPlexBase;
   // Why do we need the following? This is at least used in the operator=
   // When Rational solution needs to be copied into Real, the private member
   // _objVal is accessed.
   template <class S> friend class SolBase;

public:
   /// is the stored solution primal feasible?
   bool isPrimalFeasible() const
   {
      return _isPrimalFeasible;
   }

   /// gets the primal solution vector; returns true on success
   bool getPrimalSol(VectorBase<R>& vector) const
   {
      vector = _primal;

      return _isPrimalFeasible;
   }

   /// gets the vector of slack values; returns true on success
   bool getSlacks(VectorBase<R>& vector) const
   {
      vector = _slacks;

      return _isPrimalFeasible;
   }

   /// is a primal unbounded ray available?
   bool hasPrimalRay() const
   {
      return _hasPrimalRay;
   }

   /// gets the primal unbounded ray if available; returns true on success
   bool getPrimalRaySol(VectorBase<R>& vector) const
   {
      if(_hasPrimalRay)
         vector = _primalRay;

      return _hasPrimalRay;
   }

   /// is a dual solution available?
   bool isDualFeasible() const
   {
      return _isDualFeasible;
   }

   /// gets the dual solution vector; returns true on success
   bool getDualSol(VectorBase<R>& vector) const
   {
      vector = _dual;

      return _isDualFeasible;
   }

   /// gets the vector of reduced cost values if available; returns true on success
   bool getRedCostSol(VectorBase<R>& vector) const
   {
      vector = _redCost;

      return _isDualFeasible;
   }

   /// is a dual farkas ray available?
   bool hasDualFarkas() const
   {
      return _hasDualFarkas;
   }

   /// gets the Farkas proof if available; returns true on success
   bool getDualFarkasSol(VectorBase<R>& vector) const
   {
      if(_hasDualFarkas)
         vector = _dualFarkas;

      return _hasDualFarkas;
   }

   /// returns total size of primal solution
   int totalSizePrimal(const int base = 2) const
   {
      int size = 0;

      if(_isPrimalFeasible)
         size += totalSizeRational(_primal.get_const_ptr(), _primal.dim(), base);

      if(_hasPrimalRay)
         size += totalSizeRational(_primalRay.get_const_ptr(), _primalRay.dim(), base);

      return size;
   }

   /// returns total size of dual solution
   int totalSizeDual(const int base = 2) const
   {
      int size = 0;

      if(_isDualFeasible)
         size += totalSizeRational(_dual.get_const_ptr(), _dual.dim(), base);

      if(_hasDualFarkas)
         size += totalSizeRational(_dualFarkas.get_const_ptr(), _dualFarkas.dim(), base);

      return size;
   }

   /// returns size of least common multiple of denominators in primal solution
   int dlcmSizePrimal(const int base = 2) const
   {
      int size = 0;

      if(_isPrimalFeasible)
         size += dlcmSizeRational(_primal.get_const_ptr(), _primal.dim(), base);

      if(_hasPrimalRay)
         size += dlcmSizeRational(_primalRay.get_const_ptr(), _primalRay.dim(), base);

      return size;
   }

   /// returns  size of least common multiple of denominators in dual solution
   int dlcmSizeDual(const int base = 2) const
   {
      int size = 0;

      if(_isDualFeasible)
         size += dlcmSizeRational(_dual.get_const_ptr(), _dual.dim(), base);

      if(_hasDualFarkas)
         size += dlcmSizeRational(_dualFarkas.get_const_ptr(), _dualFarkas.dim(), base);

      return size;
   }

   /// returns size of largest denominator in primal solution
   int dmaxSizePrimal(const int base = 2) const
   {
      int size = 0;

      if(_isPrimalFeasible)
         size += dmaxSizeRational(_primal.get_const_ptr(), _primal.dim(), base);

      if(_hasPrimalRay)
         size += dmaxSizeRational(_primalRay.get_const_ptr(), _primalRay.dim(), base);

      return size;
   }

   /// returns size of largest denominator in dual solution
   int dmaxSizeDual(const int base = 2) const
   {
      int size = 0;

      if(_isDualFeasible)
         size += dmaxSizeRational(_dual.get_const_ptr(), _dual.dim(), base);

      if(_hasDualFarkas)
         size += dmaxSizeRational(_dualFarkas.get_const_ptr(), _dualFarkas.dim(), base);

      return size;
   }

   /// invalidate solution
   void invalidate()
   {
      _isPrimalFeasible = false;
      _hasPrimalRay = false;
      _isDualFeasible = false;
      _hasDualFarkas = false;
   }

private:
   VectorBase<R> _primal;
   VectorBase<R> _slacks;
   VectorBase<R> _primalRay;
   VectorBase<R> _dual;
   VectorBase<R> _redCost;
   VectorBase<R> _dualFarkas;

   R _objVal;

   unsigned int _isPrimalFeasible: 1;
   unsigned int _hasPrimalRay: 1;
   unsigned int _isDualFeasible: 1;
   unsigned int _hasDualFarkas: 1;

   /// default constructor only for friends
   SolBase<R>()
      : _objVal(0)
   {
      invalidate();
   }

   /// assignment operator only for friends
   SolBase<R>& operator=(const SolBase<R>& sol)
   {
      if(this != &sol)
      {

         _isPrimalFeasible = sol._isPrimalFeasible;
         _primal = sol._primal;
         _slacks = sol._slacks;
         _objVal = sol._objVal;

         _hasPrimalRay = sol._hasPrimalRay;

         if(_hasPrimalRay)
            _primalRay = sol._primalRay;

         _isDualFeasible = sol._isDualFeasible;
         _dual = sol._dual;
         _redCost = sol._redCost;

         _hasDualFarkas = sol._hasDualFarkas;

         if(_hasDualFarkas)
            _dualFarkas = sol._dualFarkas;
      }

      return *this;
   }

   /// assignment operator only for friends
   template <class S>
   SolBase<R>& operator=(const SolBase<S>& sol)
   {
      if((SolBase<S>*)this != &sol)
      {

         _isPrimalFeasible = sol._isPrimalFeasible;
         _primal = sol._primal;
         _slacks = sol._slacks;

         _objVal = R(sol._objVal);

         _hasPrimalRay = sol._hasPrimalRay;

         if(_hasPrimalRay)
            _primalRay = sol._primalRay;

         _isDualFeasible = sol._isDualFeasible;
         _dual = sol._dual;
         _redCost = sol._redCost;

         _hasDualFarkas = sol._hasDualFarkas;

         if(_hasDualFarkas)
            _dualFarkas = sol._dualFarkas;
      }

      return *this;
   }

};
} // namespace soplex

/* reset the SOPLEX_DEBUG flag to its original value */
#undef SOPLEX_DEBUG
#ifdef SOPLEX_DEBUG_SOLBASE
#define SOPLEX_DEBUG
#undef SOPLEX_DEBUG_SOLBASE
#endif

#endif // _SOLBASE_H_
