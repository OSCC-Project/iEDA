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

/**@file  spxfastrt.h
 * @brief Fast shifting ratio test.
 */
#ifndef _SPXFASTRT_H_
#define _SPXFASTRT_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/spxratiotester.h"

namespace soplex
{

/**@brief   Fast shifting ratio test.
   @ingroup Algo

   Class SPxFastRT is an implementation class of SPxRatioTester providing
   fast and stable ratio test. Stability is achieved by allowing some
   infeasibility to ensure numerical stability such as the Harris procedure.
   Performance is achieved by skipping the second phase if the first phase
   already shows a stable enough pivot.

   See SPxRatioTester for a class documentation.
*/
template <class R>
class SPxFastRT : public SPxRatioTester<R>
{
protected:
   //-------------------------------------
   /**@name Data */
   ///@{
   /// parameter for computing minimum stability requirement
   R minStab;
   /// |value| < epsilon is considered 0.
   R epsilon;
   /// currently allowed infeasibility.
   R fastDelta;
   /// flag used in methods minSelect/maxSelect to retrieve correct basis status
   bool iscoid;
   ///@}

   //-------------------------------------
   /**@name Private helpers */
   ///@{
   /// resets tolerances (epsilon).
   void resetTols();
   /// relaxes stability requirements.
   void relax();
   /// tightens stability requirements.
   void tighten();
   /// Compute stability requirement
   R minStability(R maxabs);

   /// Max phase 1 value.
   /** Computes the maximum value \p val that could be used for updating \p update
       such that it would still fulfill the upper and lower bounds \p upBound and
       \p lowBound, respectively, within #delta. Return value is the index where the
       maximum value is encountered. At the same time the maximum absolute value
       of \p update.delta() is computed and returned in \p maxabs. Internally all
       loops are started at \p start and incremented by \p incr.
    */
   int maxDelta(R& val, R& maxabs, UpdateVector<R>& update,
                const VectorBase<R>& lowBound, const VectorBase<R>& upBound, int start, int incr) const;

   ///
   int maxDelta(R& val, R& maxabs);

   ///
   SPxId maxDelta(int& nr, R& val, R& maxabs);

   /// Min phase 1 value.
   /** Computes the minimum value \p val that could be used for updating \p update
       such that it would still fulfill the upper and lower bounds \p upBound and
       \p lowBound, respectively, within #delta. Return value is the index where the
       minimum value is encountered. At the same time the maximum absolute value
       of \p update.delta() is computed and returned in \p maxabs. Internally all
       loops are started at \p start and incremented by \p incr.
   */
   int minDelta(R& val, R& maxabs, UpdateVector<R>& update,
                const VectorBase<R>& lowBound, const VectorBase<R>& upBound, int start, int incr) const;

   ///
   int minDelta(R& val, R& maxabs);

   ///
   SPxId minDelta(int& nr, R& val, R& maxabs);

   /// selects stable index for maximizing ratio test.
   /** Selects from all update values \p val < \p max the one with the largest
       value of \p upd.delta() which must be greater than \p stab and is
       returned in \p stab. The index is returned as well as the corresponding
       update value \p val. Internally all loops are started at \p start and
       incremented by \p incr.
   */
   int maxSelect(R& val, R& stab, R& best, R& bestDelta,
                 R max, const UpdateVector<R>& upd, const VectorBase<R>& low,
                 const VectorBase<R>& up, int start = 0, int incr = 1) const;
   ///
   int maxSelect(R& val, R& stab, R& bestDelta, R max);
   ///
   SPxId maxSelect(int& nr, R& val, R& stab,
                   R& bestDelta, R max);

   /// selects stable index for minimizing ratio test.
   /** Select from all update values \p val > \p max the one with the largest
       value of \p upd.delta() which must be greater than \p stab and is
       returned in \p stab. The index is returned as well as the corresponding
       update value \p val. Internally all loops are started at \p start and
       incremented by \p incr.
   */
   int minSelect(R& val, R& stab, R& best, R& bestDelta,
                 R max, const UpdateVector<R>& upd, const VectorBase<R>& low,
                 const VectorBase<R>& up, int start = 0, int incr = 1) const;
   ///
   int minSelect(R& val, R& stab,
                 R& bestDelta, R max);
   ///
   SPxId minSelect(int& nr, R& val, R& stab,
                   R& bestDelta, R max);

   /// tests for stop after phase 1.
   /** Tests whether a shortcut after phase 1 is feasible for the
       selected leave pivot. In this case return the update value in \p sel.
   */
   bool minShortLeave(R& sel, int leave, R maxabs);
   ///
   bool maxShortLeave(R& sel, int leave, R maxabs);

   /// numerical stability tests.
   /** Tests whether the selected leave index needs to be discarded (and do so)
       and the ratio test is to be recomputed.
       If \p polish is set to true no shifts are applied.
   */
   bool minReLeave(R& sel, int leave, R maxabs, bool polish = false);
   ///
   bool maxReLeave(R& sel, int leave, R maxabs, bool polish = false);

   /// numerical stability check.
   /** Tests whether the selected enter \p id needs to be discarded (and do so)
       and the ratio test is to be recomputed.
   */
   bool minReEnter(R& sel, R maxabs, const SPxId& id, int nr, bool polish = false);
   ///
   bool maxReEnter(R& sel, R maxabs, const SPxId& id, int nr, bool polish = false);

   /// Tests and returns whether a shortcut after phase 1 is feasible for the
   /// selected enter pivot.
   bool shortEnter(const SPxId& enterId, int nr, R max, R maxabs) const;
   ///@}

public:

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor
   SPxFastRT()
      : SPxRatioTester<R>("Fast")
      , minStab(DEFAULT_BND_VIOL)
      , epsilon(DEFAULT_EPS_ZERO)
      , fastDelta(DEFAULT_BND_VIOL)
      , iscoid(false)
   {}
   /// copy constructor
   SPxFastRT(const SPxFastRT& old)
      : SPxRatioTester<R>(old)
      , minStab(old.minStab)
      , epsilon(old.epsilon)
      , fastDelta(old.fastDelta)
      , iscoid(false)
   {}
   /// assignment operator
   SPxFastRT& operator=(const SPxFastRT& rhs)
   {
      if(this != &rhs)
      {
         SPxRatioTester<R>::operator=(rhs);
         minStab = rhs.minStab;
         epsilon = rhs.epsilon;
         fastDelta = rhs.fastDelta;
         iscoid = false;
      }

      return *this;
   }
   /// bound flipping constructor
   SPxFastRT(const char* name)
      : SPxRatioTester<R>(name)
      , minStab(DEFAULT_BND_VIOL)
      , epsilon(DEFAULT_EPS_ZERO)
      , fastDelta(DEFAULT_BND_VIOL)
      , iscoid(false)
   {}
   /// destructor
   virtual ~SPxFastRT()
   {}
   /// clone function for polymorphism
   inline virtual SPxRatioTester<R>* clone() const
   {
      return new SPxFastRT(*this);
   }
   ///@}

   //-------------------------------------
   /**@name Access / modification */
   ///@{
   ///
   virtual void load(SPxSolverBase<R>* solver);
   ///
   virtual int selectLeave(R& val, R, bool polish = false);
   ///
   virtual SPxId selectEnter(R& val, int, bool polish = false);
   ///
   virtual void setType(typename SPxSolverBase<R>::Type type);
   ///
   virtual void setDelta(R newDelta)
   {
      if(newDelta <= DEFAULT_EPS_ZERO)
         newDelta = DEFAULT_EPS_ZERO;

      this->delta = newDelta;
      fastDelta = newDelta;
   }
   ///
   virtual R getDelta()
   {
      return fastDelta;
   }
   ///@}
};
} // namespace soplex

#include "spxfastrt.hpp"

#endif // _SPXFASTRT_H_
