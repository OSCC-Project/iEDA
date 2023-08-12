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

/**@file   spxboundflippingrt.h
 * @brief  Bound flipping ratio test (long step dual) for SoPlex.
 * @author Matthias Miltenberger
 * @author Eva Ramlow
 */
#ifndef _SPXBOUNDFLIPPINGRT_H_
#define _SPXBOUNDFLIPPINGRT_H_


#include <assert.h>
#include "soplex/spxdefines.h"
#include "soplex/spxratiotester.h"
#include "soplex/spxfastrt.h"

namespace soplex
{

/**@brief   Bound flipping ratio test ("long step dual") for SoPlex.
   @ingroup Algo

   Class SPxBoundFlippingRT provides an implementation of the bound flipping
   ratio test as a derived class of SPxRatioTester. Dual step length is
   increased beyond some breakpoints and corresponding primal nonbasic
   variables are set to their other bound to handle the resulting dual infeasibility.

   The implementation mostly follows the papers
   - I. Maros, "A generalized dual phase-2 simplex algorithm",
     European Journal of Operational Research Vol 149, Issue 1, pp. 1-16, 2003
   - A. Koberstein, "Progress in the dual simplex algorithm for solving large scale LP problems:
     techniques for a fast and stable implementation",
     Computational Optimization and Applications Vol 41, Nr 2, pp. 185-204, 2008

   See SPxRatioTester for a class documentation.
*/
template <class R>
class SPxBoundFlippingRT : public SPxFastRT<R>
{
private:
   /**@name substructures */
   ///@{
   /** enumerator to remember which vector we have been searching to find a breakpoint
    */
   enum BreakpointSource
   {
      FVEC               = -1,
      PVEC               = 0,
      COPVEC             = 1
   };

   /** breakpoint structure
    */
   struct Breakpoint
   {
      R               val;                /**< breakpoint value (step length) */
      int                idx;                /**< index of corresponding row/column */
      BreakpointSource   src;                /**< origin of breakpoint, i.e. vector which was searched */
   };

   /** Compare class for breakpoints
    */
   struct BreakpointCompare
   {
   public:
      /** constructor
       */
      BreakpointCompare()
         : entry(0)
      {
      }

      const Breakpoint*  entry;

      R operator()(
         Breakpoint      i,
         Breakpoint      j
      ) const
      {
         return i.val - j.val;
      }
   };
   ///@}

   /**@name Data
    */
   ///@{
   bool                  enableBoundFlips;   /**< enable or disable long steps in BoundFlippingRT */
   bool                  enableRowBoundFlips;/**< enable bound flips also for row representation */
   R
   flipPotential;      /**< tracks bound flip history and decides which ratio test to use */
   int                   relax_count;        /**< count rounds of ratio test */
   Array<Breakpoint> breakpoints;        /**< array of breakpoints */
   SSVectorBase<R>
   updPrimRhs;         /**< right hand side vector of additional system to be solved after the ratio test */
   SSVectorBase<R>
   updPrimVec;         /**< allocation of memory for additional solution vector */
   ///@}

   /** store all available pivots/breakpoints in an array (positive pivot search direction) */
   void collectBreakpointsMax(
      int&               nBp,                /**< number of found breakpoints so far */
      int&               minIdx,             /**< index to current minimal breakpoint */
      const int*         idx,                /**< pointer to indices of current vector */
      int                nnz,                /**< number of nonzeros in current vector */
      const R*        upd,                /**< pointer to update values of current vector */
      const R*        vec,                /**< pointer to values of current vector */
      const R*        upp,                /**< pointer to upper bound/rhs of current vector */
      const R*        low,                /**< pointer to lower bound/lhs of current vector */
      BreakpointSource   src                 /**< type of vector (pVec or coPvec)*/
   );

   /** store all available pivots/breakpoints in an array (negative pivot search direction) */
   void collectBreakpointsMin(
      int&               nBp,                /**< number of found breakpoints so far */
      int&               minIdx,             /**< index to current minimal breakpoint */
      const int*         idx,                /**< pointer to indices of current vector */
      int                nnz,                /**< number of nonzeros in current vector */
      const R*        upd,                /**< pointer to update values of current vector */
      const R*        vec,                /**< pointer to values of current vector */
      const R*        upp,                /**< pointer to upper bound/rhs of current vector */
      const R*        low,                /**< pointer to lower bound/lhs of current vector */
      BreakpointSource   src                 /**< type of vector (pVec or coPvec)*/
   );

   /** get values for entering index and perform shifts if necessary */
   bool getData(
      R&              val,
      SPxId&             enterId,
      int                idx,
      R               stab,
      R               degeneps,
      const R*        upd,
      const R*        vec,
      const R*        low,
      const R*        upp,
      BreakpointSource   src,
      R               max
   );

   /** get values for leaving index and perform shifts if necessary */
   bool getData(
      R&              val,
      int&             leaveIdx,
      int                idx,
      R               stab,
      R               degeneps,
      const R*        upd,
      const R*        vec,
      const R*        low,
      const R*        upp,
      BreakpointSource   src,
      R               max
   );

   /** perform necessary bound flips to restore dual feasibility */
   void flipAndUpdate(
      int&               usedBp              /**< number of bounds that should be flipped */
   );

   /** comparison method for breakpoints */
   static bool isSmaller(
      Breakpoint         x,
      Breakpoint         y
   )
   {
      return (spxAbs(x.val) < spxAbs(y.val));
   };

public:

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// default constructor
   SPxBoundFlippingRT()
      : SPxFastRT<R>("Bound Flipping")
      , enableBoundFlips(true)
      , enableRowBoundFlips(false)
      , flipPotential(1)
      , relax_count(0)
      , breakpoints(10)
      , updPrimRhs(0)
      , updPrimVec(0)
   {}
   /// copy constructor
   SPxBoundFlippingRT(const SPxBoundFlippingRT& old)
      : SPxFastRT<R>(old)
      , enableBoundFlips(old.enableBoundFlips)
      , enableRowBoundFlips(old.enableRowBoundFlips)
      , flipPotential(1)
      , relax_count(0)
      , breakpoints(10)
      , updPrimRhs(0)
      , updPrimVec(0)
   {}
   /// assignment operator
   SPxBoundFlippingRT& operator=(const SPxBoundFlippingRT& rhs)
   {
      if(this != &rhs)
      {
         SPxFastRT<R>::operator=(rhs);
      }

      enableBoundFlips = rhs.enableBoundFlips;
      enableRowBoundFlips = rhs.enableRowBoundFlips;
      flipPotential = rhs.flipPotential;

      return *this;
   }
   /// destructor
   virtual ~SPxBoundFlippingRT()
   {}
   /// clone function for polymorphism
   inline virtual SPxRatioTester<R>* clone() const
   {
      return new SPxBoundFlippingRT(*this);
   }
   ///@}

   //-------------------------------------
   /**@name Select enter/leave */
   ///@{
   ///
   virtual int selectLeave(
      R&              val,
      R               enterTest,
      bool               polish = false
   );
   ///
   virtual SPxId selectEnter(
      R&              val,
      int                leaveIdx,
      bool               polish = false
   );

   void useBoundFlips(bool bf)
   {
      enableBoundFlips = bf;
   }

   void useBoundFlipsRow(bool bf)
   {
      enableRowBoundFlips = bf;
   }
   ///@}
};

} // namespace soplex

#include "spxboundflippingrt.hpp"

#endif // _SPXBOUNDFLIPPINGRT_H_
