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

/**@file  lpcolbase.h
 * @brief LP column.
 */
#ifndef _LPCOLBASE_H_
#define _LPCOLBASE_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/basevectors.h"

namespace soplex
{
/**@brief   LP column.
 * @ingroup Algo
 *
 *  Class LPColBase provides a datatype for storing the column of an LP a the form similar to
 *  \f[
 *     \begin{array}{rl}
 *        \hbox{max}  & c^T x         \\
 *        \hbox{s.t.} & Ax \le b      \\
 *                    & l \le x \le u
 *     \end{array}
 *  \f]
 *  Hence, an LPColBase consists of an objective value, a column DSVector and an upper and lower bound to the corresponding
 *  variable, which may include \f$\pm\infty\f$. However, it depends on the LP code to use, what values are actually
 *  treated as \f$\infty\f$.
 */
template < class R >
class LPColBase
{
   template < class S > friend class LPColBase;

private:

   // ------------------------------------------------------------------------------------------------------------------
   /**@name Data */
   ///@{

   R up;                   ///< upper bound
   R low;                  ///< lower bound
   R object;               ///< objective value
   DSVectorBase<R> vec;    ///< the column vector

   ///@}

public:

   // ------------------------------------------------------------------------------------------------------------------
   /**@name Construction / destruction */
   ///@{

   /// Default constructor.
   /** Construct LPColBase with a column vector ready for taking \p defDim nonzeros.
    */
   explicit LPColBase<R>(int defDim = 0)
      : up(R(infinity)), low(0), object(0), vec(defDim)
   {
      assert(isConsistent());
   }

   /// Initializing constructor.
   /*  Construct LPColBase with the given objective value \p obj, a column %vector \p vec, upper bound \p upper and
    *  lower bound \p lower.
    */
   LPColBase<R>(const R& p_obj, const SVectorBase<R>& p_vector, const R& p_upper, const R& p_lower)
      : up(p_upper), low(p_lower), object(p_obj), vec(p_vector)
   {
      assert(isConsistent());
   }

   /// Assignment operator.
   LPColBase<R>& operator=(const LPColBase<R>& old)
   {
      if(this != &old)
      {
         up = old.up;
         low = old.low;
         object = old.object;
         vec = old.vec;
      }

      assert(isConsistent());
   }

   /// Copy constructor.
   LPColBase<R>(const LPColBase<R>& old)
      : up(old.up), low(old.low), object(old.object), vec(old.vec)
   {
      assert(isConsistent());
   }

   /// Copy constructor.
   template < class S >
   LPColBase<R>(const LPColBase<S>& old)
      : up(old.up), low(old.low), object(old.object), vec(old.vec)
   {
      assert(isConsistent());
   }

   /// Destructor.
   ~LPColBase()
   {}

   ///@}

   // ------------------------------------------------------------------------------------------------------------------
   /**@name Access / modification */
   ///@{

   /// Gets objective value.
   R obj() const
   {
      return object;
   }

   /// Sets objective value.
   void setObj(const R& p_object)
   {
      object = p_object;
   }

   /// Gets upper bound.
   R upper() const
   {
      return up;
   }

   /// Sets upper bound.
   void setUpper(const R& p_up)
   {
      up = p_up;
   }

   /// Gets lower bound.
   R lower() const
   {
      return low;
   }
   /// Sets lower bound.
   void setLower(const R& p_low)
   {
      low = p_low;
   }

   /// Gets constraint column vector.
   const SVectorBase<R>& colVector() const
   {
      return vec;
   }

   /// Sets constraint column vector.
   void setColVector(const SVectorBase<R>& p_vec)
   {
      vec = p_vec;
   }

   ///@}

   // ------------------------------------------------------------------------------------------------------------------
   /**@name Consistency check */
   ///@{

   /// Checks consistency.
   bool isConsistent() const
   {
#ifdef ENABLE_CONSISTENCY_CHECKS
      return vec.isConsistent();
#else
      return true;
#endif
   }

   ///@}
};
} // namespace soplex
#endif // _LPCOLBASE_H_
