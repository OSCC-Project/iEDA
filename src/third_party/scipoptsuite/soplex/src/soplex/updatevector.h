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

/**@file  updatevector.h
 * @brief Dense VectorBase<R> with semi-sparse VectorBase<R> for updates
 */

#ifndef _UPDATEVECTOR_H_
#define _UPDATEVECTOR_H_

#include <assert.h>


#include "soplex/spxdefines.h"
#include "soplex/ssvector.h"

namespace soplex
{

/**@brief   Dense Vector with semi-sparse Vector for updates
   @ingroup Algebra

    In many algorithms vectors are updated in every iteration, by
    adding a multiple of another VectorBase<R> to it, i.e., given a VectorBase<R> \c
    x, a scalar \f$\alpha\f$ and another VectorBase<R> \f$\delta\f$, the
    update to \c x constists of substituting it by \f$x \leftarrow x +
    \alpha\cdot\delta\f$.

    While the update itself can easily be expressed with methods of
    the class VectorBase<R>, it is often desirable to save the last update
    VectorBase<R> \f$\delta\f$ and value \f$\alpha\f$. This is provided by
    class UpdateVector<R>.

    UpdateVectors are derived from VectorBase<R> and provide additional
    methods for saving and setting the multiplicator \f$\alpha\f$ and
    the update Vector \f$\delta\f$. Further, it allows for efficient
    sparse updates, by providing an IdxSet idx() containing the
    nonzero indices of \f$\delta\f$.
*/
template <class R>
class UpdateVector : public VectorBase<R>
{
private:

   //------------------------------------
   /**@name Data */
   ///@{
   R     theval;      ///< update multiplicator
   SSVectorBase<R> thedelta;    ///< update vector
   ///@}

public:

   //------------------------------------
   /**@name Constructors / destructors */
   ///@{
   /// default constructor.
   explicit
   UpdateVector<R>(int p_dim /*=0*/, R p_eps /*=1e-16*/)
      : VectorBase<R> (p_dim)
      , theval(0)
      , thedelta(p_dim, p_eps)
   {
      assert(isConsistent());
   }
   ///
   ~UpdateVector<R>()
   {}
   /// copy constructor
   UpdateVector<R>(const UpdateVector<R>&);
   /// assignment from VectorBase<R>
   UpdateVector<R>& operator=(const VectorBase<R>& rhs)
   {
      if(this != & rhs)
         VectorBase<R>::operator=(rhs);

      assert(isConsistent());

      return *this;
   }

   /// assignment
   UpdateVector<R>& operator=(const UpdateVector<R>& rhs);
   ///@}

   //------------------------------------
   /**@name Access */
   ///@{
   /// update multiplicator \f$\alpha\f$, writeable
   R& value()
   {
      return theval;
   }
   /// update multiplicator \f$\alpha\f$
   R value() const
   {
      return theval;
   }

   /// update VectorBase<R> \f$\delta\f$, writeable
   SSVectorBase<R>& delta()
   {
      return thedelta;
   }
   /// update VectorBase<R> \f$\delta\f$
   const SSVectorBase<R>& delta() const
   {
      return thedelta;
   }

   /// nonzero indices of update VectorBase<R> \f$\delta\f$
   const IdxSet& idx() const
   {
      return thedelta.indices();
   }
   ///@}

   //------------------------------------
   /**@name Modification */
   ///@{
   /// Perform the update
   /**  Add \c value() * \c delta() to the UpdateVector<R>. Only the indices
    *  in idx() are affected. For all other indices, delta() is asumed
    *  to be 0.
    */
   void update()
   {
      this->multAdd(theval, thedelta);
   }

   /// clear VectorBase<R> and update vector
   void clear()
   {
      VectorBase<R>::clear();
      clearUpdate();
   }

   /// clear \f$\delta\f$, \f$\alpha\f$
   void clearUpdate()
   {
      thedelta.clear();
      theval = 0;
   }

   /// reset dimension
   void reDim(int newdim)
   {
      VectorBase<R>::reDim(newdim);
      thedelta.reDim(newdim);
   }
   ///@}

   //------------------------------------
   /**@name Consistency check */
   ///@{
   ///
   bool isConsistent() const;
   ///@}
};


} // namespace soplex

// General templated functions
#include "soplex/updatevector.hpp"

#endif // _UPDATEVECTOR_H_
