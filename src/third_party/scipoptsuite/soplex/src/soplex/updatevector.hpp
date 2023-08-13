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

namespace soplex
{

template <class R>
UpdateVector<R>& UpdateVector<R>::operator=(const UpdateVector<R>& rhs)
{
   if(this != &rhs)
   {
      theval   = rhs.theval;
      thedelta = rhs.thedelta;
      VectorBase<R>::operator=(rhs);

      assert(UpdateVector<R>::isConsistent());
   }

   return *this;
}

template <class R>
UpdateVector<R>::UpdateVector(const UpdateVector<R>& base)
   : VectorBase<R>(base)
   , theval(base.theval)
   , thedelta(base.thedelta)
{
   assert(UpdateVector<R>::isConsistent());
}

template <class R>
bool UpdateVector<R>::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS

   if(this->dim() != thedelta.dim())
      return MSGinconsistent("UpdateVector");

   return VectorBase<R>::isConsistent() && thedelta.isConsistent();
#else
   return true;
#endif
}
} // namespace soplex
