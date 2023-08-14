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
#include "soplex/vector.h"

namespace soplex
{

template <class R>
void SPxSumST<R>::setupWeights(SPxSolverBase<R>& base)
{
   int count;
   int i;
   R x;
   VectorBase<R> work, delta, rowLen;

   assert(base.nRows() > 0);
   assert(base.nCols() > 0);

   rowLen.reDim(base.nRows(), true);
   work.reDim(base.nCols(), true);
   delta.reDim(base.nCols(), true);

   R* wrk = work.get_ptr();
   const R* lhs = base.lhs().get_const_ptr();
   const R* rhs = base.rhs().get_const_ptr();
   const R* up = base.upper().get_const_ptr();
   const R* low = base.lower().get_const_ptr();

   for(i = base.nRows(); --i >= 0;)
   {
      rowLen[i] = base.rowVector(i).length2();

      if(lhs[i] > 0)
         delta.multAdd(lhs[i] / rowLen[i], base.rowVector(i));
      else if(rhs[i] < 0)
         delta.multAdd(rhs[i] / rowLen[i], base.rowVector(i));
   }

   for(count = 0;; count++)
   {
      work += delta;

      for(i = base.nCols(); --i >= 0;)
      {
         if(wrk[i] > up[i])
            wrk[i] = up[i];

         if(wrk[i] < low[i])
            wrk[i] = low[i];
      }

      //      std::cout << -(work * base.maxObj()) << std::endl;
      if(count >= 12)
         break;

      delta.clear();

      for(i = base.nRows(); --i >= 0;)
      {
         x = base.rowVector(i) * work;

         if(lhs[i] > x)
            delta.multAdd((lhs[i] - x) / rowLen[i], base.rowVector(i));
         else if(rhs[i] < x)
            delta.multAdd((rhs[i] - x) / rowLen[i], base.rowVector(i));
      }
   }

   this->primal(work);
   SPxVectorST<R>::setupWeights(base);
}
} // namespace soplex
