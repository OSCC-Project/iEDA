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

namespace soplex
{

template <class R>
void SPxVectorST<R>::setupWeights(SPxSolverBase<R>& base)
{
   if(state == PVEC)
   {
      if(vec.dim() != base.nCols())
      {
         SPxWeightST<R>::setupWeights(base);
         return;
      }

      const VectorBase<R>& obj = base.maxObj();
      R eps = base.epsilon();
      R bias = 10000 * eps;
      R x, y;
      int i;

      MSG_DEBUG(std::cout << "DVECST01 colWeight[]: ";)

      for(i = base.nCols(); i--;)
      {
         x = vec[i] - base.SPxLPBase<R>::lower(i);
         y = base.SPxLPBase<R>::upper(i) - vec[i];

         if(x < y)
         {
            this->colWeight[i] = -x - bias * obj[i];
            this->colUp[i] = 0;
         }
         else
         {
            this->colWeight[i] = -y + bias * obj[i];
            this->colUp[i] = 1;
         }

         MSG_DEBUG(std::cout << this->colWeight[i] << " ";)
      }

      MSG_DEBUG(std::cout << std::endl << std::endl;)

      MSG_DEBUG(std::cout << "DVECST02 rowWeight[]: ";)

      for(i = base.nRows(); i--;)
      {
         const SVectorBase<R>& row = base.rowVector(i);
         y = vec * row;
         x = (y - base.lhs(i));
         y = (base.rhs(i) - y);

         if(x < y)
         {
            this->rowWeight[i] = -x - eps * row.size() - bias * (obj * row);
            this->rowRight[i] = 0;
         }
         else
         {
            this->rowWeight[i] = -y - eps * row.size() + bias * (obj * row);
            this->rowRight[i] = 1;
         }

         MSG_DEBUG(std::cout << this->rowWeight[i] << " ";)
      }

      MSG_DEBUG(std::cout << std::endl;)
   }

   else if(state == DVEC)
   {
      if(vec.dim() != base.nRows())
      {
         SPxWeightST<R>::setupWeights(base);
         return;
      }

      R x, y, len;
      int i, j;

      for(i = base.nRows(); i--;)
         this->rowWeight[i] += spxAbs(vec[i]);

      for(i = base.nCols(); i--;)
      {
         const SVectorBase<R>& col = base.colVector(i);

         for(y = len = 0, j = col.size(); j--;)
         {
            x = col.value(j);
            y += vec[col.index(j)] * x;
            len += x * x;
         }

         if(len > 0)
            this->colWeight[i] += spxAbs(y / len - base.maxObj(i));
      }
   }
   else
      SPxWeightST<R>::setupWeights(base);
}
} // namespace soplex
