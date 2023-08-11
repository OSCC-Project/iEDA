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
#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/rational.h"

namespace soplex
{

/** this reconstruction routine will set x equal to the mpq vector where each component is the best rational
 *  approximation of xnum / denom with where the GCD of denominators of x is at most Dbound; it will return true on
 *  success and false if more accuracy is required: specifically if componentwise rational reconstruction does not
 *  produce such a vector
 */
static int Reconstruct(VectorRational& resvec, Integer* xnum, Integer denom, int dim,
                       const Rational& denomBoundSquared, const DIdxSet* indexSet = 0)
{
   bool rval = true;
   int done = 0;

   /* denominator must be positive */
   assert(denom > 0);
   assert(denomBoundSquared > 0);

   Integer temp = 0;
   Integer td = 0;
   Integer tn = 0;
   Integer Dbound = 0;
   Integer gcd = 1;

   Dbound = numerator(denomBoundSquared) / denominator(
               denomBoundSquared); /* this is the working bound on the denominator size */

   Dbound = (Integer) sqrt(Dbound);

   MSG_DEBUG(std::cout << "reconstructing " << dim << " dimensional vector with denominator bound " <<
             Dbound << "\n");

   /* if Dbound is below 2^24 increase it to this value, this avoids changing input vectors that have low denominator
    * because they are floating point representable
    */
   if(Dbound < 16777216)
      Dbound = 16777216;

   /* The following represent a_i, the cont frac representation and p_i/q_i, the convergents */
   Integer a0 = 0;
   Integer ai = 0;

   /* here we use p[2]=pk, p[1]=pk-1,p[0]=pk-2 and same for q */
   Integer p[3];
   Integer q[3];

   for(int c = 0; (indexSet == 0 && c < dim) || (indexSet != 0 && c < indexSet->size()); c++)
   {
      int j = (indexSet == 0 ? c : indexSet->index(c));

      assert(j >= 0);
      assert(j < dim);

      MSG_DEBUG(std::cout << "  --> component " << j << " = " << &xnum[j] << " / denom\n");

      /* if xnum =0 , then just leave x[j] as zero */
      if(xnum[j] != 0)
      {
         /* setup n and d for computing a_i the cont. frac. rep */
         tn = xnum[j];
         td = denom;

         /* divide tn and td by gcd */
         SpxGcd(temp, tn, td);
         tn = tn / temp;
         td = td / temp;

         if(td <= Dbound)
         {
            MSG_DEBUG(std::cout << "marker 1\n");

            resvec[j] = Rational(tn, td);
         }
         else
         {
            MSG_DEBUG(std::cout << "marker 2\n");

            temp = 1;

            divide_qr(tn, td, a0, temp);

            tn = td;
            td = temp;

            divide_qr(tn, td, ai, temp);
            tn = td;
            td = temp;

            p[1] = a0;
            p[2] = 1;
            p[2] += a0 * ai;

            q[1] = 1;
            q[2] = ai;

            done = 0;

            /* if q is already big, skip loop */
            if(q[2] > Dbound)
            {
               MSG_DEBUG(std::cout << "marker 3\n");
               done = 1;
            }

            int cfcnt = 2;

            while(!done && td != 0)
            {
               /* update everything: compute next ai, then update convergents */

               /* update ai */
               divide_qr(tn, td, ai, temp);
               tn = td;
               td = temp;

               /* shift p,q */
               q[0] = q[1];
               q[1] =  q[2];
               p[0] =  p[1];
               p[1] =  p[2];

               /* compute next p,q */
               p[2] =  p[0];
               p[2] += p[1] * ai;
               q[2] =  q[0];
               q[2] += q[1] * ai;

               if(q[2] > Dbound)
                  done = 1;

               cfcnt++;

               MSG_DEBUG(std::cout << "  --> convergent denominator = " << &q[2] << "\n");
            }

            assert(q[1] != 0);

            /* Assign the values */
            if(q[1] >= 0)
               resvec[j] = Rational(p[1], q[1]);
            else
               resvec[j] = Rational(-p[1], -q[1]);

            SpxGcd(temp, gcd, denominator(resvec[j]));
            gcd *= temp;

            if(gcd > Dbound)
            {
               MSG_DEBUG(std::cout << "terminating with gcd " << &gcd << " exceeding Dbound " << &Dbound << "\n");
               rval = false;
               break;
            }
         }
      }
   }

   return rval;
}

/** reconstruct a rational vector */
inline bool reconstructVector(VectorRational& input, const Rational& denomBoundSquared,
                              const DIdxSet* indexSet)
{
   std::vector<Integer> xnum(input.dim()); /* numerator of input vector */
   Integer denom = 1; /* common denominator of input vector */
   int rval = true;
   int dim;

   dim = input.dim();

   /* find common denominator */
   if(indexSet == 0)
   {
      for(int i = 0; i < dim; i++)
         SpxLcm(denom, denom, denominator(input[i]));

      for(int i = 0; i < dim; i++)
      {
         xnum[i] = denom * Integer(numerator(input[i]));
         xnum[i] = xnum[i] / Integer(denominator(input[i]));
      }
   }
   else
   {
      for(int i = 0; i < indexSet->size(); i++)
      {
         assert(indexSet->index(i) >= 0);
         assert(indexSet->index(i) < input.dim());
         SpxLcm(denom, denom, denominator(input[indexSet->index(i)]));
      }

      for(int i = 0; i < indexSet->size(); i++)
      {
         int k = indexSet->index(i);
         assert(k >= 0);
         assert(k < input.dim());
         xnum[k] = denom * Integer(numerator(input[k]));
         xnum[k] = xnum[k] / Integer(denominator(input[k]));
      }
   }

   MSG_DEBUG(std::cout << "LCM = " << mpz_get_str(0, 10, denom) << "\n");

   /* reconstruct */
   rval = Reconstruct(input, xnum.data(), denom, dim, denomBoundSquared, indexSet);

   return rval;
}



/** reconstruct a rational solution */
/**@todo make this a method of class SoPlex */
inline bool reconstructSol(SolRational& solution)
{
#if 0
   VectorRational buffer;

   if(solution.hasPrimal())
   {
      buffer.reDim((solution._primal).dim());
      solution.getPrimalSol(buffer);
      reconstructVector(buffer);
      solution._primal = buffer;

      buffer.reDim((solution._slacks).dim());
      solution.getSlacks(buffer);
      reconstructVector(buffer);
      solution._slacks = buffer;
   }

   if(solution.hasPrimalray())
   {
      buffer.reDim((solution._primalray).dim());
      solution.getPrimalray(buffer);
      reconstructVector(buffer);
      solution._primalray = buffer;
   }

   if(solution.hasDual())
   {
      buffer.reDim((solution._dual).dim());
      solution.getDualSol(buffer);
      reconstructVector(buffer);
      solution._dual = buffer;

      buffer.reDim((solution._redcost).dim());
      solution.getRedcost(buffer);
      reconstructVector(buffer);
      solution._redcost = buffer;
   }

   if(solution.hasDualfarkas())
   {
      buffer.reDim((solution._dualfarkas).dim());
      solution.getDualfarkas(buffer);
      reconstructVector(buffer);
      solution._dualfarkas = buffer;
   }

#endif
   return true;
}
} // namespace soplex
