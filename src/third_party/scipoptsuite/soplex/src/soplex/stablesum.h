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

#ifndef _SOPLEX_STABLE_SUM_H_
#define _SOPLEX_STABLE_SUM_H_
#include <type_traits>

// #define CHECK_STABLESUM  // double check the stable sum computation

namespace soplex
{

template <typename T>
class StableSum
{
   typename std::remove_const<T>::type sum;

public:
   StableSum() : sum(0) {}
   StableSum(const T& init) : sum(init) {}

   void operator+=(const T& input)
   {
      sum += input;
   }

   void operator-=(const T& input)
   {
      sum -= input;
   }

   operator typename std::remove_const<T>::type() const
   {
      return sum;
   }

};

template <>
class StableSum<double>
{
   double sum = 0;
   double c = 0;
#ifdef CHECK_STABLESUM
   double checksum = 0;
#endif

public:
   StableSum() = default;
   StableSum(double init) : sum(init), c(0) {}

   void operator+=(double input)
   {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#pragma float_control( precise, on )
#endif
#ifdef CHECK_STABLESUM
      checksum += input;
#endif
      double t = sum + input;
      double z = t - sum;
      double y = (sum - (t - z)) + (input - z);
      c += y;

      sum = t;
   }

   void operator-=(double input)
   {
      (*this) += -input;
   }

   operator double() const
   {
#ifdef CHECK_STABLESUM

      if(spxAbs(checksum - (sum + c)) >= 1e-6)
         printf("stablesum viol: %g, rel: %g, checksum: %g\n", spxAbs(checksum - (sum + c)),
                spxAbs(checksum - (sum + c)) / MAXIMUM(1.0, MAXIMUM(spxAbs(checksum), spxAbs(sum + c))), checksum);

      assert(spxAbs(checksum - (sum + c)) < 1e-6);
#endif
      return sum + c;
   }
};

/// Output operator.
template < class T >
std::ostream& operator<<(std::ostream& s, const StableSum<T>& sum)
{
   s << static_cast<T>(sum);

   return s;
}

}

#endif
