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

/**@file  spxdefines.cpp
 * @brief Debugging, floating point type and parameter definitions.
 */
#include "assert.h"
#include "soplex/spxdefines.h"
#include "soplex/spxout.h"
#include "soplex/rational.h"

namespace soplex
{
// Overloaded EQ function
bool EQ(int a, int b)
{
   return (a == b);
}

THREADLOCAL const Real infinity                 = DEFAULT_INFINITY;

THREADLOCAL Real Param::s_epsilon               = DEFAULT_EPS_ZERO;

THREADLOCAL Real Param::s_epsilon_factorization = DEFAULT_EPS_FACTOR;

THREADLOCAL Real Param::s_epsilon_update        = DEFAULT_EPS_UPDATE;

THREADLOCAL Real Param::s_epsilon_pivot         = DEFAULT_EPS_PIVOT;

bool msginconsistent(const char* name, const char* file, int line)
{
   assert(name != 0);
   assert(file != 0);
   assert(line >= 0);

   MSG_ERROR(std::cerr << file << "(" << line << ") "
             << "Inconsistency detected in " << name << std::endl;)

   return 0;
}


Real Param::epsilon()
{
   return (s_epsilon);
}

void Param::setEpsilon(Real eps)
{
   s_epsilon = eps;
}


Real Param::epsilonFactorization()
{
   return s_epsilon_factorization;
}

void Param::setEpsilonFactorization(Real eps)
{
   s_epsilon_factorization = eps;
}


Real Param::epsilonUpdate()
{
   return s_epsilon_update;
}

void Param::setEpsilonUpdate(Real eps)
{
   s_epsilon_update = eps;
}

Real Param::epsilonPivot()
{
   return s_epsilon_pivot;
}

void Param::setEpsilonPivot(Real eps)
{
   s_epsilon_pivot = eps;
}

} // namespace soplex
