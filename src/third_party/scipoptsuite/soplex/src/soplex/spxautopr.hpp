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
#include "soplex/spxout.h"

namespace soplex
{

template <class R>
void SPxAutoPR<R>::load(SPxSolverBase<R>* p_solver)
{
   steep.load(p_solver);
   devex.load(p_solver);
   this->thesolver = p_solver;
   setType(p_solver->type());
}

template <class R>
void SPxAutoPR<R>::clear()
{
   steep.clear();
   devex.clear();
   this->thesolver = nullptr;
}

template <class R>
void SPxAutoPR<R>::setEpsilon(R eps)
{
   steep.setEpsilon(eps);
   devex.setEpsilon(eps);
   this->theeps = eps;
}

template <class R>
void SPxAutoPR<R>::setType(typename SPxSolverBase<R>::Type tp)
{
   activepricer->setType(tp);
}

template <class R>
void SPxAutoPR<R>::setRep(typename SPxSolverBase<R>::Representation rep)
{
   steep.setRep(rep);
   devex.setRep(rep);
}

template <class R>
bool SPxAutoPR<R>::setActivePricer(typename SPxSolverBase<R>::Type type)
{
   // switch to steep as soon as switchIters is reached
   if(activepricer == &devex && this->thesolver->iterations() >= switchIters)
   {
      activepricer = &steep;
      activepricer->setType(type);
      return true;
   }


   // use devex for the iterations < switchIters
   else if(activepricer == &steep && this->thesolver->iterations() < switchIters)
   {
      activepricer = &devex;
      activepricer->setType(type);
      return true;
   }

   return false;
}

template <class R>
int SPxAutoPR<R>::selectLeave()
{
   if(setActivePricer(SPxSolverBase<R>::LEAVE))
      MSG_INFO1((*this->thesolver->spxout),
                (*this->thesolver->spxout) << " --- active pricer: " << activepricer->getName() << std::endl;)

      return activepricer->selectLeave();
}

template <class R>
void SPxAutoPR<R>::left4(int n, SPxId id)
{
   activepricer->left4(n, id);
}

template <class R>
SPxId SPxAutoPR<R>::selectEnter()
{
   if(setActivePricer(SPxSolverBase<R>::ENTER))
      MSG_INFO1((*this->thesolver->spxout),
                (*this->thesolver->spxout) << " --- active pricer: " << activepricer->getName() << std::endl;)

      return activepricer->selectEnter();
}

template <class R>
void SPxAutoPR<R>::entered4(SPxId id, int n)
{
   activepricer->entered4(id, n);
}

} // namespace soplex
