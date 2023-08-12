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
bool SPxHybridPR<R>::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS

   if(this->thesolver != 0 &&
         (this->thesolver != steep.solver() ||
          this->thesolver != devex.solver() ||
          this->thesolver != parmult.solver()))
      return MSGinconsistent("SPxHybridPR");

   return steep.isConsistent()
          && devex.isConsistent()
          && parmult.isConsistent();
#else
   return true;
#endif
}

template <class R>
void SPxHybridPR<R>::load(SPxSolverBase<R>* p_solver)
{
   steep.load(p_solver);
   devex.load(p_solver);
   parmult.load(p_solver);
   this->thesolver = p_solver;
   setType(p_solver->type());
}

template <class R>
void SPxHybridPR<R>::clear()
{
   steep.clear();
   devex.clear();
   parmult.clear();
   this->thesolver = 0;
}

template <class R>
void SPxHybridPR<R>::setEpsilon(R eps)
{
   steep.setEpsilon(eps);
   devex.setEpsilon(eps);
   parmult.setEpsilon(eps);
}

template <class R>
void SPxHybridPR<R>::setType(typename SPxSolverBase<R>::Type tp)
{
   if(tp == SPxSolverBase<R>::LEAVE)
   {
      thepricer = &steep;
      this->thesolver->setPricing(SPxSolverBase<R>::FULL);
   }
   else
   {
      if(this->thesolver->dim() > hybridFactor * this->thesolver->coDim())
      {
         /**@todo I changed from devex to steepest edge pricing here
          *       because of numerical difficulties, this should be
          *       investigated.
          */
         // thepricer = &devex;
         thepricer = &steep;
         this->thesolver->setPricing(SPxSolverBase<R>::FULL);
      }
      else
      {
         thepricer = &parmult;
         this->thesolver->setPricing(SPxSolverBase<R>::PARTIAL);
      }
   }

   MSG_INFO1((*this->thesolver->spxout), (*this->thesolver->spxout) << "IPRHYB01 switching to "
             << thepricer->getName() << std::endl;)

   thepricer->setType(tp);
}

template <class R>
void SPxHybridPR<R>::setRep(typename SPxSolverBase<R>::Representation rep)
{
   steep.setRep(rep);
   devex.setRep(rep);
   parmult.setRep(rep);
}

template <class R>
int SPxHybridPR<R>::selectLeave()
{
   return thepricer->selectLeave();
}

template <class R>
void SPxHybridPR<R>::left4(int n, SPxId id)
{
   thepricer->left4(n, id);
}

template <class R>
SPxId SPxHybridPR<R>::selectEnter()
{
   return thepricer->selectEnter();
}

template <class R>
void SPxHybridPR<R>::entered4(SPxId id, int n)
{
   thepricer->entered4(id, n);
}

template <class R>
void SPxHybridPR<R>::addedVecs(int n)
{
   steep.addedVecs(n);
   devex.addedVecs(n);
   parmult.addedVecs(n);
}

template <class R>
void SPxHybridPR<R>::addedCoVecs(int n)
{
   steep.addedCoVecs(n);
   devex.addedCoVecs(n);
   parmult.addedCoVecs(n);
}

} // namespace soplex
