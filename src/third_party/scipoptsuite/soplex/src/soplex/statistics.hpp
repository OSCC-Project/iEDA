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

#include "soplex/timerfactory.h"

namespace soplex
{
/// default constructor
template <class R>
SoPlexBase<R>::Statistics::Statistics(Timer::TYPE ttype)
{
   timerType = ttype;
   readingTime = TimerFactory::createTimer(timerType);
   solvingTime = TimerFactory::createTimer(timerType);
   preprocessingTime = TimerFactory::createTimer(timerType);
   simplexTime = TimerFactory::createTimer(timerType);
   syncTime = TimerFactory::createTimer(timerType);
   transformTime = TimerFactory::createTimer(timerType);
   rationalTime = TimerFactory::createTimer(timerType);
   reconstructionTime = TimerFactory::createTimer(timerType);
   clearAllData();
}

/// copy constructor
template <class R>
SoPlexBase<R>::Statistics::Statistics(const Statistics& base)
{
   timerType = base.timerType;
   readingTime = TimerFactory::createTimer(timerType);
   solvingTime = TimerFactory::createTimer(timerType);
   preprocessingTime = TimerFactory::createTimer(timerType);
   simplexTime = TimerFactory::createTimer(timerType);
   syncTime = TimerFactory::createTimer(timerType);
   transformTime = TimerFactory::createTimer(timerType);
   rationalTime = TimerFactory::createTimer(timerType);
   reconstructionTime = TimerFactory::createTimer(timerType);
   clearAllData();
}

/// assignment operator
template <class R>
typename SoPlexBase<R>::Statistics& SoPlexBase<R>::Statistics::operator=(const Statistics& rhs)
{
   *readingTime = *(rhs.readingTime);
   *solvingTime = *(rhs.solvingTime);
   *preprocessingTime = *(rhs.preprocessingTime);
   *simplexTime = *(rhs.simplexTime);
   *syncTime = *(rhs.syncTime);
   *transformTime = *(rhs.transformTime);
   *rationalTime = *(rhs.rationalTime);
   *reconstructionTime = *(rhs.reconstructionTime);
   timerType = rhs.timerType;
   multTimeSparse = rhs.multTimeSparse;
   multTimeFull = rhs.multTimeFull;
   multTimeColwise = rhs.multTimeColwise;
   multTimeUnsetup = rhs.multTimeUnsetup;
   multSparseCalls = rhs.multSparseCalls;
   multFullCalls = rhs.multFullCalls;
   multColwiseCalls = rhs.multColwiseCalls;
   multUnsetupCalls = rhs.multUnsetupCalls;
   luFactorizationTimeReal = rhs.luFactorizationTimeReal;
   luSolveTimeReal = rhs.luSolveTimeReal;
   luFactorizationTimeRational = rhs.luFactorizationTimeRational;
   luSolveTimeRational = rhs.luSolveTimeRational;
   iterations = rhs.iterations;
   iterationsPrimal = rhs.iterationsPrimal;
   iterationsFromBasis = rhs.iterationsFromBasis;
   boundflips = rhs.boundflips;
   luFactorizationsReal = rhs.luFactorizationsReal;
   luSolvesReal = rhs.luSolvesReal;
   luFactorizationsRational = rhs.luFactorizationsRational;
   rationalReconstructions = rhs.rationalReconstructions;
   refinements = rhs.refinements;
   stallRefinements = rhs.stallRefinements;
   pivotRefinements = rhs.pivotRefinements;
   feasRefinements = rhs.feasRefinements;
   unbdRefinements = rhs.unbdRefinements;

   return *this;
}

/// clears all statistics
template <class R>
void SoPlexBase<R>::Statistics::clearAllData()
{
   readingTime->reset();
   clearSolvingData();
}

/// clears statistics on solving process
template <class R>
void SoPlexBase<R>::Statistics::clearSolvingData()
{
   solvingTime->reset();
   preprocessingTime->reset();
   simplexTime->reset();
   syncTime->reset();
   transformTime->reset();
   rationalTime->reset();
   reconstructionTime->reset();
   multTimeSparse = 0.0;
   multTimeFull = 0.0;
   multTimeColwise = 0.0;
   multTimeUnsetup = 0.0;
   multSparseCalls = 0;
   multFullCalls = 0;
   multColwiseCalls = 0;
   multUnsetupCalls = 0;
   luFactorizationTimeReal = 0.0;
   luSolveTimeReal = 0.0;
   luFactorizationTimeRational = 0.0;
   luSolveTimeRational = 0.0;
   iterations = 0;
   iterationsPrimal = 0;
   iterationsFromBasis = 0;
   iterationsPolish = 0;
   boundflips = 0;
   luFactorizationsReal = 0;
   luSolvesReal = 0;
   luFactorizationsRational = 0;
   rationalReconstructions = 0;
   refinements = 0;
   stallRefinements = 0;
   pivotRefinements = 0;
   feasRefinements = 0;
   unbdRefinements = 0;

   callsReducedProb = 0;
   iterationsInit = 0;
   iterationsRedProb = 0;
   iterationsCompProb = 0;
   numRedProbRows = 0;
   numRedProbCols = 0;
   degenPivotsPrimal = 0;
   degenPivotsDual = 0;
   degenPivotCandPrimal = 0;
   degenPivotCandDual = 0;
   sumDualDegen = 0;
   sumPrimalDegen = 0;
   decompBasisCondNum = 0;
   totalBoundViol = 0;
   totalRowViol = 0;
   maxBoundViol = 0;
   maxRowViol = 0;
   redProbStatus = 0;
   compProbStatus = 0;
   finalCompObj = 0;
   finalBasisCondition = 0;
}

/// prints statistics
template <class R>
void SoPlexBase<R>::Statistics::print(std::ostream& os)
{
   Real solTime = solvingTime->time();
   Real totTime = readingTime->time() + solTime;
   Real otherTime = solTime - syncTime->time() - transformTime->time() - preprocessingTime->time() -
                    simplexTime->time() - rationalTime->time();

   R avgPrimalDegeneracy = iterationsPrimal > 0 ? sumPrimalDegen / iterationsPrimal : 0.0;
   R avgDualDegeneracy = (iterations - iterationsPrimal) > 0 ?
                         (sumDualDegen / (iterations - iterationsPrimal)) : 0.0;

   SPxOut::setFixed(os, 2);

   os << "Total time          : " << totTime << "\n"
      << "  Reading           : " << readingTime->time() << "\n"
      << "  Solving           : " << solTime << "\n"
      << "  Preprocessing     : " << preprocessingTime->time();

   if(solTime > 0)
      os << " (" << 100 * (preprocessingTime->time() / solTime) << "% of solving time)";

   os << "\n  Simplex           : " << simplexTime->time();

   if(solTime > 0)
      os << " (" << 100 * (simplexTime->time() / solTime) << "% of solving time)";

   os << "\n  Synchronization   : " << syncTime->time();

   if(solTime > 0)
      os << " (" << 100 * (syncTime->time() / solTime) << "% of solving time)";

   os << "\n  Transformation    : " << transformTime->time();

   if(solTime > 0)
      os << " (" << 100 * transformTime->time() / solTime << "% of solving time)";

   os << "\n  Rational          : " << rationalTime->time();

   if(solTime > 0)
      os << " (" << 100 * rationalTime->time() / solTime << "% of solving time)";

   os << "\n  Other             : " << otherTime;

   if(solTime > 0)
      os << " (" << 100 * otherTime / solTime << "% of solving time)";

   os << "\nRefinements         : " << refinements << "\n"
      << "  Stalling          : " << stallRefinements << "\n"
      << "  Pivoting          : " << pivotRefinements << "\n"
      << "  Feasibility       : " << feasRefinements << "\n"
      << "  Unboundedness     : " << unbdRefinements << "\n";

   os << "Iterations          : " << iterations << "\n"
      << "  From scratch      : " << iterations - iterationsFromBasis;

   if(iterations > 0)
      os << " (" << 100 * double((iterations - iterationsFromBasis)) / double(iterations) << "%)";

   os << "\n  From basis        : " << iterationsFromBasis;

   if(iterations > 0)
      os << " (" << 100 * double(iterationsFromBasis) / double(iterations) << "%)";

   os << "\n  Primal            : " << iterationsPrimal;

   if(iterations > 0)
      os << " (" << 100 * double(iterationsPrimal) / double(iterations) << "%)";

   os << "\n  Dual              : " << iterations - iterationsPrimal - iterationsPolish;

   if(iterations > 0)
      os << " (" << 100 * double((iterations - iterationsPrimal)) / double(iterations) << "%)";

   os << "\n  Bound flips       : " << boundflips;
   os << "\n  Sol. polishing    : " << iterationsPolish;

   os << "\nLU factorizations   : " << luFactorizationsReal << "\n"
      << "  Factor. frequency : ";

   if(luFactorizationsReal > 0)
      os << double(iterations) / double(luFactorizationsReal) << " iterations per factorization\n";
   else
      os << "-\n";

   os << "  Factor. time      : " << luFactorizationTimeReal << "\n";

   os << "LU solves           : " << luSolvesReal << "\n"
      << "  Solve frequency   : ";

   if(iterations > 0)
      os << double(luSolvesReal) / double(iterations) << " solves per iteration\n";
   else
      os << "-\n";

   os << "  Solve time        : " << luSolveTimeReal << "\n";

   os << "Matrix-Vector ops   : \n"
      << "  Sparse    time    : " << multTimeSparse;

   if(solTime > 0)
      os << " (" << 100 * (multTimeSparse / solTime) << "% of solving time)";

   os << "\n            calls   : " << multSparseCalls;
   os << " (" << 100 * (multSparseCalls / (0.01 + iterations)) << "% of iterations)";
   os << "\n  Full      time    : " << multTimeFull;

   if(solTime > 0)
      os << " (" << 100 * (multTimeFull / solTime) << "% of solving time)";

   os << "\n            calls   : " << multFullCalls;
   os << " (" << 100 * (multFullCalls / (0.01 + iterations)) << "% of iterations)";
   os << "\n  Colwise   time    : " << multTimeColwise;

   if(solTime > 0)
      os << " (" << 100 * (multTimeColwise / solTime) << "% of solving time)";

   os << "\n            calls   : " << multColwiseCalls;
   os << " (" << 100 * (multColwiseCalls / (0.01 + iterations)) << "% of iterations)";
   os << "\n  Unsetup   time    : " << multTimeUnsetup;

   if(solTime > 0)
      os << " (" << 100 * (multTimeUnsetup / solTime) << "% of solving time)";

   os << "\n            calls   : " << multUnsetupCalls;
   os << " (" << 100 * (multUnsetupCalls / (0.01 + iterations)) << "% of iterations)";
   os << "\n";

   os << "Rat. factorizations : " << luFactorizationsRational << "\n"
      << "  Rat. factor. time : " << luFactorizationTimeRational << "\n"
      << "  Rat. solve time   : " << luSolveTimeRational << "\n";

   os << "Rat. reconstructions: " << rationalReconstructions << "\n"
      << "  Rat. rec. time    : " << reconstructionTime->time() << "\n";

   os << "Degeneracy          : \n";
   os << "  Primal Pivots     : " << degenPivotsPrimal << "\n";
   os << "  Dual Pivots       : " << degenPivotsDual << "\n";
   os << "  Primal Candidates : " << degenPivotCandPrimal << "\n";
   os << "  Dual Candidates   : " << degenPivotCandDual << "\n";
   os << "  Average Primal    : " << avgPrimalDegeneracy << "\n";
   os << "  Average Dual      : " << avgDualDegeneracy << "\n";

   if(iterationsInit > 0)
   {
      os << "Algorithm Iterations: " << callsReducedProb << "\n";
      os << "Decomp. Iterations  : \n";
      os << "  Total             : " << iterationsInit + iterationsRedProb << "\n";
      os << "  Initial           : " << iterationsInit << "\n";
      os << "  Reduced Problem   : " << iterationsRedProb << "\n";
      os << "  Comp. Problem     : " << iterationsCompProb << "\n";
      os << "Red. Problem Size   : \n";
      os << "  Rows              : " << numRedProbRows << "\n";
      os << "  Columns           : " << numRedProbCols << "\n";

      SPxOut::setScientific(os, 16);

      os << "Decomp. Basis Cond. : " << decompBasisCondNum << "\n";
      os << "Decomp Violations   : \n";
      os << "  Sum Bound         : " << totalBoundViol << "\n";
      os << "  Sum Row           : " << totalRowViol << "\n";
      os << "  Max Bound         : " << maxBoundViol << "\n";
      os << "  Max Row           : " << maxRowViol << "\n";

      SPxOut::setFixed(os, 2);

      os << "Red. Problem Status : " << redProbStatus << "\n";
      os << "Comp. Problem Status: " << compProbStatus << "\n";

      SPxOut::setScientific(os, 16);

      os << "Comp. Problem Obj.  : " << finalCompObj << "\n";
   }

   SPxOut::setScientific(os);

   os << "Numerics            :\n";
   os << "  Condition Number  : " << finalBasisCondition << "\n";

}
} // namespace soplex
