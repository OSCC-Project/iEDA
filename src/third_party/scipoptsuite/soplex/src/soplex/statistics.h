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

/**@file  statistics.h
 * @brief Class for collecting statistical information
 */
#ifndef _STATISTICS_H_
#define _STATISTICS_H_

#include <iostream>

#include "soplex.h"
#include "soplex/timer.h"

namespace soplex
{
/**@class   Statistics
 * @brief   Class for collecting statistical information
 * @ingroup Algo
 */
template <class R>
class SoPlexBase<R>::Statistics
{

public:


   ///@name Construction, resetting, printing
   ///@{

   /// default constructor
   Statistics(Timer::TYPE ttype = Timer::USER_TIME);

   /// copy constructor
   Statistics(const Statistics& base);

   /// assignment operator
   Statistics& operator=(const Statistics& rhs);

   /// destructor
   ~Statistics()
   {
      // we need to free all timers again (allocation happens in constructor)
      readingTime->~Timer();
      solvingTime->~Timer();
      preprocessingTime->~Timer();
      simplexTime->~Timer();
      syncTime->~Timer();
      transformTime->~Timer();
      rationalTime->~Timer();
      reconstructionTime->~Timer();
      spx_free(readingTime);
      spx_free(solvingTime);
      spx_free(preprocessingTime);
      spx_free(simplexTime);
      spx_free(syncTime);
      spx_free(transformTime);
      spx_free(rationalTime);
      spx_free(reconstructionTime);
   }

   /// clears all statistics
   void clearAllData();

   /// clears statistics on solving process
   void clearSolvingData();

   /// prints statistics
   void print(std::ostream& os);


   ///@}


   ///@name Data
   ///@{

   Timer* readingTime; ///< reading time not included in solving time
   Timer* solvingTime; ///< solving time
   Timer* preprocessingTime; ///< preprocessing time
   Timer* simplexTime; ///< simplex time
   Timer* syncTime; ///< time for synchronization between real and rational LP (included in solving time)
   Timer* transformTime; ///< time for transforming LPs (included in solving time)
   Timer* rationalTime; ///< time for rational LP solving (included in solving time)
   Timer* reconstructionTime; ///< time for rational reconstructions
   Timer::TYPE timerType; ///< type of timer (user or wallclock)

   Real multTimeSparse; ///< time for computing A*x exploiting sparsity (setupPupdate(), PRICE step)
   Real multTimeFull; ///< time for computing A*x ignoring sparsity (setupPupdate(), PRICE step)
   Real multTimeColwise; ///< time for computing A*x columnwise (setupPupdate(), PRICE step)
   Real multTimeUnsetup; ///< time for computing A*x w/o sparsity information (setupPupdate(), PRICE step)
   int multSparseCalls; ///< number of products A*x exploiting sparsity (setupPupdate(), PRICE step)
   int multFullCalls; ///< number of products A*x ignoring sparsity (setupPupdate(), PRICE step)
   int multColwiseCalls; ///< number of products A*x columnwise (setupPupdate(), PRICE step)
   int multUnsetupCalls; ///< number of products A*x w/o sparsity information (setupPupdate(), PRICE step)

   Real luFactorizationTimeReal; ///< time for factorizing bases matrices in real precision
   Real luSolveTimeReal; ///< time for solving linear systems in real precision
   Real luFactorizationTimeRational; ///< time for factorizing bases matrices in rational precision
   Real luSolveTimeRational; ///< time for solving linear systems in rational precision
   int iterations; ///< number of iterations/pivots
   int iterationsPrimal; ///< number of iterations with Primal
   int iterationsFromBasis; ///< number of iterations from Basis
   int iterationsPolish; ///< number of iterations during solution polishing
   int boundflips; ///< number of dual bound flips
   int luFactorizationsReal; ///< number of basis matrix factorizations in real precision
   int luSolvesReal; ///< number of (forward and backward) solves with basis matrix in real precision
   int luFactorizationsRational; ///< number of basis matrix factorizations in rational precision
   int rationalReconstructions; ///< number of rational reconstructions performed
   int refinements; ///< number of refinement steps
   int stallRefinements; ///< number of refinement steps without pivots
   int pivotRefinements; ///< number of refinement steps until final basis is reached
   int feasRefinements; ///< number of refinement steps during infeasibility test
   int unbdRefinements; ///< number of refinement steps during undboundedness test

   // Improved dual simplex statistics
   int callsReducedProb;      ///< number of times the reduced problem is solved. This includes the initial solve.
   int iterationsInit;        ///< number of iterations in the initial LP
   int iterationsRedProb;     ///< number of iterations of the reduced problem
   int iterationsCompProb;    ///< number of iterations of the complementary problem
   int numRedProbRows;        ///< number of rows in the reduced problem
   int numRedProbCols;        ///< number of columns in the reduced problem
   int degenPivotsPrimal;     ///< number of primal degenerate pivots
   int degenPivotsDual;       ///< number of dual degenerate pivots
   int degenPivotCandPrimal;  ///< number of pivoting candidates that will produce a degenerate step in the primal
   int degenPivotCandDual;    ///< number of pivoting candidates that will produce a degenerate step in the dual
   R sumDualDegen;         ///< the sum of the rate of dual degeneracy at each iteration
   R sumPrimalDegen;       ///< the sum of the rate of primal degeneracy at each iteration
   R decompBasisCondNum;   ///< the condition number for the basis used to perform the decomposition
   R totalBoundViol;       ///< the sum of the bound violations in the original problem using the red prob sol
   R totalRowViol;         ///< the sum of the row violations in the original problem using the red prob sol
   R maxBoundViol;         ///< the max bound violation in the original problem using the red prob sol
   R maxRowViol;           ///< the max row violations in the original problem using the red prob sol
   int  redProbStatus;        ///< status of the reduced problem
   int  compProbStatus;       ///< status of the complementary problem
   R finalCompObj;         ///< the final objective function of the complementary problem

   // Numerics
   R finalBasisCondition;  ///< condition number estimate of the optimal basis matrix

   ///@}

};
} // namespace soplex

// For general templated files
#include "statistics.hpp"

#endif // _STATISTICS_H_
