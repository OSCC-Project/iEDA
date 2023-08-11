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

/**@file  slufactor.h
 * @brief Implementation of Sparse Linear Solver.
 */
#ifndef _SLUFACTOR_H_
#define _SLUFACTOR_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/timerfactory.h"
#include "soplex/slinsolver.h"
#include "soplex/clufactor.h"

namespace soplex
{
/// maximum nr. of factorization updates allowed before refactorization.
#define MAXUPDATES      1000

/**@brief   Implementation of Sparse Linear Solver.
 * @ingroup Algo
 *
 * This class implements a SLinSolver interface by using the sparse LU
 * factorization implemented in CLUFactor.
 */
template <class R>
class SLUFactor : public SLinSolver<R>, protected CLUFactor<R>
{
public:

   //--------------------------------
   /**@name Types */
   ///@{
   /// Specifies how to perform \ref soplex::SLUFactor<R>::change "change" method.
   enum UpdateType
   {
      ETA = 0,       ///<
      FOREST_TOMLIN  ///<
   };
   /// for convenience
   using Status = typename SLinSolver<R>::Status;
   ///@}

private:

   //--------------------------------
   /**@name Private data */
   ///@{
   VectorBase<R>    vec;           ///< Temporary VectorBase<R>
   SSVectorBase<R>    ssvec;         ///< Temporary semi-sparse VectorBase<R>
   ///@}

protected:

   //--------------------------------
   /**@name Protected data */
   ///@{
   bool       usetup;        ///< TRUE iff update vector has been setup
   UpdateType uptype;        ///< the current \ref soplex::SLUFactor<R>::UpdateType "UpdateType".
   SSVectorBase<R>    eta;           ///<
   SSVectorBase<R>
   forest;        ///< ? Update VectorBase<R> set up by solveRight4update() and solve2right4update()
   R       lastThreshold; ///< pivoting threshold of last factorization
   ///@}

   //--------------------------------
   /**@name Control Parameters */
   ///@{
   /// minimum threshold to use.
   R minThreshold;
   /// minimum stability to achieve by setting threshold.
   R minStability;
   /// |x| < epsililon is considered to be 0.
   R epsilon;
   /// Time spent in solves
   Timer* solveTime;
   Timer::TYPE timerType;
   /// Number of solves
   int     solveCount;
   ///@}

protected:

   //--------------------------------
   /**@name Protected helpers */
   ///@{
   ///
   void freeAll();
   ///
   void changeEta(int idx, SSVectorBase<R>& eta);
   ///@}


public:

   //--------------------------------
   /**@name Update type */
   ///@{
   /// returns the current update type uptype.
   UpdateType utype() const
   {
      return uptype;
   }

   /// sets update type.
   /** The new UpdateType becomes valid only after the next call to
       method load().
   */
   void setUtype(UpdateType tp)
   {
      uptype = tp;
   }

   /// sets minimum Markowitz threshold.
   void setMarkowitz(R m)
   {
      if(m < 0.0001)
         m = 0.0001;

      if(m > 0.9999)
         m = 0.9999;

      minThreshold = m;
      lastThreshold = m;
   }

   /// returns Markowitz threshold.
   R markowitz()
   {
      return lastThreshold;
   }
   ///@}

   //--------------------------------
   /**@name Derived from SLinSolver
      See documentation of \ref soplex::SLinSolver "SLinSolver" for a
      documentation of these methods.
   */
   ///@{
   ///
   void clear();
   ///
   int dim() const
   {
      return this->thedim;
   }
   ///
   int memory() const
   {
      return this->nzCnt + this->l.start[this->l.firstUnused];
   }
   ///
   const char* getName() const
   {
      return (uptype == SLUFactor<R>::ETA) ? "SLU-Eta" : "SLU-Forest-Tomlin";
   }
   ///
   Status status() const
   {
      return Status(this->stat);
   }
   ///
   R stability() const;
   /** return one of several matrix metrics based on the diagonal of U
    * 0: condition number estimate by ratio of min/max
    * 1: trace (sum of diagonal elements)
    * 2: determinant (product of diagonal elements)
    */
   R matrixMetric(int type = 0) const;
   ///
   std::string statistics() const;
   ///
   Status load(const SVectorBase<R>* vec[], int dim);
   ///@}

public:

   //--------------------------------
   /**@name Solve */
   ///@{
   /// Solves \f$Ax=b\f$.
   void solveRight(VectorBase<R>& x, const VectorBase<R>& b);
   void solveRight(SSVectorBase<R>& x, const SSVectorBase<R>& b)
   {
      x.unSetup();
      solveRight((VectorBase<R>&) x, (const VectorBase<R>&) b);
   }
   /// Solves \f$Ax=b\f$.
   void solveRight(SSVectorBase<R>& x, const SVectorBase<R>& b);
   /// Solves \f$Ax=b\f$.
   void solveRight4update(SSVectorBase<R>& x, const SVectorBase<R>& b);
   /// Solves \f$Ax=b\f$ and \f$Ay=d\f$.
   void solve2right4update(SSVectorBase<R>& x, VectorBase<R>& y, const SVectorBase<R>& b,
                           SSVectorBase<R>& d);
   /// Sparse version of solving two systems of equations
   void solve2right4update(SSVectorBase<R>& x, SSVectorBase<R>& y, const SVectorBase<R>& b,
                           SSVectorBase<R>& d);
   /// Solves \f$Ax=b\f$, \f$Ay=d\f$ and \f$Az=e\f$.
   void solve3right4update(SSVectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& z,
                           const SVectorBase<R>& b, SSVectorBase<R>& d, SSVectorBase<R>& e);
   /// sparse version of solving three systems of equations
   void solve3right4update(SSVectorBase<R>& x, SSVectorBase<R>& y, SSVectorBase<R>& z,
                           const SVectorBase<R>& b, SSVectorBase<R>& d, SSVectorBase<R>& e);
   /// sparse version of solving one system of equations with transposed basis matrix
   void solveLeft(VectorBase<R>& x, const VectorBase<R>& b);
   void solveLeft(SSVectorBase<R>& x, const SSVectorBase<R>& b)
   {
      x.unSetup();
      solveLeft((VectorBase<R>&) x, (const VectorBase<R>&) b);
   }
   /// Solves \f$Ax=b\f$.
   void solveLeft(SSVectorBase<R>& x, const SVectorBase<R>& b);
   /// Solves \f$Ax=b\f$ and \f$Ay=d\f$.
   void solveLeft(SSVectorBase<R>& x, VectorBase<R>& y, const SVectorBase<R>& b, SSVectorBase<R>& d);
   /// sparse version of solving two systems of equations with transposed basis matrix
   void solveLeft(SSVectorBase<R>& x, SSVectorBase<R>& two, const SVectorBase<R>& b,
                  SSVectorBase<R>& rhs2);
   /// Solves \f$Ax=b\f$, \f$Ay=d\f$ and \f$Az=e\f$.
   void solveLeft(SSVectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& z,
                  const SVectorBase<R>& b, SSVectorBase<R>& d, SSVectorBase<R>& e);
   /// sparse version of solving three systems of equations with transposed basis matrix
   void solveLeft(SSVectorBase<R>& x, SSVectorBase<R>& y, SSVectorBase<R>& z,
                  const SVectorBase<R>& b, SSVectorBase<R>& d, SSVectorBase<R>& e);
   ///
   Status change(int idx, const SVectorBase<R>& subst, const SSVectorBase<R>* eta = 0);
   ///@}

   //--------------------------------
   /**@name Miscellaneous */
   ///@{
   /// time spent in factorizations
   // @todo fix the return type from of the type form Real to a cpp time (Refactoring) TODO
   Real getFactorTime() const
   {
      return this->factorTime->time();
   }
   /// reset FactorTime
   void resetFactorTime()
   {
      this->factorTime->reset();
   }
   /// number of factorizations performed
   int getFactorCount() const
   {
      return this->factorCount;
   }
   /// time spent in solves
   // @todo fix the return type of time to a cpp time type TODO
   Real getSolveTime() const
   {
      return solveTime->time();
   }
   /// reset SolveTime
   void resetSolveTime()
   {
      solveTime->reset();
   }
   /// number of solves performed
   int getSolveCount() const
   {
      return solveCount;
   }
   /// reset timers and counters
   void resetCounters()
   {
      this->factorTime->reset();
      solveTime->reset();
      this->factorCount = 0;
      this->hugeValues = 0;
      solveCount = 0;
   }
   void changeTimer(const Timer::TYPE ttype)
   {
      solveTime = TimerFactory::switchTimer(solveTime, ttype);
      this->factorTime = TimerFactory::switchTimer(this->factorTime, ttype);
      timerType = ttype;
   }
   /// prints the LU factorization to stdout.
   void dump() const;

   /// consistency check.
   bool isConsistent() const;
   ///@}

   //------------------------------------
   /**@name Constructors / Destructors */
   ///@{
   /// default constructor.
   SLUFactor<R>();
   /// assignment operator.
   SLUFactor<R>& operator=(const SLUFactor<R>& old);
   /// copy constructor.
   SLUFactor<R>(const SLUFactor<R>& old);
   /// destructor.
   virtual ~SLUFactor<R>();
   /// clone function for polymorphism
   inline virtual SLinSolver<R>* clone() const
   {
      return new SLUFactor<R>(*this);
   }
   ///@}

private:

   //------------------------------------
   /**@name Private helpers */
   ///@{
   /// used to implement the assignment operator
   void assign(const SLUFactor<R>& old);
   ///@}
};

} // namespace soplex

#include "slufactor.hpp"
#endif // _SLUFACTOR_H_
