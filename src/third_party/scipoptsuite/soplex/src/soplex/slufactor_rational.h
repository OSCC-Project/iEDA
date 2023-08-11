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

/**@file  slufactor_rational.h
 * @brief Implementation of Sparse Linear Solver with Rational precision.
 */
#ifndef _SLUFACTOR_RATIONAL_H_
#define _SLUFACTOR_RATIONAL_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/timerfactory.h"
#include "soplex/slinsolver_rational.h"
#include "soplex/clufactor_rational.h"
#include "soplex/rational.h"

namespace soplex
{
/// maximum nr. of factorization updates allowed before refactorization.
#define MAXUPDATES      1000

/**@brief   Implementation of Sparse Linear Solver with Rational precision.
 * @ingroup Algo
 *
 * This class implements a SLinSolverRational interface by using the sparse LU
 * factorization implemented in CLUFactorRational.
 */
class SLUFactorRational : public SLinSolverRational, protected CLUFactorRational
{
public:

   //--------------------------------
   /**@name Types */
   ///@{
   /// Specifies how to perform \ref soplex::SLUFactorRational::change "change" method.
   enum UpdateType
   {
      ETA = 0,       ///<
      FOREST_TOMLIN  ///<
   };
   /// for convenience
   typedef SLinSolverRational::Status Status;
   ///@}

private:

   //--------------------------------
   /**@name Private data */
   ///@{
   VectorRational    vec;           ///< Temporary vector
   SSVectorRational   ssvec;         ///< Temporary semi-sparse vector
   ///@}

protected:

   //--------------------------------
   /**@name Protected data */
   ///@{
   bool       usetup;             ///< TRUE iff update vector has been setup
   UpdateType uptype;             ///< the current \ref soplex::SLUFactor<R>::UpdateType "UpdateType".
   SSVectorRational   eta;        ///<
   SSVectorRational
   forest;     ///< ? Update vector set up by solveRight4update() and solve2right4update()
   Rational       lastThreshold;  ///< pivoting threshold of last factorization
   ///@}

   //--------------------------------
   /**@name Control Parameters */
   ///@{
   /// minimum threshold to use.
   Rational minThreshold;
   /// minimum stability to achieve by setting threshold.
   Rational minStability;
   /// Time spent in solves
   Timer*  solveTime;
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
   void changeEta(int idx, SSVectorRational& eta);
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
   void setMarkowitz(const Rational& m)
   {
      if(m < 0.01)
      {
         minThreshold = 0.01;
         lastThreshold = 0.01;
      }
      else if(m > 0.99)
      {
         minThreshold = 0.99;
         lastThreshold = 0.99;
      }
      else
      {
         minThreshold = m;
         lastThreshold = m;
      }
   }

   /// returns Markowitz threshold.
   Rational markowitz()
   {
      return lastThreshold;
   }
   ///@}

   //--------------------------------
   /**@name Derived from SLinSolverRational
      See documentation of \ref soplex::SLinSolverRational "SLinSolverRational" for a
      documentation of these methods.
   */
   ///@{
   ///
   void clear();
   ///
   int dim() const
   {
      return thedim;
   }
   ///
   int memory() const
   {
      return nzCnt + l.start[l.firstUnused];
   }
   ///
   const char* getName() const
   {
      return (uptype == SLUFactorRational::ETA) ? "SLU-Eta" : "SLU-Forest-Tomlin";
   }
   ///
   Status status() const
   {
      return Status(stat);
   }
   ///
   Rational stability() const;
   ///
   std::string statistics() const;
   ///
   Status load(const SVectorRational* vec[], int dim);
   ///@}

public:

   //--------------------------------
   /**@name Solve */
   ///@{
   /// Solves \f$Ax=b\f$.
   void solveRight(VectorRational& x, const VectorRational& b);
   /// Solves \f$Ax=b\f$.
   void solveRight(SSVectorRational& x, const SVectorRational& b);
   /// Solves \f$Ax=b\f$.
   void solveRight4update(SSVectorRational& x, const SVectorRational& b);
   /// Solves \f$Ax=b\f$ and \f$Ay=d\f$.
   void solve2right4update(SSVectorRational& x, VectorRational& y, const SVectorRational& b,
                           SSVectorRational& d);
   /// Solves \f$Ax=b\f$, \f$Ay=d\f$ and \f$Az=e\f$.
   void solve3right4update(SSVectorRational& x, VectorRational& y, VectorRational& z,
                           const SVectorRational& b, SSVectorRational& d, SSVectorRational& e);
   /// Solves \f$Ax=b\f$.
   void solveLeft(VectorRational& x, const VectorRational& b);
   /// Solves \f$Ax=b\f$.
   void solveLeft(SSVectorRational& x, const SVectorRational& b);
   /// Solves \f$Ax=b\f$ and \f$Ay=d\f$.
   void solveLeft(SSVectorRational& x, VectorRational& y, const SVectorRational& b,
                  SSVectorRational& d);
   /// Solves \f$Ax=b\f$, \f$Ay=d\f$ and \f$Az=e\f$.
   void solveLeft(SSVectorRational& x, VectorRational& y, VectorRational& z,
                  const SVectorRational& b, SSVectorRational& d, SSVectorRational& e);
   ///
   Status change(int idx, const SVectorRational& subst, const SSVectorRational* eta = 0);
   ///@}

   //--------------------------------
   /**@name Miscellaneous */
   ///@{
   /// time spent in factorizations
   Real getFactorTime() const
   {
      return factorTime->time();
   }
   /// set time limit on factorization
   void setTimeLimit(const Real limit)
   {
      timeLimit = limit;
   }
   /// reset FactorTime
   void resetFactorTime()
   {
      factorTime->reset();
   }
   /// number of factorizations performed
   int getFactorCount() const
   {
      return factorCount;
   }
   /// time spent in solves
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
      factorTime->reset();
      solveTime->reset();
      factorCount = 0;
      solveCount = 0;
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
   SLUFactorRational()
      : CLUFactorRational()
      , vec(1)
      , ssvec(1)
      , usetup(false)
      , uptype(FOREST_TOMLIN)
      , eta(1)
      , forest(1)
      , minThreshold(0.01)
      , timerType(Timer::USER_TIME)
   {
      row.perm    = 0;
      row.orig    = 0;
      col.perm    = 0;
      col.orig    = 0;
      u.row.elem  = 0;
      u.row.idx   = 0;
      u.row.start = 0;
      u.row.len   = 0;
      u.row.max   = 0;
      u.col.elem  = 0;
      u.col.idx   = 0;
      u.col.start = 0;
      u.col.len   = 0;
      u.col.max   = 0;
      l.idx       = 0;
      l.start     = 0;
      l.row       = 0;
      l.ridx      = 0;
      l.rbeg      = 0;
      l.rorig     = 0;
      l.rperm     = 0;

      nzCnt  = 0;
      thedim = 0;

      try
      {
         solveTime = TimerFactory::createTimer(timerType);
         factorTime = TimerFactory::createTimer(timerType);
         spx_alloc(row.perm, thedim);
         spx_alloc(row.orig, thedim);
         spx_alloc(col.perm, thedim);
         spx_alloc(col.orig, thedim);
         diag.reDim(thedim);

         work = vec.get_ptr();

         u.row.used = 0;
         spx_alloc(u.row.elem,  thedim);
         u.row.val.reDim(1);
         spx_alloc(u.row.idx,   u.row.val.dim());
         spx_alloc(u.row.start, thedim + 1);
         spx_alloc(u.row.len,   thedim + 1);
         spx_alloc(u.row.max,   thedim + 1);

         u.row.list.idx      = thedim;
         u.row.start[thedim] = 0;
         u.row.max  [thedim] = 0;
         u.row.len  [thedim] = 0;

         u.col.size = 1;
         u.col.used = 0;
         spx_alloc(u.col.elem,  thedim);
         spx_alloc(u.col.idx,   u.col.size);
         spx_alloc(u.col.start, thedim + 1);
         spx_alloc(u.col.len,   thedim + 1);
         spx_alloc(u.col.max,   thedim + 1);
         u.col.val.reDim(0);

         u.col.list.idx      = thedim;
         u.col.start[thedim] = 0;
         u.col.max[thedim]   = 0;
         u.col.len[thedim]   = 0;

         l.val.reDim(1);
         spx_alloc(l.idx, l.val.dim());

         l.startSize   = 1;
         l.firstUpdate = 0;
         l.firstUnused = 0;

         spx_alloc(l.start, l.startSize);
         spx_alloc(l.row,   l.startSize);
      }
      catch(const SPxMemoryException& x)
      {
         freeAll();
         throw x;
      }

      l.rval.reDim(0);
      l.ridx  = 0;
      l.rbeg  = 0;
      l.rorig = 0;
      l.rperm = 0;

      SLUFactorRational::clear(); // clear() is virtual

      factorCount = 0;
      timeLimit = -1.0;
      solveCount  = 0;

      assert(row.perm != 0);
      assert(row.orig != 0);
      assert(col.perm != 0);
      assert(col.orig != 0);

      assert(u.row.elem  != 0);
      assert(u.row.idx   != 0);
      assert(u.row.start != 0);
      assert(u.row.len   != 0);
      assert(u.row.max   != 0);

      assert(u.col.elem  != 0);
      assert(u.col.idx   != 0);
      assert(u.col.start != 0);
      assert(u.col.len   != 0);
      assert(u.col.max   != 0);

      assert(l.idx   != 0);
      assert(l.start != 0);
      assert(l.row   != 0);

      assert(SLUFactorRational::isConsistent());
   }
   /// assignment operator.
   SLUFactorRational& operator=(const SLUFactorRational& old)
   {

      if(this != &old)
      {
         // we don't need to copy them, because they are temporary vectors
         vec.clear();
         ssvec.clear();

         eta    = old.eta;
         forest = old.forest;

         freeAll();

         try
         {
            assign(old);
         }
         catch(const SPxMemoryException& x)
         {
            freeAll();
            throw x;
         }

         assert(isConsistent());
      }

      return *this;
   }

   /// copy constructor.
   SLUFactorRational(const SLUFactorRational& old)
      : SLinSolverRational(old)
      , CLUFactorRational()
      , vec(1)     // we don't need to copy it, because they are temporary vectors
      , ssvec(1)   // we don't need to copy it, because they are temporary vectors
      , usetup(old.usetup)
      , eta(old.eta)
      , forest(old.forest)
      , timerType(old.timerType)
   {
      row.perm    = 0;
      row.orig    = 0;
      col.perm    = 0;
      col.orig    = 0;
      u.row.elem  = 0;
      u.row.idx   = 0;
      u.row.start = 0;
      u.row.len   = 0;
      u.row.max   = 0;
      u.col.elem  = 0;
      u.col.idx   = 0;
      u.col.start = 0;
      u.col.len   = 0;
      u.col.max   = 0;
      l.idx       = 0;
      l.start     = 0;
      l.row       = 0;
      l.ridx      = 0;
      l.rbeg      = 0;
      l.rorig     = 0;
      l.rperm     = 0;

      solveCount = 0;
      solveTime = TimerFactory::createTimer(timerType);
      factorTime = TimerFactory::createTimer(timerType);

      try
      {
         assign(old);
      }
      catch(const SPxMemoryException& x)
      {
         freeAll();
         throw x;
      }

      assert(SLUFactorRational::isConsistent());
   }

   /// destructor.
   virtual ~SLUFactorRational();
   /// clone function for polymorphism
   inline virtual SLinSolverRational* clone() const
   {
      return new SLUFactorRational(*this);
   }
   ///@}

private:

   //------------------------------------
   /**@name Private helpers */
   ///@{
   /// used to implement the assignment operator
   void assign(const SLUFactorRational& old);
   ///@}
};

} // namespace soplex
#include "slufactor_rational.hpp"
#endif // _SLUFACTOR_RATIONAL_H_
