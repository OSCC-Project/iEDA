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

/**@file slufactor.hpp
 * @todo SLUfactor seems to be partly an wrapper for CLUFactor (was C).
 *       This should be properly integrated and demangled.
 * @todo Does is make sense, to call x.clear() when next x.altValues()
 *       is called.
 */

#include <assert.h>
#include <sstream>

#ifdef SOPLEX_DEBUG
#include <stdio.h>
#endif

namespace soplex
{
#define MINSTABILITY    R(4e-2)

template <class R>
void SLUFactor<R>::solveRight(VectorBase<R>& x, const VectorBase<R>& b) //const
{

   this->solveTime->start();

   this->vec = b;
   x.clear();
   CLUFactor<R>::solveRight(x.get_ptr(), vec.get_ptr());

   solveCount++;
   solveTime->stop();
}

template <class R>
void SLUFactor<R>::solveRight(SSVectorBase<R>& x, const SVectorBase<R>& b)  //const
{

   solveTime->start();

   vec.assign(b);
   x.clear();
   CLUFactor<R>::solveRight(x.altValues(), vec.get_ptr());

   solveCount++;
   solveTime->stop();
}

template <class R>
void SLUFactor<R>::solveRight4update(SSVectorBase<R>& x, const SVectorBase<R>& b)
{

   solveTime->start();

   int m;
   int n;
   int f;

   x.clear();
   ssvec = b;
   n = ssvec.size();

   if(this->l.updateType == ETA)
   {
      m = this->vSolveRight4update(x.getEpsilon(), x.altValues(), x.altIndexMem(),
                                   ssvec.altValues(), ssvec.altIndexMem(), n, 0, 0, 0);
      x.setSize(m);
      //x.forceSetup();
      x.unSetup();
      eta.setup_and_assign(x);
   }
   else
   {
      forest.clear();
      m = this->vSolveRight4update(x.getEpsilon(), x.altValues(), x.altIndexMem(),
                                   ssvec.altValues(), ssvec.altIndexMem(), n,
                                   forest.altValues(), &f, forest.altIndexMem());
      forest.setSize(f);
      forest.forceSetup();
      x.setSize(m);
      x.forceSetup();
   }

   usetup = true;
   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount++;
   solveTime->stop();
}

template <class R>
void SLUFactor<R>::solve2right4update(
   SSVectorBase<R>&       x,
   VectorBase<R>&        y,
   const SVectorBase<R>& b,
   SSVectorBase<R>&       rhs)
{

   solveTime->start();

   int  m;
   int  n;
   int  f;
   int* sidx = ssvec.altIndexMem();
   ssvec.setSize(0);
   ssvec.forceSetup();
   int  rsize = rhs.size();
   int* ridx = rhs.altIndexMem();

   x.clear();
   y.clear();
   usetup = true;
   ssvec = b;

   if(this->l.updateType == ETA)
   {
      n = ssvec.size();
      m = this->vSolveRight4update2(x.getEpsilon(), x.altValues(), x.altIndexMem(),
                                    ssvec.get_ptr(), sidx, n, y.get_ptr(),
                                    rhs.getEpsilon(), rhs.altValues(), ridx, rsize, 0, 0, 0);
      x.setSize(m);
      //      x.forceSetup();
      x.unSetup();
      eta.setup_and_assign(x);
   }
   else
   {
      forest.clear();
      n = ssvec.size();
      m = this->vSolveRight4update2(x.getEpsilon(), x.altValues(), x.altIndexMem(),
                                    ssvec.get_ptr(), sidx, n, y.get_ptr(),
                                    rhs.getEpsilon(), rhs.altValues(), ridx, rsize,
                                    forest.altValues(), &f, forest.altIndexMem());
      x.setSize(m);
      x.forceSetup();
      forest.setSize(f);
      forest.forceSetup();
   }

   rhs.forceSetup();
   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount += 2;
   solveTime->stop();
}

template <class R>
void SLUFactor<R>::solve2right4update(
   SSVectorBase<R>&       x,
   SSVectorBase<R>&       y,
   const SVectorBase<R>& b,
   SSVectorBase<R>&       rhs)
{

   solveTime->start();

   int  n;
   int  f;
   int* sidx = ssvec.altIndexMem();
   ssvec.setSize(0);
   ssvec.forceSetup();
   int  rsize = rhs.size();
   int* ridx = rhs.altIndexMem();

   x.clear();
   y.clear();
   usetup = true;
   ssvec = b;

   if(this->l.updateType == ETA)
   {
      n = ssvec.size();
      this->vSolveRight4update2sparse(x.getEpsilon(), x.altValues(), x.altIndexMem(),
                                      ssvec.get_ptr(), sidx, n,
                                      y.getEpsilon(), y.altValues(), y.altIndexMem(),
                                      rhs.altValues(), ridx, rsize,
                                      0, 0, 0);
      x.setSize(n);
      //      x.forceSetup();
      x.unSetup();
      y.setSize(rsize);
      y.unSetup();
      eta.setup_and_assign(x);
   }
   else
   {
      forest.clear();
      n = ssvec.size();
      this->vSolveRight4update2sparse(x.getEpsilon(), x.altValues(), x.altIndexMem(),
                                      ssvec.get_ptr(), sidx, n,
                                      y.getEpsilon(), y.altValues(), y.altIndexMem(),
                                      rhs.altValues(), ridx, rsize,
                                      forest.altValues(), &f, forest.altIndexMem());
      x.setSize(n);
      x.forceSetup();
      y.setSize(rsize);
      y.forceSetup();
      forest.setSize(f);
      forest.forceSetup();
   }

   rhs.forceSetup();
   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount += 2;
   solveTime->stop();
}


template <class R>
void SLUFactor<R>::solve3right4update(
   SSVectorBase<R>&       x,
   VectorBase<R>&        y,
   VectorBase<R>&        y2,
   const SVectorBase<R>& b,
   SSVectorBase<R>&       rhs,
   SSVectorBase<R>&       rhs2)
{

   solveTime->start();

   int  m;
   int  n;
   int  f;
   int* sidx = ssvec.altIndexMem();
   ssvec.setSize(0);
   ssvec.forceSetup();
   int  rsize = rhs.size();
   int* ridx = rhs.altIndexMem();
   int  rsize2 = rhs2.size();
   int* ridx2 = rhs2.altIndexMem();

   x.clear();
   y.clear();
   y2.clear();
   usetup = true;
   ssvec = b;

   if(this->l.updateType == ETA)
   {
      n = ssvec.size();
      m = this->vSolveRight4update3(x.getEpsilon(),
                                    x.altValues(), x.altIndexMem(), ssvec.get_ptr(), sidx, n,
                                    y.get_ptr(), rhs.getEpsilon(), rhs.altValues(), ridx, rsize,
                                    y2.get_ptr(), rhs2.getEpsilon(), rhs2.altValues(), ridx2, rsize2,
                                    0, 0, 0);
      x.setSize(m);
      //      x.forceSetup();
      x.unSetup();
      eta.setup_and_assign(x);
   }
   else
   {
      forest.clear();
      n = ssvec.size();
      m = this->vSolveRight4update3(x.getEpsilon(),
                                    x.altValues(), x.altIndexMem(), ssvec.get_ptr(), sidx, n,
                                    y.get_ptr(), rhs.getEpsilon(), rhs.altValues(), ridx, rsize,
                                    y2.get_ptr(), rhs2.getEpsilon(), rhs2.altValues(), ridx2, rsize2,
                                    forest.altValues(), &f, forest.altIndexMem());
      x.setSize(m);
      x.forceSetup();
      forest.setSize(f);
      forest.forceSetup();
   }

   rhs.forceSetup();
   rhs2.forceSetup();
   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount += 3;
   solveTime->stop();
}

template <class R>
void SLUFactor<R>::solve3right4update(
   SSVectorBase<R>&       x,
   SSVectorBase<R>&       y,
   SSVectorBase<R>&       y2,
   const SVectorBase<R>& b,
   SSVectorBase<R>&       rhs,
   SSVectorBase<R>&       rhs2)
{
   solveTime->start();

   int  n;
   int  f;
   int* sidx = ssvec.altIndexMem();
   ssvec.setSize(0);
   ssvec.forceSetup();
   int  rsize = rhs.size();
   int* ridx = rhs.altIndexMem();
   int  rsize2 = rhs2.size();
   int* ridx2 = rhs2.altIndexMem();

   x.clear();
   y.clear();
   y2.clear();
   usetup = true;
   ssvec = b;

   if(this->l.updateType == ETA)
   {
      n = ssvec.size();
      this->vSolveRight4update3sparse(x.getEpsilon(), x.altValues(), x.altIndexMem(),
                                      ssvec.get_ptr(), sidx, n,
                                      y.getEpsilon(), y.altValues(), y.altIndexMem(),
                                      rhs.altValues(), ridx, rsize,
                                      y2.getEpsilon(), y2.altValues(), y2.altIndexMem(),
                                      rhs2.altValues(), ridx2, rsize2,
                                      0, 0, 0);
      x.setSize(n);
      //      x.forceSetup();
      x.unSetup();
      y.setSize(rsize);
      y.unSetup();
      y2.setSize(rsize2);
      y2.unSetup();
      eta.setup_and_assign(x);
   }
   else
   {
      forest.clear();
      n = ssvec.size();
      this->vSolveRight4update3sparse(x.getEpsilon(), x.altValues(), x.altIndexMem(),
                                      ssvec.get_ptr(), sidx, n,
                                      y.getEpsilon(), y.altValues(), y.altIndexMem(),
                                      rhs.altValues(), ridx, rsize,
                                      y2.getEpsilon(), y2.altValues(), y2.altIndexMem(),
                                      rhs2.altValues(), ridx2, rsize2,
                                      forest.altValues(), &f, forest.altIndexMem());
      x.setSize(n);
      x.forceSetup();
      y.setSize(rsize);
      y.forceSetup();
      y2.setSize(rsize2);
      y2.forceSetup();

      forest.setSize(f);
      forest.forceSetup();
   }

   rhs.forceSetup();
   rhs2.forceSetup();
   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount += 3;
   solveTime->stop();
}


template <class R>
void SLUFactor<R>::solveLeft(VectorBase<R>& x, const VectorBase<R>& b) //const
{

   solveTime->start();

   vec = b;
   x.clear();
   CLUFactor<R>::solveLeft(x.get_ptr(), vec.get_ptr());

   solveCount++;
   solveTime->stop();
}

template <class R>
void SLUFactor<R>::solveLeft(SSVectorBase<R>& x, const SVectorBase<R>& b)  //const
{

   solveTime->start();

   // copy to SSVec is done to avoid having to deal with the Nonzero datatype
   // TODO change SVec to standard sparse format
   ssvec.assign(b);

   x.clear();
   int sz = ssvec.size(); // see .altValues()
   int n = this->vSolveLeft(x.getEpsilon(), x.altValues(), x.altIndexMem(),
                            ssvec.altValues(), ssvec.altIndexMem(), sz);

   if(n > 0)
   {
      x.setSize(n);
      x.forceSetup();
   }
   else
      x.unSetup();

   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount++;
   solveTime->stop();
}

template <class R>
void SLUFactor<R>::solveLeft(
   SSVectorBase<R>&       x,
   VectorBase<R>&        y,
   const SVectorBase<R>& rhs1,
   SSVectorBase<R>&       rhs2) //const
{

   solveTime->start();

   int   n;
   R* svec = ssvec.altValues();
   int*  sidx = ssvec.altIndexMem();
   int   rn   = rhs2.size();
   int*  ridx = rhs2.altIndexMem();

   x.clear();
   y.clear();
   ssvec.assign(rhs1);
   n = ssvec.size(); // see altValues();
   n = this->vSolveLeft2(x.getEpsilon(), x.altValues(), x.altIndexMem(), svec, sidx, n,
                         y.get_ptr(), rhs2.altValues(), ridx, rn);

   // this will unsetup x
   x.setSize(n);

   if(n > 0)
      x.forceSetup();

   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount += 2;
   solveTime->stop();
}

template <class R>
void SLUFactor<R>::solveLeft(
   SSVectorBase<R>&       x,
   SSVectorBase<R>&       y,
   const SVectorBase<R>& rhs1,
   SSVectorBase<R>&       rhs2) //const
{

   solveTime->start();

   int   n1, n2;
   R* svec = ssvec.altValues();
   int*  sidx = ssvec.altIndexMem();

   x.clear();
   y.clear();
   ssvec.assign(rhs1);
   n1 = ssvec.size(); // see altValues();
   n2 = rhs2.size();

   if(n2 < 10)
   {
      this->vSolveLeft2sparse(x.getEpsilon(),
                              x.altValues(), x.altIndexMem(),
                              svec, sidx, n1,
                              y.altValues(), y.altIndexMem(),
                              rhs2.altValues(), rhs2.altIndexMem(), n2);
      y.setSize(n2);

      if(n2 > 0)
         y.forceSetup();
   }
   else
   {
      n1 = this->vSolveLeft2(x.getEpsilon(), x.altValues(), x.altIndexMem(), svec, sidx, n1,
                             y.altValues(), rhs2.altValues(), rhs2.altIndexMem(), n2);
      //      y.setup();
   }

   x.setSize(n1);

   if(n1 > 0)
      x.forceSetup();

   //   if (n2 > 0)
   //      y.forceSetup();

   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount += 2;
   solveTime->stop();
}


template <class R>
void SLUFactor<R>::solveLeft(
   SSVectorBase<R>&       x,
   VectorBase<R>&        y,
   VectorBase<R>&        z,
   const SVectorBase<R>& rhs1,
   SSVectorBase<R>&       rhs2,
   SSVectorBase<R>&       rhs3)
{

   solveTime->start();

   int   n, n2, n3;
   R* svec = ssvec.altValues();
   int*  sidx = ssvec.altIndexMem();

   x.clear();
   y.clear();
   z.clear();
   ssvec.assign(rhs1);
   n = ssvec.size(); // see altValues();
   n2 = rhs2.size();
   n3 = rhs3.size();

   n = this->vSolveLeft3(x.getEpsilon(), x.altValues(), x.altIndexMem(), svec, sidx, n,
                         y.get_ptr(), rhs2.altValues(), rhs2.altIndexMem(), n2,
                         z.get_ptr(), rhs3.altValues(), rhs3.altIndexMem(), n3);

   x.setSize(n);

   if(n > 0)
      x.forceSetup();

   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount += 3;
   solveTime->stop();
}

template <class R>
void SLUFactor<R>::solveLeft(
   SSVectorBase<R>&       x,
   SSVectorBase<R>&       y,
   SSVectorBase<R>&       z,
   const SVectorBase<R>& rhs1,
   SSVectorBase<R>&       rhs2,
   SSVectorBase<R>&       rhs3)
{

   solveTime->start();

   int   n1, n2, n3;
   R* svec = ssvec.altValues();
   int*  sidx = ssvec.altIndexMem();

   x.clear();
   y.clear();
   z.clear();
   ssvec.assign(rhs1);
   n1 = ssvec.size(); // see altValues();
   n2 = rhs2.size();
   n3 = rhs3.size();
   this->vSolveLeft3sparse(x.getEpsilon(),
                           x.altValues(), x.altIndexMem(),
                           svec, sidx, n1,
                           y.altValues(), y.altIndexMem(),
                           rhs2.altValues(), rhs2.altIndexMem(), n2,
                           z.altValues(), z.altIndexMem(),
                           rhs3.altValues(), rhs3.altIndexMem(), n3);
   x.setSize(n1);
   y.setSize(n2);
   z.setSize(n3);

   if(n1 > 0)
      x.forceSetup();

   if(n2 > 0)
      y.forceSetup();

   if(n3 > 0)
      z.forceSetup();

   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount += 3;
   solveTime->stop();
}


template <class R>
R SLUFactor<R>::stability() const
{
   if(status() != this->OK)
      return 0;

   if(this->maxabs < this->initMaxabs)
      return 1;

   assert(this->maxabs != 0.0);
   return this->initMaxabs / this->maxabs;
}

template <class R>
R SLUFactor<R>::matrixMetric(int type) const
{
   R result = 0.0;

   // catch corner case of empty matrix
   if(dim() == 0)
      return 1.0;

   switch(type)
   {
   // compute condition estimate by ratio of max/min of elements on the diagonal
   case 0:
   {
      R mindiag = spxAbs(this->diag[0]);
      R maxdiag = spxAbs(this->diag[0]);

      for(int i = 1; i < dim(); ++i)
      {
         R absdiag = spxAbs(this->diag[i]);

         if(absdiag < mindiag)
            mindiag = absdiag;
         else if(absdiag > maxdiag)
            maxdiag = absdiag;
      }

      result = maxdiag / mindiag;
      break;
   }

   // compute sum of inverses of all elements on the diagonal
   case 1:
      result = 0.0;

      for(int i = 0; i < dim(); ++i)
         result += 1.0 / this->diag[i];

      break;

   // compute determinant (product of all diagonal elements of U)
   case 2:
      result = 1.0;

      for(int i = 0; i < dim(); ++i)
         result *= this->diag[i];

      result = 1.0 / result;
      break;
   }

   return result;
}

template <class R>
std::string SLUFactor<R>::statistics() const
{
   std::stringstream s;
   s  << "Factorizations     : " << std::setw(10) << getFactorCount() << std::endl
      << "  Time spent       : " << std::setw(10) << std::fixed << std::setprecision(
         2) << getFactorTime() << std::endl
      << "Solves             : " << std::setw(10) << getSolveCount() << std::endl
      << "  Time spent       : " << std::setw(10) << getSolveTime() << std::endl;

   return s.str();
}

template <class R>
void SLUFactor<R>::changeEta(int idx, SSVectorBase<R>& et)
{

   int es = et.size(); // see altValues()
   this->update(idx, et.altValues(), et.altIndexMem(), es);
   et.setSize(0);
   et.forceSetup();
}

template <class R>
typename SLUFactor<R>::Status SLUFactor<R>::change(
   int             idx,
   const SVectorBase<R>&  subst,
   const SSVectorBase<R>* e)
{

   // BH 2005-08-23: The boolean usetup indicates that an "update VectorBase<R>"
   // has been set up. I suppose that SSVectorBase<R>  forest is this
   // update VectorBase<R>, which is set up by solveRight4update() and
   // solve2right4update() in order to optimize the basis update.

   if(usetup)
   {
      if(this->l.updateType == FOREST_TOMLIN)               // FOREST_TOMLIN updates
      {
         // BH 2005-08-19: The size of a SSVectorBase<R>  is the size of the
         // index set, i.e.  the number of nonzeros which is only
         // defined if the SSVectorBase<R>  is set up.  Since
         // SSVectorBase<R> ::altValues() calls unSetup() the size needs to be
         // stored before the following call.
         int fsize = forest.size(); // see altValues()
         this->forestUpdate(idx, forest.altValues(), fsize, forest.altIndexMem());
         forest.setSize(0);
         forest.forceSetup();
      }
      else
      {
         assert(this->l.updateType == ETA);
         changeEta(idx, eta);
      }
   }
   else if(e != 0)                                    // ETA updates
   {
      this->l.updateType = ETA;
      this->updateNoClear(idx, e->values(), e->indexMem(), e->size());
      this->l.updateType = uptype;
   }
   else if(this->l.updateType == FOREST_TOMLIN)             // FOREST_TOMLIN updates
   {
      assert(0);  // probably this part is never called.
      // forestUpdate() with the last parameter set to NULL should fail.
      forest = subst;
      CLUFactor<R>::solveLright(forest.altValues());
      this->forestUpdate(idx, forest.altValues(), 0, nullptr);
      forest.setSize(0);
      forest.forceSetup();
   }
   else                                               // ETA updates
   {
      assert(this->l.updateType == ETA);
      vec = subst;
      eta.clear();
      CLUFactor<R>::solveRight(eta.altValues(), vec.get_ptr());
      changeEta(idx, eta);
   }

   usetup = false;

   MSG_DEBUG(std::cout << "DSLUFA01\tupdated\t\tstability = " << stability()
             << std::endl;)

   return status();
}

template <class R>
void SLUFactor<R>::clear()
{

   this->rowMemMult    = 5;          /* factor of minimum Memory * #of nonzeros */
   this->colMemMult    = 5;          /* factor of minimum Memory * #of nonzeros */
   this->lMemMult      = 1;          /* factor of minimum Memory * #of nonzeros */

   this->l.firstUpdate = 0;
   this->l.firstUnused = 0;
   this->thedim        = 0;

   epsilon       = Param::epsilonFactorization();
   usetup        = false;
   this->maxabs        = 1;
   this->initMaxabs    = 1;
   lastThreshold = minThreshold;
   minStability  = MINSTABILITY;
   this->stat          = this->UNLOADED;

   vec.clear();
   eta.clear();
   ssvec.clear();
   forest.clear();

   this->u.row.size    = 100;
   this->u.col.size    = 100;
   this->l.size        = 100;
   this->l.startSize   = 100;

   if(this->l.ridx)
      spx_free(this->l.ridx);

   if(this->l.rbeg)
      spx_free(this->l.rbeg);

   if(this->l.rorig)
      spx_free(this->l.rorig);

   if(this->l.rperm)
      spx_free(this->l.rperm);

   if(!this->u.row.val.empty())
      this->u.row.val.clear();

   if(this->u.row.idx)
      spx_free(this->u.row.idx);

   if(this->u.col.idx)
      spx_free(this->u.col.idx);

   if(this->l.val.empty())
      this->l.val.clear();

   if(this->l.idx)
      spx_free(this->l.idx);

   if(this->l.start)
      spx_free(this->l.start);

   if(this->l.row)
      spx_free(this->l.row);

   // G clear() is used in constructor of SLUFactor<R> so we have to
   // G clean up if anything goes wrong here
   try
   {
      this->u.row.val.resize(this->u.row.size);
      spx_alloc(this->u.row.idx, this->u.row.size);
      spx_alloc(this->u.col.idx, this->u.col.size);

      this->l.val.resize(this->l.size);
      spx_alloc(this->l.idx,   this->l.size);
      spx_alloc(this->l.start, this->l.startSize);
      spx_alloc(this->l.row,   this->l.startSize);
   }
   catch(const SPxMemoryException& x)
   {
      freeAll();
      throw x;
   }
}

/** assignment used to implement operator=() and copy constructor.
 *  If this is initialised, freeAll() has to be called before.
 *  Class objects from SLUFactor<R> are not copied here.
 */
template <class R>
void SLUFactor<R>::assign(const SLUFactor<R>& old)
{
   this->spxout = old.spxout;

   solveTime = TimerFactory::createTimer(old.solveTime->type());
   this->factorTime = TimerFactory::createTimer(old.factorTime->type());

   // slufactor
   uptype        = old.uptype;
   minThreshold  = old.minThreshold;
   minStability  = old.minStability;
   epsilon       = old.epsilon;
   lastThreshold = old.lastThreshold;

   // clufactor
   this->stat          = old.stat;
   this->thedim        = old.thedim;
   this->nzCnt         = old.nzCnt;
   this->initMaxabs    = old.initMaxabs;
   this->maxabs        = old.maxabs;
   this->rowMemMult    = old.rowMemMult;
   this->colMemMult    = old.colMemMult;
   this->lMemMult      = old.lMemMult;

   spx_alloc(this->row.perm, this->thedim);
   spx_alloc(this->row.orig, this->thedim);
   spx_alloc(this->col.perm, this->thedim);
   spx_alloc(this->col.orig, this->thedim);
   // spx_alloc(this->diag,     this->thedim);
   this->diag.reserve(this->thedim); // small performance improvement before copying

   memcpy(this->row.perm, old.row.perm, (unsigned int)this->thedim * sizeof(*this->row.perm));
   memcpy(this->row.orig, old.row.orig, (unsigned int)this->thedim * sizeof(*this->row.orig));
   memcpy(this->col.perm, old.col.perm, (unsigned int)this->thedim * sizeof(*this->col.perm));
   memcpy(this->col.orig, old.col.orig, (unsigned int)this->thedim * sizeof(*this->col.orig));

   this->diag = old.diag;

   this->work = vec.get_ptr();

   /* setup U
    */
   this->u.row.size = old.u.row.size;
   this->u.row.used = old.u.row.used;

   spx_alloc(this->u.row.elem,  this->thedim);
   this->u.row.val.reserve(this->u.row.size); // small performance improvement
   spx_alloc(this->u.row.idx,   this->u.row.size);
   spx_alloc(this->u.row.start, this->thedim + 1);
   spx_alloc(this->u.row.len,   this->thedim + 1);
   spx_alloc(this->u.row.max,   this->thedim + 1);

   memcpy(this->u.row.elem,  old.u.row.elem,
          (unsigned int)this->thedim       * sizeof(*this->u.row.elem));
   this->u.row.val = old.u.row.val;
   memcpy(this->u.row.idx,   old.u.row.idx,
          (unsigned int)this->u.row.size   * sizeof(*this->u.row.idx));
   memcpy(this->u.row.start, old.u.row.start,
          (unsigned int)(this->thedim + 1) * sizeof(*this->u.row.start));
   memcpy(this->u.row.len,   old.u.row.len,
          (unsigned int)(this->thedim + 1) * sizeof(*this->u.row.len));
   memcpy(this->u.row.max,   old.u.row.max,
          (unsigned int)(this->thedim + 1) * sizeof(*this->u.row.max));

   // need to make row list ok ?
   if(this->thedim > 0 && this->stat == this->OK)
   {
      this->u.row.list.idx = old.u.row.list.idx; // .idx neu

      const typename CLUFactor<R>::Dring* oring = &old.u.row.list;
      typename CLUFactor<R>::Dring*       ring  = &this->u.row.list;

      while(oring->next != &old.u.row.list)
      {
         ring->next       = &this->u.row.elem[oring->next->idx];
         ring->next->prev = ring;
         oring            = oring->next;
         ring             = ring->next;
      }

      ring->next       = &this->u.row.list;
      ring->next->prev = ring;
   }

   this->u.col.size = old.u.col.size;
   this->u.col.used = old.u.col.used;

   spx_alloc(this->u.col.elem,  this->thedim);
   spx_alloc(this->u.col.idx,   this->u.col.size);
   spx_alloc(this->u.col.start, this->thedim + 1);
   spx_alloc(this->u.col.len,   this->thedim + 1);
   spx_alloc(this->u.col.max,   this->thedim + 1);

   if(!old.u.col.val.empty())
   {
      this->u.col.val.reserve(this->u.col.size); // small performance improvement before copying
      this->u.col.val = old.u.col.val;
   }
   else
   {
      this->u.col.val.clear();
   }

   memcpy(this->u.col.elem,  old.u.col.elem,
          (unsigned int)this->thedim       * sizeof(*this->u.col.elem));
   memcpy(this->u.col.idx,   old.u.col.idx,
          (unsigned int)this->u.col.size   * sizeof(*this->u.col.idx));
   memcpy(this->u.col.start, old.u.col.start,
          (unsigned int)(this->thedim + 1) * sizeof(*this->u.col.start));
   memcpy(this->u.col.len,   old.u.col.len,
          (unsigned int)(this->thedim + 1) * sizeof(*this->u.col.len));
   memcpy(this->u.col.max,   old.u.col.max,
          (unsigned int)(this->thedim + 1) * sizeof(*this->u.col.max));

   // need to make col list ok
   if(this->thedim > 0 && this->stat == this->OK)
   {
      this->u.col.list.idx = old.u.col.list.idx; // .idx neu

      const typename CLUFactor<R>::Dring* oring = &old.u.col.list;
      typename CLUFactor<R>::Dring*       ring  = &this->u.col.list;

      while(oring->next != &old.u.col.list)
      {
         ring->next       = &this->u.col.elem[oring->next->idx];
         ring->next->prev = ring;
         oring            = oring->next;
         ring             = ring->next;
      }

      ring->next       = &this->u.col.list;
      ring->next->prev = ring;
   }

   /* Setup L
    */
   this->l.size        = old.l.size;
   this->l.startSize   = old.l.startSize;
   this->l.firstUpdate = old.l.firstUpdate;
   this->l.firstUnused = old.l.firstUnused;
   this->l.updateType  = old.l.updateType;

   this->l.val.reserve(this->l.size); // small performance improvement for copying
   spx_alloc(this->l.idx,   this->l.size);
   spx_alloc(this->l.start, this->l.startSize);
   spx_alloc(this->l.row,   this->l.startSize);

   this->l.val = old.l.val;
   memcpy(this->l.idx,   old.l.idx, (unsigned int)this->l.size      * sizeof(*this->l.idx));
   memcpy(this->l.start, old.l.start, (unsigned int)this->l.startSize * sizeof(*this->l.start));
   memcpy(this->l.row,   old.l.row, (unsigned int)this->l.startSize * sizeof(*this->l.row));

   if(!this->l.rval.empty())
   {
      assert(old.l.ridx  != 0);
      assert(old.l.rbeg  != 0);
      assert(old.l.rorig != 0);
      assert(old.l.rperm != 0);

      int memsize = this->l.start[this->l.firstUpdate];

      this->l.rval.reserve(memsize); // small performance improvement for copying
      spx_alloc(this->l.ridx,  memsize);
      spx_alloc(this->l.rbeg,  this->thedim + 1);
      spx_alloc(this->l.rorig, this->thedim);
      spx_alloc(this->l.rperm, this->thedim);

      this->l.rval = old.l.rval;
      memcpy(this->l.ridx,  old.l.ridx, (unsigned int)memsize     * sizeof(*this->l.ridx));
      memcpy(this->l.rbeg,  old.l.rbeg, (unsigned int)(this->thedim + 1) * sizeof(*this->l.rbeg));
      memcpy(this->l.rorig, old.l.rorig, (unsigned int)this->thedim      * sizeof(*this->l.rorig));
      memcpy(this->l.rperm, old.l.rperm, (unsigned int)this->thedim      * sizeof(*this->l.rperm));
   }
   else
   {
      assert(old.l.ridx  == 0);
      assert(old.l.rbeg  == 0);
      assert(old.l.rorig == 0);
      assert(old.l.rperm == 0);

      this->l.ridx  = 0;
      this->l.rbeg  = 0;
      this->l.rorig = 0;
      this->l.rperm = 0;
   }

   assert(this->row.perm != 0);
   assert(this->row.orig != 0);
   assert(this->col.perm != 0);
   assert(this->col.orig != 0);

   assert(this->u.row.elem  != 0);
   assert(this->u.row.idx   != 0);
   assert(this->u.row.start != 0);
   assert(this->u.row.len   != 0);
   assert(this->u.row.max   != 0);

   assert(this->u.col.elem  != 0);
   assert(this->u.col.idx   != 0);
   assert(this->u.col.start != 0);
   assert(this->u.col.len   != 0);
   assert(this->u.col.max   != 0);

   assert(this->l.idx   != 0);
   assert(this->l.start != 0);
   assert(this->l.row   != 0);

}

template <class R>
SLUFactor<R>& SLUFactor<R>::operator=(const SLUFactor<R>& old)
{

   if(this != &old)
   {
      // we don't need to copy them, because they are temporary vectors
      vec.clear();
      ssvec.clear();

      eta    = old.eta;
      forest = old.forest;

      timerType = old.timerType;

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

template <class R>
SLUFactor<R>::SLUFactor()
   : vec(1)
   , ssvec(1)
   , usetup(false)
   , uptype(FOREST_TOMLIN)
   , eta(1)
   , forest(1)
   , minThreshold(0.01)
   , timerType(Timer::USER_TIME)
{
   this->row.perm    = 0;
   this->row.orig    = 0;
   this->col.perm    = 0;
   this->col.orig    = 0;
   this->u.row.elem  = 0;
   this->u.row.idx   = 0;
   this->u.row.start = 0;
   this->u.row.len   = 0;
   this->u.row.max   = 0;
   this->u.col.elem  = 0;
   this->u.col.idx   = 0;
   this->u.col.start = 0;
   this->u.col.len   = 0;
   this->u.col.max   = 0;
   this->l.idx       = 0;
   this->l.start     = 0;
   this->l.row       = 0;
   this->l.ridx      = 0;
   this->l.rbeg      = 0;
   this->l.rorig     = 0;
   this->l.rperm     = 0;

   this->nzCnt  = 0;
   this->thedim = 0;

   try
   {
      solveTime = TimerFactory::createTimer(timerType);
      this->factorTime = TimerFactory::createTimer(timerType);
      spx_alloc(this->row.perm, this->thedim);
      spx_alloc(this->row.orig, this->thedim);
      spx_alloc(this->col.perm, this->thedim);
      spx_alloc(this->col.orig, this->thedim);

      this->diag.resize(this->thedim);

      this->work = vec.get_ptr();

      this->u.row.size = 1;
      this->u.row.used = 0;
      spx_alloc(this->u.row.elem,  this->thedim);
      this->u.row.val.resize(this->u.row.size);
      spx_alloc(this->u.row.idx,   this->u.row.size);
      spx_alloc(this->u.row.start, this->thedim + 1);
      spx_alloc(this->u.row.len,   this->thedim + 1);
      spx_alloc(this->u.row.max,   this->thedim + 1);

      this->u.row.list.idx      = this->thedim;
      this->u.row.start[this->thedim] = 0;
      this->u.row.max  [this->thedim] = 0;
      this->u.row.len  [this->thedim] = 0;

      this->u.col.size = 1;
      this->u.col.used = 0;
      spx_alloc(this->u.col.elem,  this->thedim);
      spx_alloc(this->u.col.idx,   this->u.col.size);
      spx_alloc(this->u.col.start, this->thedim + 1);
      spx_alloc(this->u.col.len,   this->thedim + 1);
      spx_alloc(this->u.col.max,   this->thedim + 1);

      this->u.col.list.idx      = this->thedim;
      this->u.col.start[this->thedim] = 0;
      this->u.col.max[this->thedim]   = 0;
      this->u.col.len[this->thedim]   = 0;

      this->l.size = 1;

      this->l.val.resize(this->l.size);
      spx_alloc(this->l.idx, this->l.size);

      this->l.startSize   = 1;
      this->l.firstUpdate = 0;
      this->l.firstUnused = 0;

      spx_alloc(this->l.start, this->l.startSize);
      spx_alloc(this->l.row,   this->l.startSize);
   }
   catch(const SPxMemoryException& x)
   {
      freeAll();
      throw x;
   }

   this->l.ridx  = 0;
   this->l.rbeg  = 0;
   this->l.rorig = 0;
   this->l.rperm = 0;

   SLUFactor<R>::clear(); // clear() is virtual

   this->factorCount = 0;
   this->hugeValues = 0;
   solveCount  = 0;
   assert(this->row.perm != 0);

   assert(this->row.orig != 0);
   assert(this->col.perm != 0);
   assert(this->col.orig != 0);

   assert(this->u.row.elem  != 0);
   assert(this->u.row.idx   != 0);
   assert(this->u.row.start != 0);
   assert(this->u.row.len   != 0);
   assert(this->u.row.max   != 0);

   assert(this->u.col.elem  != 0);
   assert(this->u.col.idx   != 0);
   assert(this->u.col.start != 0);
   assert(this->u.col.len   != 0);
   assert(this->u.col.max   != 0);

   assert(this->l.idx   != 0);
   assert(this->l.start != 0);
   assert(this->l.row   != 0);

   assert(SLUFactor<R>::isConsistent());
}

template <class R>
SLUFactor<R>::SLUFactor(const SLUFactor<R>& old)
   : SLinSolver<R>(old)
   , vec(1)     // we don't need to copy it, because they are temporary vectors
   , ssvec(1)   // we don't need to copy it, because they are temporary vectors
   , usetup(old.usetup)
   , eta(old.eta)
   , forest(old.forest)
   , timerType(old.timerType)
{
   this->row.perm    = 0;
   this->row.orig    = 0;
   this->col.perm    = 0;
   this->col.orig    = 0;
   this->u.row.elem  = 0;
   this->u.row.val.clear();
   this->u.row.idx   = 0;
   this->u.row.start = 0;
   this->u.row.len   = 0;
   this->u.row.max   = 0;
   this->u.col.elem  = 0;
   this->u.col.idx   = 0;
   this->u.col.start = 0;
   this->u.col.len   = 0;
   this->u.col.max   = 0;
   this->l.idx       = 0;
   this->l.start     = 0;
   this->l.row       = 0;
   this->l.ridx      = 0;
   this->l.rbeg      = 0;
   this->l.rorig     = 0;
   this->l.rperm     = 0;

   solveCount = 0;

   try
   {
      assign(old);
   }
   catch(const SPxMemoryException& x)
   {
      freeAll();
      throw x;
   }

   assert(SLUFactor<R>::isConsistent());
}

template <class R>
void SLUFactor<R>::freeAll()
{

   if(this->row.perm)
      spx_free(this->row.perm);

   if(this->row.orig)
      spx_free(this->row.orig);

   if(this->col.perm)
      spx_free(this->col.perm);

   if(this->col.orig)
      spx_free(this->col.orig);

   if(this->u.row.elem)
      spx_free(this->u.row.elem);

   if(!this->u.row.val.empty())
      this->u.row.val.clear();

   if(this->u.row.idx)
      spx_free(this->u.row.idx);

   if(this->u.row.start)
      spx_free(this->u.row.start);

   if(this->u.row.len)
      spx_free(this->u.row.len);

   if(this->u.row.max)
      spx_free(this->u.row.max);

   if(this->u.col.elem)
      spx_free(this->u.col.elem);

   if(this->u.col.idx)
      spx_free(this->u.col.idx);

   if(this->u.col.start)
      spx_free(this->u.col.start);

   if(this->u.col.len)
      spx_free(this->u.col.len);

   if(this->u.col.max)
      spx_free(this->u.col.max);

   if(!this->l.val.empty())
      this->l.val.clear();

   if(this->l.idx)
      spx_free(this->l.idx);

   if(this->l.start)
      spx_free(this->l.start);

   if(this->l.row)
      spx_free(this->l.row);


   if(!this->u.col.val.empty())
      this->u.col.val.clear();

   if(this->l.ridx)
      spx_free(this->l.ridx);

   if(this->l.rbeg)
      spx_free(this->l.rbeg);

   if(this->l.rorig)
      spx_free(this->l.rorig);

   if(this->l.rperm)
      spx_free(this->l.rperm);

   if(solveTime)
   {
      solveTime->~Timer();
      spx_free(solveTime);
   }

   if(this->factorTime)
   {
      this->factorTime->~Timer();
      spx_free(this->factorTime);
   }
}

template <class R>
SLUFactor<R>::~SLUFactor<R>()
{
   freeAll();
}

template <class R>
static R betterThreshold(R th)
{
   assert(th < R(1.0));

   if(LT(th, R(0.1)))
      th *= R(10.0);
   else if(LT(th, R(0.9)))
      th = (th + R(1.0)) / R(2.0);
   else if(LT(th, R(0.999)))
      th = R(0.99999);

   assert(th < R(1.0));

   return th;
}

template <class R>
typename SLUFactor<R>::Status SLUFactor<R>::load(const SVectorBase<R>* matrix[], int dm)
{
   assert(dm     >= 0);
   assert(matrix != 0);

   R lastStability = stability();

   initDR(this->u.row.list);
   initDR(this->u.col.list);

   usetup        = false;
   this->l.updateType  = uptype;
   this->l.firstUpdate = 0;
   this->l.firstUnused = 0;

   if(dm != this->thedim)
   {
      clear();

      this->thedim = dm;
      vec.reDim(this->thedim);
      ssvec.reDim(this->thedim);
      eta.reDim(this->thedim);
      forest.reDim(this->thedim);
      this->work = vec.get_ptr();

      spx_realloc(this->row.perm, this->thedim);
      spx_realloc(this->row.orig, this->thedim);
      spx_realloc(this->col.perm, this->thedim);
      spx_realloc(this->col.orig, this->thedim);
      this->diag.resize(this->thedim);

      spx_realloc(this->u.row.elem,  this->thedim);
      spx_realloc(this->u.row.len,   this->thedim + 1);
      spx_realloc(this->u.row.max,   this->thedim + 1);
      spx_realloc(this->u.row.start, this->thedim + 1);

      spx_realloc(this->u.col.elem,  this->thedim);
      spx_realloc(this->u.col.len,   this->thedim + 1);
      spx_realloc(this->u.col.max,   this->thedim + 1);
      spx_realloc(this->u.col.start, this->thedim + 1);

      this->l.startSize = this->thedim + MAXUPDATES;

      spx_realloc(this->l.row,   this->l.startSize);
      spx_realloc(this->l.start, this->l.startSize);
   }
   // the last factorization was reasonably stable, so we decrease the Markowitz threshold (stored in lastThreshold) in
   // order to favour sparsity
   else if(lastStability > 2.0 * minStability)
   {
      // we reset lastThreshold to its previous value in the sequence minThreshold, betterThreshold(minThreshold),
      // betterThreshold(betterThreshold(minThreshold)), ...
      R last   = minThreshold;
      R better = betterThreshold(last);

      while(better < lastThreshold)
      {
         last   = better;
         better = betterThreshold(last);
      }

      lastThreshold = last;

      // we reset the minimum stability (which might have been decreased below) to ensure that the increased sparsity
      // does not hurt the stability
      minStability  = 2 * MINSTABILITY;
   }

   this->u.row.list.idx      = this->thedim;
   this->u.row.start[this->thedim] = 0;
   this->u.row.max[this->thedim]   = 0;
   this->u.row.len[this->thedim]   = 0;

   this->u.col.list.idx      = this->thedim;
   this->u.col.start[this->thedim] = 0;
   this->u.col.max[this->thedim]   = 0;
   this->u.col.len[this->thedim]   = 0;

   for(;;)
   {
      ///@todo if the factorization fails with stat = SINGULAR, distinuish between proven singularity (e.g., because of
      ///an empty column) and singularity due to numerics, that could be avoided by changing minStability and
      ///lastThreshold; in the first case, we want to abort, otherwise change the numerics
      this->stat = this->OK;
      this->factor(matrix, lastThreshold, epsilon);

      // finish if the factorization is stable
      if(stability() >= minStability)
         break;

      // otherwise, we increase the Markowitz threshold
      R x = lastThreshold;
      lastThreshold = betterThreshold(lastThreshold);

      // until it doesn't change anymore at its maximum value
      if(EQ(x, lastThreshold))
         break;

      // we relax the stability requirement
      minStability /= 2.0;

      MSG_INFO3((*this->spxout), (*this->spxout) <<
                "ISLUFA01 refactorizing with increased Markowitz threshold: "
                << lastThreshold << std::endl;)
   }

   MSG_DEBUG(std::cout << "DSLUFA02 threshold = " << lastThreshold
             << "\tstability = " << stability()
             << "\tminStability = " << minStability << std::endl;)
   MSG_DEBUG(
   {
      int i;
      FILE* fl = fopen("dump.lp", "w");
      std::cout << "DSLUFA03 Basis:\n";
      int j = 0;

      for(i = 0; i < dim(); ++i)
         j += matrix[i]->size();
      for(i = 0; i < dim(); ++i)
      {
         for(j = 0; j < matrix[i]->size(); ++j)
         {
            fprintf(fl, "%8d  %8d  ",
                    i + 1, matrix[i]->index(j) + 1);
            std::cout << matrix[i]->value(j) << std::endl;
         }
      }
      fclose(fl);
      std::cout << "DSLUFA04 LU-Factors:" << std::endl;
      dump();

      std::cout << "DSLUFA05 threshold = " << lastThreshold
                << "\tstability = " << stability() << std::endl;
   }
   )

   assert(isConsistent());
   return Status(this->stat);
}


template <class R>
bool SLUFactor<R>::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS
   return CLUFactor<R>::isConsistent();
#else
   return true;
#endif
}

template <class R>
void SLUFactor<R>::dump() const
{
   CLUFactor<R>::dump();
}
} // namespace soplex
