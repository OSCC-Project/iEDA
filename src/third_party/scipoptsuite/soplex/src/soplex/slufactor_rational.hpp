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

/**@file slufactor_rational.hpp
 * @todo SLUFactorRational seems to be partly an wrapper for CLUFactorRational (was C).
 *       This should be properly integrated and demangled.
 * @todo Does is make sense, to call x.clear() when next x.altValues()
 *       is called.
 */

#include <assert.h>
#include <sstream>
#include "soplex/cring.h"
#include "soplex/spxdefines.h"

#ifdef SOPLEX_DEBUG
#include <stdio.h>
#endif

namespace soplex
{
#define MINSTABILITYRAT    REAL(4e-2)

inline void SLUFactorRational::solveRight(VectorRational& x, const VectorRational& b) //const
{

   solveTime->start();

   vec = b;
   CLUFactorRational::solveRight(x.get_ptr(), vec.get_ptr());

   solveCount++;
   solveTime->stop();
}

inline void SLUFactorRational::solveRight(SSVectorRational& x, const SVectorRational& b) //const
{

   solveTime->start();

   vec.assign(b);
   x.clear();
   CLUFactorRational::solveRight(x.altValues(), vec.get_ptr());

   solveCount++;
   solveTime->stop();
}

inline void SLUFactorRational::solveRight4update(SSVectorRational& x, const SVectorRational& b)
{

   solveTime->start();

   int m;
   int n;
   int f = 0;

   x.clear();
   ssvec = b;
   n = ssvec.size();

   if(l.updateType == ETA)
   {
      m = vSolveRight4update(x.altValues(), x.altIndexMem(),
                             ssvec.altValues(), ssvec.altIndexMem(), n, 0, 0, 0);
      x.setSize(m);
      //x.forceSetup();
      x.unSetup();
      eta.setup_and_assign(x);
   }
   else
   {
      forest.clear();
      m = vSolveRight4update(x.altValues(), x.altIndexMem(),
                             ssvec.altValues(), ssvec.altIndexMem(), n,
                             forest.altValues(), &f, forest.altIndexMem());
      forest.setSize(f);
      forest.forceSetup();
      x.setSize(m);
      x.forceSetup();
   }

   usetup = true;

   solveCount++;
   solveTime->stop();
}

inline void SLUFactorRational::solve2right4update(
   SSVectorRational&      x,
   VectorRational&        y,
   const SVectorRational& b,
   SSVectorRational&      rhs)
{

   solveTime->start();

   int  m;
   int  n;
   int  f = 0;
   int* sidx = ssvec.altIndexMem();
   int  rsize = rhs.size();
   int* ridx = rhs.altIndexMem();

   x.clear();
   y.clear();
   usetup = true;
   ssvec = b;

   if(l.updateType == ETA)
   {
      n = ssvec.size();
      m = vSolveRight4update2(x.altValues(), x.altIndexMem(),
                              ssvec.get_ptr(), sidx, n, y.get_ptr(),
                              rhs.altValues(), ridx, rsize, 0, 0, 0);
      x.setSize(m);
      //      x.forceSetup();
      x.unSetup();
      eta.setup_and_assign(x);
   }
   else
   {
      forest.clear();
      n = ssvec.size();
      m = vSolveRight4update2(x.altValues(), x.altIndexMem(),
                              ssvec.get_ptr(), sidx, n, y.get_ptr(),
                              rhs.altValues(), ridx, rsize,
                              forest.altValues(), &f, forest.altIndexMem());
      x.setSize(m);
      x.forceSetup();
      forest.setSize(f);
      forest.forceSetup();
   }

   solveCount++;
   solveTime->stop();
}

inline void SLUFactorRational::solve3right4update(
   SSVectorRational&      x,
   VectorRational&        y,
   VectorRational&        y2,
   const SVectorRational& b,
   SSVectorRational&      rhs,
   SSVectorRational&      rhs2)
{

   solveTime->start();

   int  m;
   int  n;
   int  f;
   int* sidx = ssvec.altIndexMem();
   int  rsize = rhs.size();
   int* ridx = rhs.altIndexMem();
   int rsize2 = rhs2.size();
   int* ridx2 = rhs2.altIndexMem();

   x.clear();
   y.clear();
   y2.clear();
   usetup = true;
   ssvec = b;

   if(l.updateType == ETA)
   {
      n = ssvec.size();
      m = vSolveRight4update3(x.altValues(), x.altIndexMem(), ssvec.get_ptr(), sidx, n,
                              y.get_ptr(), rhs.altValues(), ridx, rsize,
                              y2.get_ptr(), rhs2.altValues(), ridx2, rsize2,
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
      m = vSolveRight4update3(x.altValues(), x.altIndexMem(), ssvec.get_ptr(), sidx, n,
                              y.get_ptr(), rhs.altValues(), ridx, rsize,
                              y2.get_ptr(), rhs2.altValues(), ridx2, rsize2,
                              forest.altValues(), &f, forest.altIndexMem());
      x.setSize(m);
      x.forceSetup();
      forest.setSize(f);
      forest.forceSetup();
   }

   solveCount++;
   solveTime->stop();
}

inline void SLUFactorRational::solveLeft(VectorRational& x, const VectorRational& b) //const
{

   solveTime->start();

   vec = b;
   ///@todo Why is x.clear() here used and not with solveRight() ?
   x.clear();
   CLUFactorRational::solveLeft(x.get_ptr(), vec.get_ptr());

   solveCount++;
   solveTime->stop();
}

inline void SLUFactorRational::solveLeft(SSVectorRational& x, const SVectorRational& b) //const
{

   solveTime->start();

   ssvec.assign(b);

   x.clear();
   int sz = ssvec.size(); // see .altValues()
   int n = vSolveLeft(x.altValues(), x.altIndexMem(),
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

inline void SLUFactorRational::solveLeft(
   SSVectorRational&      x,
   VectorRational&        y,
   const SVectorRational& rhs1,
   SSVectorRational&      rhs2) //const
{

   solveTime->start();

   int   n;
   Rational* svec = ssvec.altValues();
   int*  sidx = ssvec.altIndexMem();
   int   rn   = rhs2.size();
   int*  ridx = rhs2.altIndexMem();

   x.clear();
   y.clear();
   ssvec.assign(rhs1);
   n = ssvec.size(); // see altValues();
   n = vSolveLeft2(x.altValues(), x.altIndexMem(), svec, sidx, n,
                   y.get_ptr(), rhs2.altValues(), ridx, rn);

   x.setSize(n);

   if(n > 0)
      x.forceSetup();
   else
      x.unSetup();

   rhs2.setSize(0);
   rhs2.forceSetup();
   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount++;
   solveTime->stop();
}

inline void SLUFactorRational::solveLeft(
   SSVectorRational&      x,
   VectorRational&        y,
   VectorRational&        z,
   const SVectorRational& rhs1,
   SSVectorRational&      rhs2,
   SSVectorRational&      rhs3)
{

   solveTime->start();

   int   n;
   Rational* svec = ssvec.altValues();
   int*  sidx = ssvec.altIndexMem();

   x.clear();
   y.clear();
   z.clear();
   ssvec.assign(rhs1);
   n = ssvec.size(); // see altValues();
   n = vSolveLeft3(x.altValues(), x.altIndexMem(), svec, sidx, n,
                   y.get_ptr(), rhs2.altValues(), rhs2.altIndexMem(), rhs2.size(),
                   z.get_ptr(), rhs3.altValues(), rhs3.altIndexMem(), rhs3.size());

   x.setSize(n);

   if(n > 0)
      x.forceSetup();
   else
      x.unSetup();

   ssvec.setSize(0);
   ssvec.forceSetup();

   solveCount++;
   solveTime->stop();
}

inline Rational SLUFactorRational::stability() const
{

   if(status() != OK)
      return 0;

   if(maxabs < initMaxabs)
      return 1;

   return initMaxabs / maxabs;
}

inline std::string SLUFactorRational::statistics() const
{
   std::stringstream s;
   s  << "Factorizations     : " << std::setw(10) << getFactorCount() << std::endl
      << "  Time spent       : " << std::setw(10) << std::fixed << std::setprecision(
         2) << getFactorTime() << std::endl
      << "Solves             : " << std::setw(10) << getSolveCount() << std::endl
      << "  Time spent       : " << std::setw(10) << getSolveTime() << std::endl;

   return s.str();
}

inline void SLUFactorRational::changeEta(int idx, SSVectorRational& et)
{

   int es = et.size(); // see altValues()
   update(idx, et.altValues(), et.altIndexMem(), es);
   et.setSize(0);
   et.forceSetup();
}

inline SLUFactorRational::Status SLUFactorRational::change(
   int             idx,
   const SVectorRational&  subst,
   const SSVectorRational* e)
{

   // BH 2005-08-23: The boolean usetup indicates that an "update vector"
   // has been set up. I suppose that SSVectorRational forest is this
   // update vector, which is set up by solveRight4update() and
   // solve2right4update() in order to optimize the basis update.

   if(usetup)
   {
      if(l.updateType == FOREST_TOMLIN)               // FOREST_TOMLIN updates
      {
         // BH 2005-08-19: The size of a SSVectorRational is the size of the
         // index set, i.e.  the number of nonzeros which is only
         // defined if the SSVectorRational is set up.  Since
         // SSVectorRational::altValues() calls unSetup() the size needs to be
         // stored before the following call.
         int fsize = forest.size(); // see altValues()
         forestUpdate(idx, forest.altValues(), fsize, forest.altIndexMem());
         forest.setSize(0);
         forest.forceSetup();
      }
      else
      {
         assert(l.updateType == ETA);
         changeEta(idx, eta);
      }
   }
   else if(e != 0)                                    // ETA updates
   {
      l.updateType = ETA;
      updateNoClear(idx, e->values(), e->indexMem(), e->size());
      l.updateType = uptype;
   }
   else if(l.updateType == FOREST_TOMLIN)             // FOREST_TOMLIN updates
   {
      assert(0);  // probably this part is never called.
      // forestUpdate() with the last parameter set to NULL should fail.
      forest = subst;
      CLUFactorRational::solveLright(forest.altValues());
      forestUpdate(idx, forest.altValues(), 0, nullptr);
      forest.setSize(0);
      forest.forceSetup();
   }
   else                                               // ETA updates
   {
      assert(l.updateType == ETA);
      vec = subst;
      eta.clear();
      CLUFactorRational::solveRight(eta.altValues(), vec.get_ptr());
      changeEta(idx, eta);
   }

   usetup = false;

   MSG_DEBUG(std::cout << "DSLUFA01\tupdated\t\tstability = " << stability()
             << std::endl;)

   return status();
}

inline void SLUFactorRational::clear()
{

   rowMemMult    = 5;          /* factor of minimum Memory * #of nonzeros */
   colMemMult    = 5;          /* factor of minimum Memory * #of nonzeros */
   lMemMult      = 1;          /* factor of minimum Memory * #of nonzeros */

   l.firstUpdate = 0;
   l.firstUnused = 0;
   thedim        = 0;

   usetup        = false;
   maxabs        = 1;
   initMaxabs    = 1;
   lastThreshold = minThreshold;
   minStability  = MINSTABILITYRAT;
   stat          = UNLOADED;

   vec.clear();
   eta.clear();
   ssvec.clear();
   forest.clear();

   u.col.size    = 100;
   l.startSize   = 100;

   l.rval.reDim(0);

   if(l.ridx)
      spx_free(l.ridx);

   if(l.rbeg)
      spx_free(l.rbeg);

   if(l.rorig)
      spx_free(l.rorig);

   if(l.rperm)
      spx_free(l.rperm);

   if(u.row.idx)
      spx_free(u.row.idx);

   if(u.col.idx)
      spx_free(u.col.idx);

   if(l.idx)
      spx_free(l.idx);

   if(l.start)
      spx_free(l.start);

   if(l.row)
      spx_free(l.row);

   // G clear() is used in constructor of SLUFactorRational so we have to
   // G clean up if anything goes wrong here
   try
   {
      u.row.val.reDim(100);
      spx_alloc(u.row.idx, u.row.val.dim());
      spx_alloc(u.col.idx, u.col.size);

      l.val.reDim(100);
      spx_alloc(l.idx,   l.val.dim());
      spx_alloc(l.start, l.startSize);
      spx_alloc(l.row,   l.startSize);
   }
   catch(const SPxMemoryException& x)
   {
      freeAll();
      throw x;
   }
}

/** assignment used to implement operator=() and copy constructor.
 *  If this is initialised, freeAll() has to be called before.
 *  Class objects from SLUFactorRational are not copied here.
 */
inline void SLUFactorRational::assign(const SLUFactorRational& old)
{
   unsigned int thediminc;

   // slufactor_rational
   uptype        = old.uptype;
   minThreshold  = old.minThreshold;
   minStability  = old.minStability;
   lastThreshold = old.lastThreshold;

   // clufactor_rational
   stat          = old.stat;
   thedim        = old.thedim;
   nzCnt         = old.nzCnt;
   initMaxabs    = old.initMaxabs;
   maxabs        = old.maxabs;
   rowMemMult    = old.rowMemMult;
   colMemMult    = old.colMemMult;
   lMemMult      = old.lMemMult;
   factorCount   = old.factorCount;
   factorTime    = old.factorTime;
   timeLimit     = old.timeLimit;

   spx_alloc(row.perm, thedim);
   spx_alloc(row.orig, thedim);
   spx_alloc(col.perm, thedim);
   spx_alloc(col.orig, thedim);

   memcpy(row.perm, old.row.perm, (unsigned int)thedim * sizeof(*row.perm));
   memcpy(row.orig, old.row.orig, (unsigned int)thedim * sizeof(*row.orig));
   memcpy(col.perm, old.col.perm, (unsigned int)thedim * sizeof(*col.perm));
   memcpy(col.orig, old.col.orig, (unsigned int)thedim * sizeof(*col.orig));
   diag = old.diag;

   work = vec.get_ptr();

   /* setup U
    */
   thediminc = (unsigned int)(thedim + 1);
   u.row.used = old.u.row.used;

   spx_alloc(u.row.elem,  thedim);
   spx_alloc(u.row.idx,   u.row.val.dim());
   spx_alloc(u.row.start, thediminc);
   spx_alloc(u.row.len, thediminc);
   spx_alloc(u.row.max, thediminc);

   memcpy(u.row.elem,  old.u.row.elem, (unsigned int)thedim       * sizeof(*u.row.elem));
   u.row.val = old.u.row.val;
   memcpy(u.row.idx,   old.u.row.idx, (unsigned int)u.row.val.dim()   * sizeof(*u.row.idx));
   memcpy(u.row.start, old.u.row.start, thediminc * sizeof(*u.row.start));
   memcpy(u.row.len,   old.u.row.len, thediminc * sizeof(*u.row.len));
   memcpy(u.row.max,   old.u.row.max, thediminc * sizeof(*u.row.max));

   // need to make row list ok ?
   if(thedim > 0 && stat == OK)
   {
      u.row.list.idx = old.u.row.list.idx; // .idx neu

      const Dring* oring = &old.u.row.list;
      Dring*       ring  = &u.row.list;

      while(oring->next != &old.u.row.list)
      {
         ring->next       = &u.row.elem[oring->next->idx];
         ring->next->prev = ring;
         oring            = oring->next;
         ring             = ring->next;
      }

      ring->next       = &u.row.list;
      ring->next->prev = ring;
   }

   u.col.size = old.u.col.size;
   u.col.used = old.u.col.used;

   spx_alloc(u.col.elem,  thedim);
   spx_alloc(u.col.idx,   u.col.size);
   spx_alloc(u.col.start, thediminc);
   spx_alloc(u.col.len, thediminc);
   spx_alloc(u.col.max, thediminc);
   u.col.val = old.u.col.val;

   memcpy(u.col.elem,  old.u.col.elem, (unsigned int)thedim       * sizeof(*u.col.elem));
   memcpy(u.col.idx,   old.u.col.idx, (unsigned int)u.col.size   * sizeof(*u.col.idx));
   memcpy(u.col.start, old.u.col.start, thediminc * sizeof(*u.col.start));
   memcpy(u.col.len,   old.u.col.len, thediminc * sizeof(*u.col.len));
   memcpy(u.col.max,   old.u.col.max, thediminc * sizeof(*u.col.max));

   // need to make col list ok
   if(thedim > 0 && stat == OK)
   {
      u.col.list.idx = old.u.col.list.idx; // .idx neu

      const Dring* oring = &old.u.col.list;
      Dring*       ring  = &u.col.list;

      while(oring->next != &old.u.col.list)
      {
         ring->next       = &u.col.elem[oring->next->idx];
         ring->next->prev = ring;
         oring            = oring->next;
         ring             = ring->next;
      }

      ring->next       = &u.col.list;
      ring->next->prev = ring;
   }

   /* Setup L
    */
   l.startSize   = old.l.startSize;
   l.firstUpdate = old.l.firstUpdate;
   l.firstUnused = old.l.firstUnused;
   l.updateType  = old.l.updateType;

   l.val = old.l.val;
   spx_alloc(l.idx,   l.val.dim());
   spx_alloc(l.start, l.startSize);
   spx_alloc(l.row,   l.startSize);

   memcpy(l.idx,   old.l.idx, (unsigned int)l.val.dim() * sizeof(*l.idx));
   memcpy(l.start, old.l.start, (unsigned int)l.startSize * sizeof(*l.start));
   memcpy(l.row,   old.l.row, (unsigned int)l.startSize * sizeof(*l.row));

   if(l.rval.dim() != 0)
   {
      assert(old.l.ridx  != 0);
      assert(old.l.rbeg  != 0);
      assert(old.l.rorig != 0);
      assert(old.l.rperm != 0);

      int memsize = l.start[l.firstUpdate];

      l.rval = old.l.rval;
      spx_alloc(l.ridx,  memsize);
      spx_alloc(l.rbeg,  thediminc);
      spx_alloc(l.rorig, thedim);
      spx_alloc(l.rperm, thedim);

      memcpy(l.ridx,  old.l.ridx, (unsigned int)memsize     * sizeof(*l.ridx));
      memcpy(l.rbeg,  old.l.rbeg, thediminc * sizeof(*l.rbeg));
      memcpy(l.rorig, old.l.rorig, (unsigned int)thedim      * sizeof(*l.rorig));
      memcpy(l.rperm, old.l.rperm, (unsigned int)thedim      * sizeof(*l.rperm));
   }
   else
   {
      assert(old.l.ridx  == 0);
      assert(old.l.rbeg  == 0);
      assert(old.l.rorig == 0);
      assert(old.l.rperm == 0);

      l.rval.reDim(0);
      l.ridx  = 0;
      l.rbeg  = 0;
      l.rorig = 0;
      l.rperm = 0;
   }

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

}

inline void SLUFactorRational::freeAll()
{

   if(row.perm)
      spx_free(row.perm);

   if(row.orig)
      spx_free(row.orig);

   if(col.perm)
      spx_free(col.perm);

   if(col.orig)
      spx_free(col.orig);

   if(u.row.elem)
      spx_free(u.row.elem);

   if(u.row.idx)
      spx_free(u.row.idx);

   if(u.row.start)
      spx_free(u.row.start);

   if(u.row.len)
      spx_free(u.row.len);

   if(u.row.max)
      spx_free(u.row.max);

   if(u.col.elem)
      spx_free(u.col.elem);

   if(u.col.idx)
      spx_free(u.col.idx);

   if(u.col.start)
      spx_free(u.col.start);

   if(u.col.len)
      spx_free(u.col.len);

   if(u.col.max)
      spx_free(u.col.max);

   if(l.idx)
      spx_free(l.idx);

   if(l.start)
      spx_free(l.start);

   if(l.row)
      spx_free(l.row);

   if(l.ridx)
      spx_free(l.ridx);

   if(l.rbeg)
      spx_free(l.rbeg);

   if(l.rorig)
      spx_free(l.rorig);

   if(l.rperm)
      spx_free(l.rperm);

   spx_free(solveTime);
   spx_free(factorTime);
}

inline SLUFactorRational::~SLUFactorRational()
{
   freeAll();
}

static Rational betterThreshold(Rational th)
{
   assert(th < 1);

   if(10 * th < 1)
      th *= 10;
   else if(10 * th < 8)
      th = (th + 1) / 2;
   else if(th < 0.999)
      th = 0.99999;

   assert(th < 1);

   return th;
}

inline SLUFactorRational::Status SLUFactorRational::load(const SVectorRational* matrix[], int dm)
{
   assert(dm     >= 0);
   assert(matrix != 0);

   Rational lastStability = stability();

   initDR(u.row.list);
   initDR(u.col.list);

   usetup        = false;
   l.updateType  = uptype;
   l.firstUpdate = 0;
   l.firstUnused = 0;

   if(dm != thedim)
   {
      clear();

      thedim = dm;
      vec.reDim(thedim);
      ssvec.reDim(thedim);
      eta.reDim(thedim);
      forest.reDim(thedim);
      work = vec.get_ptr();

      spx_realloc(row.perm, thedim);
      spx_realloc(row.orig, thedim);
      spx_realloc(col.perm, thedim);
      spx_realloc(col.orig, thedim);
      diag.reDim(thedim);

      spx_realloc(u.row.elem,  thedim);
      spx_realloc(u.row.len,   thedim + 1);
      spx_realloc(u.row.max,   thedim + 1);
      spx_realloc(u.row.start, thedim + 1);

      spx_realloc(u.col.elem,  thedim);
      spx_realloc(u.col.len,   thedim + 1);
      spx_realloc(u.col.max,   thedim + 1);
      spx_realloc(u.col.start, thedim + 1);

      l.startSize = thedim + MAXUPDATES;

      spx_realloc(l.row,   l.startSize);
      spx_realloc(l.start, l.startSize);
   }
   // the last factorization was reasonably stable, so we decrease the Markowitz threshold (stored in lastThreshold) in
   // order favour sparsity
   else if(lastStability > 2.0 * MINSTABILITYRAT)
   {
      // we reset lastThreshold to its previous value in the sequence minThreshold, betterThreshold(minThreshold),
      // betterThreshold(betterThreshold(minThreshold)), ...
      Rational last   = minThreshold;
      Rational better = betterThreshold(last);

      while(better < lastThreshold)
      {
         last   = better;
         better = betterThreshold(last);
      }

      lastThreshold = last;

      // we reset the minimum stability (which might have been decreased below) to ensure that the increased sparsity
      // does not hurt the stability
      minStability  = 2 * MINSTABILITYRAT;
   }

   u.row.list.idx      = thedim;
   u.row.start[thedim] = 0;
   u.row.max[thedim]   = 0;
   u.row.len[thedim]   = 0;

   u.col.list.idx      = thedim;
   u.col.start[thedim] = 0;
   u.col.max[thedim]   = 0;
   u.col.len[thedim]   = 0;

   stat = OK;
   factor(matrix, lastThreshold);

   MSG_DEBUG(std::cout << "DSLUFA02 threshold = " << lastThreshold
             << "\tstability = " << stability()
             << "\tMINSTABILITYRAT = " << MINSTABILITYRAT << std::endl;)
   MSG_DEBUG(
      int i;
      FILE* fl = fopen("dump.lp", "w");
      std::cout << "DSLUFA03 Basis:\n";
      int j = 0;

      for(i = 0; i < dim(); ++i)
      j += matrix[i]->size();
      for(i = 0; i < dim(); ++i)
   {
      for(j = 0; j < matrix[i]->size(); ++j)
            fprintf(fl, "%8d  %8d  %s\n",
                    i + 1, matrix[i]->index(j) + 1, matrix[i]->value(j).str());
      }
   fclose(fl);
   std::cout << "DSLUFA04 LU-Factors:" << std::endl;
             dump();

             std::cout << "DSLUFA05 threshold = " << lastThreshold
             << "\tstability = " << stability() << std::endl;
   )

   assert(isConsistent());
   return Status(stat);
}


inline bool SLUFactorRational::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS
   return CLUFactorRational::isConsistent();
#else
   return true;
#endif
}

inline void SLUFactorRational::dump() const
{
   CLUFactorRational::dump();
}
} // namespace soplex
