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

namespace soplex
{
/**@todo suspicious: *max is not set, but it is used
 * (with the default setting *max=1) in selectLeave and selectEnter
 * The question might be if max shouldn't be updated with themax?
 *
 * numCycle and maxCycle are integers. So degeneps will be
 * exactly delta until numCycle >= maxCycle. Then it will be
 * 0 until numCycle >= 2 * maxCycle, after wich it becomes
 * negative. This does not look ok.
 */
template <class R>
R SPxHarrisRT<R>::degenerateEps() const
{
   return this->solver()->delta()
          * (1.0 - this->solver()->numCycle() / this->solver()->maxCycle());
}

template <class R>
int SPxHarrisRT<R>::maxDelta(
   R* /*max*/,             /* max abs value in upd */
   R* val,             /* initial and chosen value */
   int num,             /* # of indices in idx */
   const int* idx,             /* nonzero indices in upd */
   const R* upd,             /* update VectorBase<R> for vec */
   const R* vec,             /* current VectorBase<R> */
   const R* low,             /* lower bounds for vec */
   const R* up,              /* upper bounds for vec */
   R epsilon)  const       /* what is 0? */
{
   R x;
   R theval;
   /**@todo patch suggests using *max instead of themax */
   R themax;
   int sel;
   int i;

   assert(*val >= 0);

   theval = *val;
   themax = 0;
   sel = -1;

   while(num--)
   {
      i = idx[num];
      x = upd[i];

      if(x > epsilon)
      {
         themax = (x > themax) ? x : themax;
         x = (up[i] - vec[i] + this->delta) / x;

         if(x < theval && up[i] < R(infinity))
            theval = x;
      }
      else if(x < -epsilon)
      {
         themax = (-x > themax) ? -x : themax;
         x = (low[i] - vec[i] - this->delta) / x;

         if(x < theval && low[i] > R(-infinity))
            theval = x;
      }
   }

   *val = theval;
   return sel;
}

/**@todo suspicious: *max is not set, but it is used
   (with the default setting *max=1)
   in selectLeave and selectEnter
*/
template <class R>
int SPxHarrisRT<R>::minDelta(
   R* /*max*/,             /* max abs value in upd */
   R* val,             /* initial and chosen value */
   int num,             /* # of indices in idx */
   const int* idx,             /* nonzero indices in upd */
   const R* upd,             /* update VectorBase<R> for vec */
   const R* vec,             /* current VectorBase<R> */
   const R* low,             /* lower bounds for vec */
   const R* up,              /* upper bounds for vec */
   R epsilon) const         /* what is 0? */
{
   R x;
   R theval;
   /**@todo patch suggests using *max instead of themax */
   R themax;
   int sel;
   int i;

   assert(*val < 0);

   theval = *val;
   themax = 0;
   sel = -1;

   while(num--)
   {
      i = idx[num];
      x = upd[i];

      if(x > epsilon)
      {
         themax = (x > themax) ? x : themax;
         x = (low[i] - vec[i] - this->delta) / x;

         if(x > theval && low[i] > R(-infinity))
            theval = x;
      }
      else if(x < -epsilon)
      {
         themax = (-x > themax) ? -x : themax;
         x = (up[i] - vec[i] + this->delta) / x;

         if(x > theval && up[i] < R(infinity))
            theval = x;
      }
   }

   *val = theval;
   return sel;
}

/**
   Here comes our implementation of the Harris procedure improved by shifting
   bounds. The basic idea is to used the tolerated infeasibility within
   solver()->entertol() for searching numerically stable pivots.

   The algorithms operates in two phases. In a first phase, the maximum \p val
   is determined, when infeasibility within solver()->entertol() is allowed. In the second
   phase, between all variables with values < \p val the one is selected which
   gives the best step forward in the simplex iteration. However, this may not
   allways yield an improvement. In that case, we shift the variable toward
   infeasibility and retry. This avoids cycling in the shifted LP.
*/
template <class R>
int SPxHarrisRT<R>::selectLeave(R& val, R, bool)
{
   int i, j;
   R stab, x, y;
   R max;
   R sel;
   R lastshift;
   R useeps;
   int leave = -1;
   R maxabs = 1;

   R epsilon  = this->solver()->epsilon();
   R degeneps = degenerateEps();

   SSVectorBase<R>& upd = this->solver()->fVec().delta();
   VectorBase<R>& vec = this->solver()->fVec();

   const VectorBase<R>& up = this->solver()->ubBound();
   const VectorBase<R>& low = this->solver()->lbBound();

   assert(this->delta > epsilon);
   assert(epsilon > 0);
   assert(this->solver()->maxCycle() > 0);

   max = val;
   lastshift = this->solver()->shift();

   this->solver()->fVec().delta().setup();

   if(max > epsilon)
   {
      // phase 1:
      maxDelta(
         &maxabs,             /* max abs value in upd */
         &max,                /* initial and chosen value */
         upd.size(),          /* # of indices in upd */
         upd.indexMem(),      /* nonzero indices in upd */
         upd.values(),        /* update VectorBase<R> for vec */
         vec.get_const_ptr(),         /* current VectorBase<R> */
         low.get_const_ptr(),                 /* lower bounds for vec */
         up.get_const_ptr(),                  /* upper bounds for vec */
         epsilon);             /* what is 0? */

      if(max == val)
         return -1;


      // phase 2:
      stab = 0;
      sel = R(-infinity);
      useeps = maxabs * epsilon * 0.001;

      if(useeps < epsilon)
         useeps = epsilon;

      for(j = upd.size() - 1; j >= 0; --j)
      {
         i = upd.index(j);
         x = upd[i];

         if(x > useeps)
         {
            y = up[i] - vec[i];

            if(y < -degeneps)
               this->solver()->shiftUBbound(i, vec[i]); // ensure simplex improvement
            else
            {
               y /= x;

               if(y <= max && y > sel - epsilon && x > stab)
               {
                  sel = y;
                  leave = i;
                  stab = x;
               }
            }
         }
         else if(x < -useeps)
         {
            y = low[i] - vec[i];

            if(y > degeneps)
               this->solver()->shiftLBbound(i, vec[i]); // ensure simplex improvement
            else
            {
               y /= x;

               if(y <= max && y > sel - epsilon && -x > stab)
               {
                  sel = y;
                  leave = i;
                  stab = -x;
               }
            }
         }
         else
            upd.clearNum(j);
      }
   }


   else if(max < -epsilon)
   {
      // phase 1:
      minDelta(
         &maxabs,             /* max abs value in upd */
         &max,                /* initial and chosen value */
         upd.size(),          /* # of indices in upd */
         upd.indexMem(),      /* nonzero indices in upd */
         upd.values(),        /* update VectorBase<R> for vec */
         vec.get_const_ptr(),                 /* current VectorBase<R> */
         low.get_const_ptr(),                 /* lower bounds for vec */
         up.get_const_ptr(),                  /* upper bounds for vec */
         epsilon);             /* what is 0? */

      if(max == val)
         return -1;

      // phase 2:
      stab = 0;
      sel = R(infinity);
      useeps = maxabs * epsilon * 0.001;

      if(useeps < epsilon)
         useeps = epsilon;

      for(j = upd.size() - 1; j >= 0; --j)
      {
         i = upd.index(j);
         x = upd[i];

         if(x < -useeps)
         {
            y = up[i] - vec[i];

            if(y < -degeneps)
               this->solver()->shiftUBbound(i, vec[i]);   // ensure simplex improvement
            else
            {
               y /= x;

               if(y >= max && y < sel + epsilon && -x > stab)
               {
                  sel = y;
                  leave = i;
                  stab = -x;
               }
            }
         }
         else if(x > useeps)
         {
            y = low[i] - vec[i];

            if(y > degeneps)
               this->solver()->shiftLBbound(i, vec[i]); // ensure simplex improvement
            else
            {
               y /= x;

               if(y >= max && y < sel + epsilon && x > stab)
               {
                  sel = y;
                  leave = i;
                  stab = x;
               }
            }
         }
         else
            upd.clearNum(j);
      }
   }

   else
      return -1;


   if(lastshift != this->solver()->shift())
      return selectLeave(val, 0, false);

   assert(leave >= 0);

   val = sel;
   return leave;
}

template <class R>
SPxId SPxHarrisRT<R>::selectEnter(R& val, int, bool)
{
   int i, j;
   SPxId enterId;
   R stab, x, y;
   R max = 0.0;
   R sel = 0.0;
   R lastshift;
   R cuseeps;
   R ruseeps;
   R cmaxabs = 1;
   R rmaxabs = 1;
   int pnr, cnr;

   R minStability = 0.0001;
   R epsilon      = this->solver()->epsilon();
   R degeneps     = degenerateEps();

   VectorBase<R>& pvec = this->solver()->pVec();
   SSVectorBase<R>& pupd = this->solver()->pVec().delta();

   VectorBase<R>& cvec = this->solver()->coPvec();
   SSVectorBase<R>& cupd = this->solver()->coPvec().delta();

   const VectorBase<R>& upb = this->solver()->upBound();
   const VectorBase<R>& lpb = this->solver()->lpBound();
   const VectorBase<R>& ucb = this->solver()->ucBound();
   const VectorBase<R>& lcb = this->solver()->lcBound();

   assert(this->delta > epsilon);
   assert(epsilon > 0);
   assert(this->solver()->maxCycle() > 0);

   this->solver()->coPvec().delta().setup();
   this->solver()->pVec().delta().setup();

   if(val > epsilon)
   {
      for(;;)
      {
         pnr = -1;
         cnr = -1;
         max = val;
         lastshift = this->solver()->shift();
         assert(this->delta > epsilon);

         // phase 1:
         maxDelta(
            &rmaxabs,            /* max abs value in upd */
            &max,                /* initial and chosen value */
            pupd.size(),         /* # of indices in pupd */
            pupd.indexMem(),     /* nonzero indices in pupd */
            pupd.values(),       /* update VectorBase<R> for vec */
            pvec.get_const_ptr(),                /* current VectorBase<R> */
            lpb.get_const_ptr(),                 /* lower bounds for vec */
            upb.get_const_ptr(),                 /* upper bounds for vec */
            epsilon);             /* what is 0? */

         maxDelta(
            &cmaxabs,            /* max abs value in upd */
            &max,                /* initial and chosen value */
            cupd.size(),         /* # of indices in cupd */
            cupd.indexMem(),     /* nonzero indices in cupd */
            cupd.values(),       /* update VectorBase<R> for vec */
            cvec.get_const_ptr(),                /* current VectorBase<R> */
            lcb.get_const_ptr(),                 /* lower bounds for vec */
            ucb.get_const_ptr(),                 /* upper bounds for vec */
            epsilon);            /* what is 0? */

         if(max == val)
            return enterId;


         // phase 2:
         stab = 0;
         sel = R(-infinity);
         ruseeps = rmaxabs * 0.001 * epsilon;

         if(ruseeps < epsilon)
            ruseeps = epsilon;

         cuseeps = cmaxabs * 0.001 * epsilon;

         if(cuseeps < epsilon)
            cuseeps = epsilon;

         for(j = pupd.size() - 1; j >= 0; --j)
         {
            i = pupd.index(j);
            x = pupd[i];

            if(x > ruseeps)
            {
               y = upb[i] - pvec[i];

               if(y < -degeneps)
                  this->solver()->shiftUPbound(i, pvec[i] - degeneps);
               else
               {
                  y /= x;

                  if(y <= max && x >= stab)        // &&  y > sel-epsilon
                  {
                     enterId = this->solver()->id(i);
                     sel = y;
                     pnr = i;
                     stab = x;
                  }
               }
            }
            else if(x < -ruseeps)
            {
               y = lpb[i] - pvec[i];

               if(y > degeneps)
                  this->solver()->shiftLPbound(i, pvec[i] + degeneps);
               else
               {
                  y /= x;

                  if(y <= max && -x >= stab)       // &&  y > sel-epsilon
                  {
                     enterId = this->solver()->id(i);
                     sel = y;
                     pnr = i;
                     stab = -x;
                  }
               }
            }
            else
            {
               MSG_DEBUG(std::cout << "DHARRI01 removing value " << pupd[i] << std::endl;)
               pupd.clearNum(j);
            }
         }

         for(j = cupd.size() - 1; j >= 0; --j)
         {
            i = cupd.index(j);
            x = cupd[i];

            if(x > cuseeps)
            {
               y = ucb[i] - cvec[i];

               if(y < -degeneps)
                  this->solver()->shiftUCbound(i, cvec[i] - degeneps);
               else
               {
                  y /= x;

                  if(y <= max && x >= stab)        // &&  y > sel-epsilon
                  {
                     enterId = this->solver()->coId(i);
                     sel = y;
                     cnr = j;
                     stab = x;
                  }
               }
            }
            else if(x < -cuseeps)
            {
               y = lcb[i] - cvec[i];

               if(y > degeneps)
                  this->solver()->shiftLCbound(i, cvec[i] + degeneps);
               else
               {
                  y /= x;

                  if(y <= max && -x >= stab)       // &&  y > sel-epsilon
                  {
                     enterId = this->solver()->coId(i);
                     sel = y;
                     cnr = j;
                     stab = -x;
                  }
               }
            }
            else
            {
               MSG_DEBUG(std::cout << "DHARRI02 removing value " << cupd[i] << std::endl;)
               cupd.clearNum(j);
            }
         }

         if(lastshift == this->solver()->shift())
         {
            if(cnr >= 0)
            {
               if(this->solver()->isBasic(enterId))
               {
                  cupd.clearNum(cnr);
                  continue;
               }
               else
                  break;
            }
            else if(pnr >= 0)
            {
               pvec[pnr] = this->solver()->vector(pnr) * cvec;

               if(this->solver()->isBasic(enterId))
               {
                  pupd.setValue(pnr, 0.0);
                  continue;
               }
               else
               {
                  x = pupd[pnr];

                  if(x > 0)
                  {
                     sel = upb[pnr] - pvec[pnr];

                     if(x < minStability && sel < this->delta)
                     {
                        minStability /= 2.0;
                        this->solver()->shiftUPbound(pnr, pvec[pnr]);
                        continue;
                     }
                  }
                  else
                  {
                     sel = lpb[pnr] - pvec[pnr];

                     if(-x < minStability && -sel < this->delta)
                     {
                        minStability /= 2.0;
                        this->solver()->shiftLPbound(pnr, pvec[pnr]);
                        continue;
                     }
                  }

                  sel /= x;
               }
            }
            else
            {
               val = 0;
               enterId.inValidate();
               return enterId;
            }

            if(sel > max)              // instability detected => recompute
               continue;               // ratio test with corrected value

            break;
         }
      }
   }
   else if(val < -epsilon)
   {
      for(;;)
      {
         pnr = -1;
         cnr = -1;
         max = val;
         lastshift = this->solver()->shift();
         assert(this->delta > epsilon);


         // phase 1:
         minDelta
         (
            &rmaxabs,            /* max abs value in upd */
            &max,                /* initial and chosen value */
            pupd.size(),         /* # of indices in pupd */
            pupd.indexMem(),     /* nonzero indices in pupd */
            pupd.values(),       /* update VectorBase<R> for vec */
            pvec.get_const_ptr(),                /* current VectorBase<R> */
            lpb.get_const_ptr(),                 /* lower bounds for vec */
            upb.get_const_ptr(),                 /* upper bounds for vec */
            epsilon);             /* what is 0? */

         minDelta
         (
            &cmaxabs,            /* max abs value in upd */
            &max,                /* initial and chosen value */
            cupd.size(),         /* # of indices in cupd */
            cupd.indexMem(),     /* nonzero indices in cupd */
            cupd.values(),       /* update VectorBase<R> for vec */
            cvec.get_const_ptr(),                /* current VectorBase<R> */
            lcb.get_const_ptr(),                 /* lower bounds for vec */
            ucb.get_const_ptr(),                 /* upper bounds for vec */
            epsilon);             /* what is 0? */

         if(max == val)
            return enterId;


         // phase 2:
         stab = 0;
         sel = R(infinity);
         ruseeps = rmaxabs * epsilon * 0.001;
         cuseeps = cmaxabs * epsilon * 0.001;

         for(j = pupd.size() - 1; j >= 0; --j)
         {
            i = pupd.index(j);
            x = pupd[i];

            if(x > ruseeps)
            {
               y = lpb[i] - pvec[i];

               if(y > degeneps)
                  this->solver()->shiftLPbound(i, pvec[i]);  // ensure simplex improvement
               else
               {
                  y /= x;

                  if(y >= max && x > stab)         // &&  y < sel+epsilon
                  {
                     enterId = this->solver()->id(i);
                     sel = y;
                     pnr = i;
                     stab = x;
                  }
               }
            }
            else if(x < -ruseeps)
            {
               y = upb[i] - pvec[i];

               if(y < -degeneps)
                  this->solver()->shiftUPbound(i, pvec[i]);  // ensure simplex improvement
               else
               {
                  y /= x;

                  if(y >= max && -x > stab)        // &&  y < sel+epsilon
                  {
                     enterId = this->solver()->id(i);
                     sel = y;
                     pnr = i;
                     stab = -x;
                  }
               }
            }
            else
            {
               MSG_DEBUG(std::cout << "DHARRI03 removing value " << pupd[i] << std::endl;)
               pupd.clearNum(j);
            }
         }

         for(j = cupd.size() - 1; j >= 0; --j)
         {
            i = cupd.index(j);
            x = cupd[i];

            if(x > cuseeps)
            {
               y = lcb[i] - cvec[i];

               if(y > degeneps)
                  this->solver()->shiftLCbound(i, cvec[i]);  // ensure simplex improvement
               else
               {
                  y /= x;

                  if(y >= max && x > stab)         // &&  y < sel+epsilon
                  {
                     enterId = this->solver()->coId(i);
                     sel = y;
                     cnr = j;
                     stab = x;
                  }
               }
            }
            else if(x < -cuseeps)
            {
               y = ucb[i] - cvec[i];

               if(y < -degeneps)
                  this->solver()->shiftUCbound(i, cvec[i]);  // ensure simplex improvement
               else
               {
                  y /= x;

                  if(y >= max && -x > stab)        // &&  y < sel+epsilon
                  {
                     enterId = this->solver()->coId(i);
                     sel = y;
                     cnr = j;
                     stab = -x;
                  }
               }
            }
            else
            {
               MSG_DEBUG(std::cout << "DHARRI04 removing value " << x << std::endl;);
               cupd.clearNum(j);
            }
         }

         if(lastshift == this->solver()->shift())
         {
            if(cnr >= 0)
            {
               if(this->solver()->isBasic(enterId))
               {
                  cupd.clearNum(cnr);
                  continue;
               }
               else
                  break;
            }
            else if(pnr >= 0)
            {
               pvec[pnr] = this->solver()->vector(pnr) * cvec;

               if(this->solver()->isBasic(enterId))
               {
                  pupd.setValue(pnr, 0.0);
                  continue;
               }
               else
               {
                  x = pupd[pnr];

                  if(x > 0)
                  {
                     sel = lpb[pnr] - pvec[pnr];

                     if(x < minStability && -sel < this->delta)
                     {
                        minStability /= 2;
                        this->solver()->shiftLPbound(pnr, pvec[pnr]);
                        continue;
                     }
                  }
                  else
                  {
                     sel = upb[pnr] - pvec[pnr];

                     if(-x < minStability && sel < this->delta)
                     {
                        minStability /= 2;
                        this->solver()->shiftUPbound(pnr, pvec[pnr]);
                        continue;
                     }
                  }

                  sel /= x;
               }
            }
            else
            {
               val = 0;
               enterId.inValidate();
               return enterId;
            }

            if(sel < max)              // instability detected => recompute
               continue;               // ratio test with corrected value

            break;
         }
      }
   }

   assert(max * val >= 0);
   assert(enterId.type() != SPxId::INVALID);

   val = sel;

   return enterId;
}
} // namespace soplex
