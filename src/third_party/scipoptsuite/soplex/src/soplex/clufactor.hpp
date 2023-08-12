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

#include <assert.h>
#include "soplex/cring.h"

namespace soplex
{

/* Macro to print a warning message for huge values */
#ifndef NDEBUG
#define DEBUG_CHECK_HUGE_VALUE( prefix, value )                                                    \
   if(spxAbs(value) >= 1e40 && this->hugeValues < 3)                                               \
   {                                                                                               \
      this->hugeValues++;                                                                          \
      std::cout << prefix                                                                          \
                << " Huge value during triangular solve: "                                         \
                << value << std::endl;                                                             \
      if(this->hugeValues >= 3)                                                                    \
         std::cout << "Skipping further warnings of this type during current solve." << std::endl; \
   }
#else
#define DEBUG_CHECK_HUGE_VALUE( prefix, value ) /**/
#endif

/* This number is used to decide wether a value is zero
 * or was explicitly set to zero.
 */
#define SOPLEX_FACTOR_MARKER     1e-100

static const Real verySparseFactor = 0.001;
static const Real verySparseFactor4right = 0.2;
static const Real verySparseFactor4left  = 0.1;

/* generic heap management */
static inline void enQueueMax(int* heap, int* size, int elem)
{
   int i, j;

   j = (*size)++;

   while(j > 0)
   {
      i = (j - 1) / 2;

      if(elem > heap[i])
      {
         heap[j] = heap[i];
         j = i;
      }
      else
         break;
   }

   heap[j] = elem;

#ifdef SOPLEX_DEBUG

   // no NDEBUG define, since this block is really expensive
   for(i = 1; i < *size; ++i)
      for(j = 0; j < i; ++j)
         assert(heap[i] != heap[j]);

#endif  /* SOPLEX_DEBUG */
}

static inline int deQueueMax(int* heap, int* size)
{
   int e, elem;
   int i, j, s;
   int e1, e2;

   elem = *heap;
   e = heap[s = --(*size)];
   --s;

   for(j = 0, i = 1; i < s; i = 2 * j + 1)
   {
      e1 = heap[i];
      e2 = heap[i + 1];

      if(e1 > e2)
      {
         if(e < e1)
         {
            heap[j] = e1;
            j = i;
         }
         else
         {
            heap[j] = e;
            return elem;
         }
      }
      else
      {
         if(e < e2)
         {
            heap[j] = e2;
            j = i + 1;
         }
         else
         {
            heap[j] = e;
            return elem;
         }
      }
   }

   if(i < *size && e < heap[i])
   {
      heap[j] = heap[i];
      j = i;
   }

   heap[j] = e;

   return elem;
}

static inline void enQueueMin(int* heap, int* size, int elem)
{
   int i, j;

   j = (*size)++;

   while(j > 0)
   {
      i = (j - 1) / 2;

      if(elem < heap[i])
      {
         heap[j] = heap[i];
         j = i;
      }
      else
         break;
   }

   heap[j] = elem;

#ifdef SOPLEX_DEBUG

   // no NDEBUG define, since this block is really expensive
   for(i = 1; i < *size; ++i)
      for(j = 0; j < i; ++j)
         assert(heap[i] != heap[j]);

#endif  /* SOPLEX_DEBUG */
}

static inline int deQueueMin(int* heap, int* size)
{
   int e, elem;
   int i, j, s;
   int e1, e2;

   elem = *heap;
   e = heap[s = --(*size)];
   --s;

   for(j = 0, i = 1; i < s; i = 2 * j + 1)
   {
      e1 = heap[i];
      e2 = heap[i + 1];

      if(e1 < e2)
      {
         if(e > e1)
         {
            heap[j] = e1;
            j = i;
         }
         else
         {
            heap[j] = e;
            return elem;
         }
      }
      else
      {
         if(e > e2)
         {
            heap[j] = e2;
            j = i + 1;
         }
         else
         {
            heap[j] = e;
            return elem;
         }
      }
   }

   if(i < *size && e > heap[i])
   {
      heap[j] = heap[i];
      j = i;
   }

   heap[j] = e;

   return elem;
}

/************************************************************/
template <class R>
CLUFactor<R>::Temp::Temp()
   : s_mark(0)
   , s_cact(0)
   , stage(0)
   , pivot_col(0)
   , pivot_colNZ(0)
   , pivot_row(0)
   , pivot_rowNZ(0)
{}

template <class R>
void CLUFactor<R>::Temp::init(int p_dim)
{
   s_max.resize(p_dim);
   spx_realloc(s_cact, p_dim);
   spx_realloc(s_mark, p_dim);
   stage = 0;
}

template <class R>
void CLUFactor<R>::Temp::clear()
{
   if(s_mark != nullptr)
      spx_free(s_mark);

   if(s_cact != nullptr)
      spx_free(s_cact);

   if(!s_max.empty())
      s_max.clear();

   if(pivot_col != nullptr)
      spx_free(pivot_col);

   if(pivot_colNZ != nullptr)
      spx_free(pivot_colNZ);

   if(pivot_row != nullptr)
      spx_free(pivot_row);

   if(pivot_rowNZ != nullptr)
      spx_free(pivot_rowNZ);
}

template <class R>
CLUFactor<R>::Temp::~Temp()
{
   clear();
}

/************************************************************/
template <class R>
void CLUFactor<R>::initPerm()
{

   for(int i = 0; i < thedim; ++i)
      row.orig[i] = row.perm[i] = col.orig[i] = col.perm[i] = -1;
}

/*****************************************************************************/

template <class R>
void CLUFactor<R>::setPivot(const int p_stage,
                            const int p_col,
                            const int p_row,
                            const R val)
{
   assert(row.perm[p_row] < 0);
   assert(col.perm[p_col] < 0);

   row.orig[p_stage] = p_row;
   col.orig[p_stage] = p_col;
   row.perm[p_row]   = p_stage;
   col.perm[p_col]   = p_stage;
   diag[p_row]       = R(1.0) / val;

   if(spxAbs(val) < Param::epsilonPivot())
   {
#ifndef NDEBUG
      MSG_ERROR(std::cerr
                << "LU pivot element is almost zero (< "
                << Param::epsilonPivot()
                << ") - Basis is numerically singular"
                << std::endl;
               )
#endif
      this->stat = SLinSolver<R>::SINGULAR;
   }

   if(spxAbs(diag[p_row]) > maxabs)
      maxabs = spxAbs(diag[p_row]);
}

/*****************************************************************************/
/*
 *      Perform garbage collection on row file
 */
template <class R>
void CLUFactor<R>::packRows()
{
   int n, i, j, l_row;
   Dring* ring, *list;

   int* l_ridx = u.row.idx;
   R* l_rval = u.row.val.data();
   int* l_rlen = u.row.len;
   int* l_rmax = u.row.max;
   int* l_rbeg = u.row.start;

   n = 0;
   list = &(u.row.list);

   for(ring = list->next; ring != list; ring = ring->next)
   {
      l_row = ring->idx;

      if(l_rbeg[l_row] != n)
      {
         do
         {
            l_row = ring->idx;
            i = l_rbeg[l_row];
            assert(l_rlen[l_row] <= l_rmax[l_row]);
            l_rbeg[l_row] = n;
            l_rmax[l_row] = l_rlen[l_row];
            j = i + l_rlen[l_row];

            for(; i < j; ++i, ++n)
            {
               assert(n <= i);
               l_ridx[n] = l_ridx[i];
               l_rval[n] = l_rval[i];
            }

            ring = ring->next;
         }
         while(ring != list);

         goto terminatePackRows;
      }

      n += l_rlen[l_row];

      l_rmax[l_row] = l_rlen[l_row];
   }

terminatePackRows:

   u.row.max[thedim] = 0;
   u.row.used = n;
}

/*****************************************************************************/
/*
 *      Perform garbage collection on column file
 */
template <class R>
void CLUFactor<R>::forestPackColumns()
{
   int n, i, j, colno;
   Dring* ring, *list;

   R* cval = u.col.val.data();
   int* cidx = u.col.idx;
   int* clen = u.col.len;
   int* cmax = u.col.max;
   int* cbeg = u.col.start;

   n = 0;
   list = &u.col.list;

   for(ring = list->next; ring != list; ring = ring->next)
   {
      colno = ring->idx;

      if(cbeg[colno] != n)
      {
         do
         {
            colno = ring->idx;
            i = cbeg[colno];
            cbeg[colno] = n;
            cmax[colno] = clen[colno];
            j = i + clen[colno];

            for(; i < j; ++i)
            {
               cval[n] = cval[i];
               cidx[n++] = cidx[i];
            }

            ring = ring->next;
         }
         while(ring != list);

         goto terminatePackColumns;
      }

      n += clen[colno];

      cmax[colno] = clen[colno];
   }

terminatePackColumns :

   u.col.used = n;
   u.col.max[thedim] = 0;
}

/*
 *      Make row of fac large enough to hold len nonzeros.
 */
template <class R>
void CLUFactor<R>::remaxRow(int p_row, int len)
{
   assert(u.row.max[p_row] < len);

   if(u.row.elem[p_row].next == &(u.row.list))      /* last in row file */
   {
      int delta = len - u.row.max[p_row];

      if(delta > u.row.size - u.row.used)
      {
         packRows();
         delta = len - u.row.max[p_row];  // packRows() changes u.row.max[] !

         if(u.row.size < rowMemMult * u.row.used + len)
            minRowMem(2 * u.row.used + len);

         /* minRowMem(rowMemMult * u.row.used + len); */
      }

      assert(delta <= u.row.size - u.row.used

             && "ERROR: could not allocate memory for row file");

      u.row.used += delta;
      u.row.max[p_row] = len;
   }
   else                        /* row must be moved to end of row file */
   {
      int i, j, k;
      int* idx;
      R* val;
      Dring* ring;

      if(len > u.row.size - u.row.used)
      {
         packRows();

         if(u.row.size < rowMemMult * u.row.used + len)
            minRowMem(2 * u.row.used + len);

         /* minRowMem(rowMemMult * u.row.used + len);*/
      }

      assert(len <= u.row.size - u.row.used

             && "ERROR: could not allocate memory for row file");

      j = u.row.used;
      i = u.row.start[p_row];
      k = u.row.len[p_row] + i;
      u.row.start[p_row] = j;
      u.row.used += len;

      u.row.max[u.row.elem[p_row].prev->idx] += u.row.max[p_row];
      u.row.max[p_row] = len;
      removeDR(u.row.elem[p_row]);
      ring = u.row.list.prev;
      init2DR(u.row.elem[p_row], *ring);

      idx = u.row.idx;
      val = u.row.val.data();

      for(; i < k; ++i, ++j)
      {
         val[j] = val[i];
         idx[j] = idx[i];
      }
   }

   assert(u.row.start[u.row.list.prev->idx] + u.row.max[u.row.list.prev->idx]

          == u.row.used);
}

/*************************************************************************/
/*
 *      Perform garbage collection on column file
 */
template <class R>
void CLUFactor<R>::packColumns()
{
   int n, i, j, l_col;
   Dring* ring, *list;

   int* l_cidx = u.col.idx;
   int* l_clen = u.col.len;
   int* l_cmax = u.col.max;
   int* l_cbeg = u.col.start;

   n = 0;
   list = &(u.col.list);

   for(ring = list->next; ring != list; ring = ring->next)
   {
      l_col = ring->idx;

      if(l_cbeg[l_col] != n)
      {
         do
         {
            l_col = ring->idx;
            i = l_cbeg[l_col];
            l_cbeg[l_col] = n;
            l_cmax[l_col] = l_clen[l_col];
            j = i + l_clen[l_col];

            for(; i < j; ++i)
               l_cidx[n++] = l_cidx[i];

            ring = ring->next;
         }
         while(ring != list);

         goto terminatePackColumns;
      }

      n += l_clen[l_col];

      l_cmax[l_col] = l_clen[l_col];
   }

terminatePackColumns :

   u.col.used = n;
   u.col.max[thedim] = 0;
}

/*
 *      Make column col of fac large enough to hold len nonzeros.
 */
template <class R>
void CLUFactor<R>::remaxCol(int p_col, int len)
{
   assert(u.col.max[p_col] < len);

   if(u.col.elem[p_col].next == &(u.col.list))      /* last in column file */
   {
      int delta = len - u.col.max[p_col];

      if(delta > u.col.size - u.col.used)
      {
         packColumns();
         delta = len - u.col.max[p_col];

         if(u.col.size < colMemMult * u.col.used + len)
            minColMem(2 * u.col.used + len);

         /* minColMem(colMemMult * u.col.used + len); */
      }

      assert(delta <= u.col.size - u.col.used

             && "ERROR: could not allocate memory for column file");

      u.col.used += delta;
      u.col.max[p_col] = len;
   }

   else                        /* column must be moved to end of column file */
   {
      int i, j, k;
      int* idx;
      Dring* ring;

      if(len > u.col.size - u.col.used)
      {
         packColumns();

         if(u.col.size < colMemMult * u.col.used + len)
            minColMem(2 * u.col.used + len);

         /* minColMem(colMemMult * u.col.used + len); */
      }

      assert(len <= u.col.size - u.col.used

             && "ERROR: could not allocate memory for column file");

      j = u.col.used;
      i = u.col.start[p_col];
      k = u.col.len[p_col] + i;
      u.col.start[p_col] = j;
      u.col.used += len;

      u.col.max[u.col.elem[p_col].prev->idx] += u.col.max[p_col];
      u.col.max[p_col] = len;
      removeDR(u.col.elem[p_col]);
      ring = u.col.list.prev;
      init2DR(u.col.elem[p_col], *ring);

      idx = u.col.idx;

      for(; i < k; ++i)
         idx[j++] = idx[i];
   }
}

/*
 *      Make column col of fac large enough to hold len nonzeros.
 */
template <class R>
void CLUFactor<R>::forestReMaxCol(int p_col, int len)
{
   assert(u.col.max[p_col] < len);

   if(u.col.elem[p_col].next == &(u.col.list))      /* last in column file */
   {
      int delta = len - u.col.max[p_col];

      if(delta > u.col.size - u.col.used)
      {
         forestPackColumns();
         delta = len - u.col.max[p_col];

         if(u.col.size < colMemMult * u.col.used + len)
            forestMinColMem(int(colMemMult * u.col.used + len));
      }

      assert(delta <= u.col.size - u.col.used

             && "ERROR: could not allocate memory for column file");

      u.col.used += delta;
      u.col.max[p_col] = len;
   }

   else                        /* column must be moved to end of column file */
   {
      int i, j, k;
      int* idx;
      R* val;
      Dring* ring;

      if(len > u.col.size - u.col.used)
      {
         forestPackColumns();

         if(u.col.size < colMemMult * u.col.used + len)
            forestMinColMem(int(colMemMult * u.col.used + len));
      }

      assert(len <= u.col.size - u.col.used

             && "ERROR: could not allocate memory for column file");

      j = u.col.used;
      i = u.col.start[p_col];
      k = u.col.len[p_col] + i;
      u.col.start[p_col] = j;
      u.col.used += len;

      u.col.max[u.col.elem[p_col].prev->idx] += u.col.max[p_col];
      u.col.max[p_col] = len;
      removeDR(u.col.elem[p_col]);
      ring = u.col.list.prev;
      init2DR(u.col.elem[p_col], *ring);

      idx = u.col.idx;
      val = u.col.val.data();

      for(; i < k; ++i)
      {
         val[j] = val[i];
         idx[j++] = idx[i];
      }
   }
}

/*****************************************************************************/

/**
 *   \brief Performs the Forrest-Tomlin update of the LU factorization.
 *
 *   BH: I suppose this is implemented as described in UH Suhl, LM Suhl: A fast LU
 *       update for linear programming, Annals of OR 43, p. 33-47, 1993.
 *
 *   @param  p_col      Index of basis column to replace.
 *   @param  p_work     Dense VectorBase<R> to substitute in the basis.
 *   @param  num        Number of nonzeros in VectorBase<R> represented by p_work.
 *   @param  nonz       Indices of nonzero elements in VectorBase<R> p_work.
 *
 *   The parameters num and nonz are used for the following optimization: If both
 *   are nonzero, indices of the nonzero elements provided in nonz (num giving
 *   their number) allow to access only those nonzero entries.  Otherwise we have
 *   to go through the entire dense VectorBase<R> element by element.
 *
 *   After copying p_work into U, p_work is used to expand the row r, which is
 *   needed to restore the triangular structure of U.
 *
 *   Also num and nonz are used to maintain a heap if there are only very few
 *   nonzeros to be eliminated. This is plainly wrong if the method is called with
 *   nonz==0, see todo at the corresponding place below.
 *
 *   @throw SPxStatusException if the loaded matrix is singular
 *
 *   @todo Use an extra member variable as a buffer for working with the dense
 *         row instead of misusing p_work. I think that should be as efficient and
 *         much cleaner.
 */
template <class R>
void CLUFactor<R>::forestUpdate(int p_col, R* p_work, int num, int* nonz)
{
   int i, j, k, h, m, n;
   int ll, c, r, rowno;
   R x;

   R* lval;
   int* lidx;
   int* lbeg = l.start;

   R* cval;
   int* cidx = u.col.idx;
   int* cmax = u.col.max;
   int* clen = u.col.len;
   int* cbeg = u.col.start;

   R* rval = u.row.val.data();
   int* ridx = u.row.idx;
   int* rmax = u.row.max;
   int* rlen = u.row.len;
   int* rbeg = u.row.start;

   int* rperm = row.perm;
   int* rorig = row.orig;
   int* cperm = col.perm;
   int* corig = col.orig;

   R l_maxabs = maxabs;
   int dim = thedim;

   /*  Remove column p_col from U
    */
   j = cbeg[p_col];
   i = clen[p_col];
   nzCnt -= i;

   for(i += j - 1; i >= j; --i)
   {
      m = cidx[i];          // remove column p_col from row m
      k = rbeg[m];
      h = --(rlen[m]) + k;    // decrease length of row m

      while(ridx[k] != p_col)
         ++k;

      assert(k <= h);       // k is the position of p_col, h is last position

      ridx[k] = ridx[h];    // store last index at the position of p_col

      rval[k] = rval[h];
   }

   /*  Insert new VectorBase<R> column p_col thereby determining the highest permuted
    *       row index r.
    *
    *       Distinguish between optimized call (num > 0, nonz != 0) and
    *       non-optimized one.
    */
   assert(num);   // otherwise the assert( nonz != 0 ) below should fail

   if(num)
   {
      // Optimized call.
      assert(nonz != 0);

      clen[p_col] = 0;

      if(num > cmax[p_col])
         forestReMaxCol(p_col, num);

      cidx = u.col.idx;

      cval = u.col.val.data();

      k = cbeg[p_col];

      r = 0;

      for(j = 0; j < num; ++j)
      {
         i = nonz[j];
         x = p_work[i];
         p_work[i] = 0.0;

         if(isNotZero(x, R(Param::epsilonUpdate())))
         {
            if(spxAbs(x) > l_maxabs)
               l_maxabs = spxAbs(x);

            /* insert to column file */
            assert(k - cbeg[p_col] < cmax[p_col]);

            cval[k] = x;

            cidx[k++] = i;

            /* insert to row file */
            if(rmax[i] <= rlen[i])
            {
               remaxRow(i, rlen[i] + 1);
               rval = u.row.val.data();
               ridx = u.row.idx;
            }

            h = rbeg[i] + (rlen[i])++;

            rval[h] = x;
            ridx[h] = p_col;

            /* check permuted row index */

            if(rperm[i] > r)
               r = rperm[i];
         }
      }

      nzCnt += (clen[p_col] = k - cbeg[p_col]);
   }
   else
   {
      // Non-optimized call: We have to access all elements of p_work.
      assert(nonz == 0);

      /*
       *      clen[col] = 0;
       *      reMaxCol(fac, col, dim);
       */
      cidx = u.col.idx;
      cval = u.col.val.data();
      k = cbeg[p_col];
      j = k + cmax[p_col];
      r = 0;

      for(i = 0; i < dim; ++i)
      {
         x = p_work[i];
         p_work[i] = 0.0;

         if(isNotZero(x, R(Param::epsilonUpdate())))
         {
            if(spxAbs(x) > l_maxabs)
               l_maxabs = spxAbs(x);

            /* insert to column file */
            if(k >= j)
            {
               clen[p_col] = k - cbeg[p_col];
               forestReMaxCol(p_col, dim - i);
               cidx = u.col.idx;
               cval = u.col.val.data();
               k = cbeg[p_col];
               j = k + cmax[p_col];
               k += clen[p_col];
            }

            assert(k - cbeg[p_col] < cmax[p_col]);

            cval[k] = x;
            cidx[k++] = i;

            /* insert to row file */

            if(rmax[i] <= rlen[i])
            {
               remaxRow(i, rlen[i] + 1);
               rval = u.row.val.data();
               ridx = u.row.idx;
            }

            h = rbeg[i] + (rlen[i])++;

            rval[h] = x;
            ridx[h] = p_col;

            /* check permuted row index */

            if(rperm[i] > r)
               r = rperm[i];
         }
      }

      nzCnt += (clen[p_col] = k - cbeg[p_col]);

      if(cbeg[p_col] + cmax[p_col] == u.col.used)
      {
         u.col.used -= cmax[p_col];
         cmax[p_col] = clen[p_col];
         u.col.used += cmax[p_col];
      }
   }

   c = cperm[p_col];

   if(r > c)                          /* Forest Tomlin update */
   {
      /*      update permutations
       */
      j = rorig[c];

      // memmove is more efficient than a for loop
      // for ( i = c; i < r; ++i )
      //    rorig[i] = rorig[i + 1];
      memmove(&rorig[c], &rorig[c + 1], (unsigned int)(r - c) * sizeof(int));

      rorig[r] = j;

      for(i = c; i <= r; ++i)
         rperm[rorig[i]] = i;

      j = corig[c];

      // memmove is more efficient than a for loop
      // for ( i = c; i < r; ++i )
      //    corig[i] = corig[i + 1];
      memmove(&corig[c], &corig[c + 1], (unsigned int)(r - c) * sizeof(int));

      corig[r] = j;

      for(i = c; i <= r; ++i)
         cperm[corig[i]] = i;


      rowno = rorig[r];

      j = rbeg[rowno];

      i = rlen[rowno];

      nzCnt -= i;

      if(i < verySparseFactor * (dim - c))      // few nonzeros to be eliminated
      {
         /**
          *          The following assert is obviously violated if this method is called
          *          with nonzero==0.
          *
          *          @todo Use an extra member variable as a buffer for the heap instead of
          *                misusing nonz and num. The method enQueueMin() seems to
          *                sort the nonzeros or something, for which it only needs
          *                some empty VectorBase<R> of size num.
          */
         assert(nonz != 0);

         /*  move row r from U to p_work
          */
         num = 0;

         for(i += j - 1; i >= j; --i)
         {
            k = ridx[i];
            p_work[k] = rval[i];
            enQueueMin(nonz, &num, cperm[k]);
            m = --(clen[k]) + cbeg[k];

            for(h = m; cidx[h] != rowno; --h)
               ;

            cidx[h] = cidx[m];

            cval[h] = cval[m];
         }


         /*  Eliminate row r from U to L file
          */
         ll = makeLvec(r - c, rowno);

         lval = l.val.data();

         lidx = l.idx;

         assert((num == 0) || (nonz != 0));

         /* for(i = c; i < r; ++i)       */
         while(num)
         {
#ifndef NDEBUG
            // The numbers seem to be often 1e-100, is this ok ?

            for(i = 0; i < num; ++i)
               assert(p_work[corig[nonz[i]]] != 0.0);

#endif // NDEBUG
            i = deQueueMin(nonz, &num);

            if(i == r)
               break;

            k = corig[i];

            assert(p_work[k] != 0.0);

            n = rorig[i];

            x = p_work[k] * diag[n];

            lidx[ll] = n;

            lval[ll] = x;

            p_work[k] = 0.0;

            ll++;

            if(spxAbs(x) > l_maxabs)
               l_maxabs = spxAbs(x);

            j = rbeg[n];

            m = rlen[n] + j;

            for(; j < m; ++j)
            {
               int jj = ridx[j];
               R y = p_work[jj];

               if(y == 0)
                  enQueueMin(nonz, &num, cperm[jj]);

               y -= x * rval[j];

               p_work[jj] = y + ((y == 0) ? SOPLEX_FACTOR_MARKER : 0.0);
            }
         }

         if(lbeg[l.firstUnused - 1] == ll)
            (l.firstUnused)--;
         else
            lbeg[l.firstUnused] = ll;


         /*  Set diagonal value
          */
         if(i != r)
         {
            this->stat = SLinSolver<R>::SINGULAR;
            throw SPxStatusException("XFORE01 The loaded matrix is singular");
         }

         k = corig[r];

         x = p_work[k];
         diag[rowno] = 1 / x;
         p_work[k] = 0.0;


         /*  make row large enough to fit all nonzeros.
          */

         if(rmax[rowno] < num)
         {
            rlen[rowno] = 0;
            remaxRow(rowno, num);
            rval = u.row.val.data();
            ridx = u.row.idx;
         }

         nzCnt += num;

         /*  Insert work to updated row thereby clearing work;
          */
         n = rbeg[rowno];

         for(i = 0; i < num; ++i)
         {
            j = corig[nonz[i]];
            x = p_work[j];

            // BH 2005-08-24: This if is very important. It may well happen that
            // during the elimination of row r a nonzero elements cancels out
            // and becomes zero. This would lead to an infinite loop in the
            // above elimination code, since the corresponding column index would
            // be enqueued for further elimination again and agian.

            if(x != 0.0)
            {
               if(spxAbs(x) > l_maxabs)
                  l_maxabs = spxAbs(x);

               ridx[n] = j;

               rval[n] = x;

               p_work[j] = 0.0;

               ++n;

               if(clen[j] >= cmax[j])
               {
                  forestReMaxCol(j, clen[j] + 1);
                  cidx = u.col.idx;
                  cval = u.col.val.data();
               }

               cval[cbeg[j] + clen[j]] = x;

               cidx[cbeg[j] + clen[j]++] = rowno;
            }
         }

         rlen[rowno] = n - rbeg[rowno];
      }
      else            /* few nonzeros to be eliminated        */
      {
         /*  move row r from U to p_work
          */
         for(i += j - 1; i >= j; --i)
         {
            k = ridx[i];
            p_work[k] = rval[i];
            m = --(clen[k]) + cbeg[k];

            for(h = m; cidx[h] != rowno; --h)
               ;

            cidx[h] = cidx[m];

            cval[h] = cval[m];
         }


         /*  Eliminate row r from U to L file
          */
         ll = makeLvec(r - c, rowno);

         lval = l.val.data();

         lidx = l.idx;

         for(i = c; i < r; ++i)
         {
            k = corig[i];

            if(p_work[k] != 0.0)
            {
               n = rorig[i];
               x = p_work[k] * diag[n];
               lidx[ll] = n;
               lval[ll] = x;
               p_work[k] = 0.0;
               ll++;

               if(spxAbs(x) > l_maxabs)
                  l_maxabs = spxAbs(x);

               j = rbeg[n];

               m = rlen[n] + j;

               for(; j < m; ++j)
                  p_work[ridx[j]] -= x * rval[j];
            }
         }

         if(lbeg[l.firstUnused - 1] == ll)
            (l.firstUnused)--;
         else
            lbeg[l.firstUnused] = ll;


         /*  Set diagonal value
          */
         k = corig[r];

         x = p_work[k];

         if(x == 0.0)
         {
            this->stat = SLinSolver<R>::SINGULAR;
            throw SPxStatusException("XFORE02 The loaded matrix is singular");
            //            return;
         }

         diag[rowno] = 1 / x;

         p_work[k] = 0.0;


         /*  count remaining nonzeros in work and make row large enough
          *  to fit them all.
          */
         n = 0;

         for(i = r + 1; i < dim; ++i)
            if(p_work[corig[i]] != 0.0)
               n++;

         if(rmax[rowno] < n)
         {
            rlen[rowno] = 0;
            remaxRow(rowno, n);
            rval = u.row.val.data();
            ridx = u.row.idx;
         }

         nzCnt += n;

         /*  Insert p_work to updated row thereby clearing p_work;
          */
         n = rbeg[rowno];

         for(i = r + 1; i < dim; ++i)
         {
            j = corig[i];
            x = p_work[j];

            if(x != 0.0)
            {
               if(spxAbs(x) > l_maxabs)
                  l_maxabs = spxAbs(x);

               ridx[n] = j;

               rval[n] = x;

               p_work[j] = 0.0;

               ++n;

               if(clen[j] >= cmax[j])
               {
                  forestReMaxCol(j, clen[j] + 1);
                  cidx = u.col.idx;
                  cval = u.col.val.data();
               }

               cval[cbeg[j] + clen[j]] = x;

               cidx[cbeg[j] + clen[j]++] = rowno;
            }
         }

         rlen[rowno] = n - rbeg[rowno];
      }
   }

   else if(r == c)
   {
      /*  Move diagonal element to diag.  Note, that it must be the last
       *  element, since it has just been inserted above.
       */
      rowno = rorig[r];
      i = rbeg[rowno] + --(rlen[rowno]);
      diag[rowno] = 1 / rval[i];

      for(j = i = --(clen[p_col]) + cbeg[p_col]; cidx[i] != rowno; --i)
         ;

      cidx[i] = cidx[j];

      cval[i] = cval[j];
   }
   else /* r < c */
   {
      this->stat = SLinSolver<R>::SINGULAR;
      throw SPxStatusException("XFORE03 The loaded matrix is singular");
      //      return;
   }

   maxabs = l_maxabs;

   assert(isConsistent());
   this->stat = SLinSolver<R>::OK;
}

template <class R>
void CLUFactor<R>::update(int p_col, R* p_work, const int* p_idx, int num)
{
   int ll, i, j;
   int* lidx;
   R* lval;
   R x, rezi;

   assert(p_work[p_col] != 0.0);
   rezi = 1 / p_work[p_col];
   p_work[p_col] = 0.0;

   ll = makeLvec(num, p_col);
   //   ll = fac->makeLvec(num, col);
   lval = l.val.data();
   lidx = l.idx;

   for(i = num - 1; (j = p_idx[i]) != p_col; --i)
   {
      lidx[ll] = j;
      lval[ll] = rezi * p_work[j];
      p_work[j] = 0.0;
      ++ll;
   }

   lidx[ll] = p_col;

   lval[ll] = 1 - rezi;
   ++ll;

   for(--i; i >= 0; --i)
   {
      j = p_idx[i];
      lidx[ll] = j;
      lval[ll] = x = rezi * p_work[j];
      p_work[j] = 0.0;
      ++ll;

      if(spxAbs(x) > maxabs)
         maxabs = spxAbs(x);
   }

   this->stat = SLinSolver<R>::OK;
}

template <class R>
void CLUFactor<R>::updateNoClear(
   int p_col,
   const R* p_work,
   const int* p_idx,
   int num)
{
   int ll, i, j;
   int* lidx;
   R* lval;
   R x, rezi;

   assert(p_work[p_col] != 0.0);
   rezi = 1 / p_work[p_col];
   ll = makeLvec(num, p_col);
   //ll = fac->makeLvec(num, col);
   lval = l.val.data();
   lidx = l.idx;

   for(i = num - 1; (j = p_idx[i]) != p_col; --i)
   {
      lidx[ll] = j;
      lval[ll] = rezi * p_work[j];
      ++ll;
   }

   lidx[ll] = p_col;

   lval[ll] = 1 - rezi;
   ++ll;

   for(--i; i >= 0; --i)
   {
      j = p_idx[i];
      lidx[ll] = j;
      lval[ll] = x = rezi * p_work[j];
      ++ll;

      if(spxAbs(x) > maxabs)
         maxabs = spxAbs(x);
   }

   this->stat = SLinSolver<R>::OK;
}

/*****************************************************************************/
/*
 *      Temporary data structures.
 */

/*
 *        For the i=th column the situation might look like this:
 *
 * \begin{verbatim}
 *        idx     = ....................iiiIIIIII-----..............
 *        cbeg[i] =                     ^
 *        cact[i] =                        +----+
 *        clen[i] =                     +-------+
 *        cmax[i] =                     +------------+
 *
 *        Indices clen[i]-cbeg[i]:      ^^^
 * \end{verbatim}
 *        belong to column i, but have already been pivotal and don't belong to
 *        the active submatrix.
 */

/****************************************************************************/
/*
 *      Initialize row and column file of working matrix and
 *      mark column singletons.
 */
template <class R>
void CLUFactor<R>::initFactorMatrix(const SVectorBase<R>** vec, const R eps)
{

   R x;
   int m;
   int tot;
   Dring* rring, *lastrring;
   Dring* cring, *lastcring;
   const SVectorBase<R>* psv;
   int* sing = temp.s_mark;

   /*  Initialize:
    *  - column file thereby remembering column singletons in |sing|.
    *  - nonzeros counts per row
    *  - total number of nonzeros
    */

   for(int i = 0; i < thedim; i++)
      u.row.max[i] = u.row.len[i] = 0;

   tot = 0;

   for(int i = 0; i < thedim; i++)
   {
      int k;

      psv = vec[i];
      k = psv->size();

      if(k > 1)
      {
         tot += k;

         for(int j = 0; j < k; ++j)
            u.row.max[psv->index(j)]++;
      }
      else if(k == 0)
      {
         this->stat = SLinSolver<R>::SINGULAR;
         return;
      }
   }

   /*  Resize nonzero memory if necessary
    */
   minRowMem(int(rowMemMult * tot));

   minColMem(int(colMemMult * tot));

   minLMem(int(lMemMult * tot));


   /*  Initialize:
    *  - row ring lists
    *  - row vectors in file
    *  - column ring lists
    */
   u.row.start[0] = 0;

   rring = u.row.elem;

   lastrring = &(u.row.list);

   lastrring->idx = thedim;

   lastrring->next = rring;

   cring = u.col.elem;

   lastcring = &(u.col.list);

   lastcring->idx = thedim;

   lastcring->next = cring;

   m = 0;

   for(int i = 0; i < thedim; i++)
   {
      u.row.start[i] = m;
      m += u.row.max[i];

      rring->idx = i;
      rring->prev = lastrring;
      lastrring->next = rring;
      lastrring = rring;
      ++rring;

      cring->idx = i;
      cring->prev = lastcring;
      lastcring->next = cring;
      lastcring = cring;
      ++cring;
   }

   u.row.start[thedim]       = 0;

   u.row.max[thedim]       = 0;
   u.row.used = m;

   lastrring->next = &(u.row.list);
   lastrring->next->prev = lastrring;

   lastcring->next = &(u.col.list);
   lastcring->next->prev = lastcring;

   /*  Copy matrix to row and column file
    *  excluding and marking column singletons!
    */
   m = 0;
   temp.stage = 0;

   initMaxabs = 0;

   for(int i = 0; i < thedim; i++)
   {
      int nnonzeros;

      psv = vec[i];
      u.col.start[i] = m;

      /* check whether number of nonzeros above tolerance is 0, 1 or >= 2 */
      nnonzeros = 0;

      for(int j = 0; j < psv->size() && nnonzeros <= 1; j++)
      {
         if(isNotZero(psv->value(j), eps))
            nnonzeros++;
      }

      /* basis is singular due to empty column */
      if(nnonzeros == 0)
      {
         this->stat = SLinSolver<R>::SINGULAR;
         return;
      }

      /* exclude column singletons */
      else if(nnonzeros == 1)
      {
         int j = 0;

         /* find nonzero */

         for(j = 0; isZero(psv->value(j), eps); j++)
            ;

         assert(j < psv->size());

         /* basis is singular due to two linearly dependent column singletons */
         if(row.perm[psv->index(j)] >= 0)
         {
            this->stat = SLinSolver<R>::SINGULAR;
            return;
         }

         /* update maximum absolute nonzero value */
         x = psv->value(j);

         if(spxAbs(x) > initMaxabs)
            initMaxabs = spxAbs(x);

         /* permute to front and mark as singleton */
         setPivot(temp.stage, i, psv->index(j), x);

         sing[temp.stage] = i;

         temp.stage++;

         /* set column length to zero */
         temp.s_cact[i] = u.col.len[i] = u.col.max[i] = 0;
      }

      /* add to active matrix if not a column singleton */
      else
      {
         int end;
         int k;

         /* go through all nonzeros in column */
         assert(nnonzeros >= 2);
         nnonzeros = 0;

         for(int j = 0; j < psv->size(); j++)
         {
            x = psv->value(j);

            if(isNotZero(x, eps))
            {
               /* add to column array */
               k = psv->index(j);
               u.col.idx[m] = k;
               m++;

               /* add to row array */
               end = u.row.start[k] + u.row.len[k];
               u.row.idx[end] = i;
               u.row.val[end] = x;
               u.row.len[k]++;

               /* update maximum absolute nonzero value */

               if(spxAbs(x) > initMaxabs)
                  initMaxabs = spxAbs(x);

               nnonzeros++;
            }
         }

         assert(nnonzeros >= 2);

         /* set column length */
         temp.s_cact[i] = u.col.len[i] = u.col.max[i] = nnonzeros;
      }
   }

   u.col.used = m;
}



/*****************************************************************************/
/*
 *      Remove column singletons
 */
template <class R>
void CLUFactor<R>::colSingletons()
{
   int i, j, k, n;
   int len;
   int p_col, p_row, newrow;
   int* idx;
   int* rorig = row.orig;
   int* rperm = row.perm;
   int* sing = temp.s_mark;


   /*  Iteratively update column counts due to removed column singletons
    *  thereby removing new arising columns singletons
    *  and computing the index of the first row singleton (-1)
    *  until no more can be found.
    */

   for(i = 0; i < temp.stage; ++i)
   {
      p_row = rorig[i];
      assert(p_row >= 0);
      idx = &(u.row.idx[u.row.start[p_row]]);
      len = u.row.len[p_row];

      for(j = 0; j < len; ++j)
      {
         /*  Move pivotal nonzeros to front of column.
          */
         p_col = idx[j];
         assert(temp.s_cact[p_col] > 0);

         n = u.col.start[p_col] + u.col.len[p_col] - temp.s_cact[p_col];

         for(k = n; u.col.idx[k] != p_row; ++k)
            ;

         assert(k < u.col.start[p_col] + u.col.len[p_col]);

         u.col.idx[k] = u.col.idx[n];

         u.col.idx[n] = p_row;

         n = --(temp.s_cact[p_col]);          /* column nonzeros of ACTIVE matrix */

         if(n == 1)                   /* Here is another singleton */
         {
            newrow = u.col.idx[--u.col.len[p_col] + u.col.start[p_col]];

            /*      Ensure, matrix not singular
             */

            if(rperm[newrow] >= 0)
            {
               this->stat = SLinSolver<R>::SINGULAR;
               return;
            }

            /*      Find singleton in row.
             */
            n = u.row.start[newrow] + (--(u.row.len[newrow]));

            for(k = n; u.row.idx[k] != p_col; --k)
               ;

            /*      Remove singleton from column.
             */
            setPivot(temp.stage, p_col, newrow, u.row.val[k]);

            sing[temp.stage++] = p_col;

            /*      Move pivot element to diag.
             */
            u.row.val[k] = u.row.val[n];

            u.row.idx[k] = u.row.idx[n];
         }
         else if(n == 0)
         {
            this->stat = SLinSolver<R>::SINGULAR;
            return;
         }
      }
   }

   assert(temp.stage <= thedim);
}


/*****************************************************************************/
/*
 *      Remove row singletons
 */
template <class R>
void CLUFactor<R>::rowSingletons()
{
   R pval;
   int i, j, k, ll, r;
   int p_row, p_col, len, rs, lk;
   int* idx;
   int* rperm = row.perm;
   int* sing = temp.s_mark;

   /*  Mark row singletons
    */
   rs = temp.stage;

   for(i = 0; i < thedim; ++i)
   {
      if(rperm[i] < 0 && u.row.len[i] == 1)
         sing[temp.stage++] = i;
   }

   /*  Eliminate row singletons
    *  thereby marking newly arising ones
    *  until no more can be found.
    */
   for(; rs < temp.stage; ++rs)
   {
      /*      Move pivot element from row file to diag
       */
      p_row = sing[rs];
      j = u.row.start[p_row];
      p_col = u.row.idx[j];
      pval = u.row.val[j];
      setPivot(rs, p_col, p_row, pval);
      u.row.len[p_row] = 0;

      /*      Remove pivot column form workingmatrix
       *      thereby building up L VectorBase<R>.
       */
      idx = &(u.col.idx[u.col.start[p_col]]);
      i = temp.s_cact[p_col];                /* nr. nonzeros of new L VectorBase<R> */
      lk = makeLvec(i - 1, p_row);
      len = u.col.len[p_col];
      i = (u.col.len[p_col] -= i);         /* remove pivot column from U */

      for(; i < len; ++i)
      {
         r = idx[i];

         if(r != p_row)
         {
            /*      Find pivot column in row.
             */
            ll = --(u.row.len[r]);
            k = u.row.start[r] + ll;

            for(j = k; u.row.idx[j] != p_col; --j)
               ;

            assert(k >= u.row.start[r]);

            /*      Initialize L VectorBase<R>
             */
            l.idx[lk] = r;

            l.val[lk] = u.row.val[j] / pval;

            ++lk;

            /*      Remove pivot column from row.
             */
            u.row.idx[j] = u.row.idx[k];

            u.row.val[j] = u.row.val[k];

            /*      Check new row length.
             */
            if(ll == 1)
               sing[temp.stage++] = r;
            else if(ll == 0)
            {
               this->stat = SLinSolver<R>::SINGULAR;
               return;
            }
         }
      }
   }
}


/*****************************************************************************/
/*
 *      Init nonzero number Ring lists
 *      and required entries of arrays max and mark
 */

template <class R>
void CLUFactor<R>::initFactorRings()
{
   int i;
   int* rperm = row.perm;
   int* cperm = col.perm;
   CLUFactor<R>::Pring* ring;

   assert(thedim >= 0);
   spx_alloc(temp.pivot_col,   thedim + 1);
   spx_alloc(temp.pivot_colNZ, thedim + 1);
   spx_alloc(temp.pivot_row,   thedim + 1);
   spx_alloc(temp.pivot_rowNZ, thedim + 1);

   for(i = thedim - temp.stage; i >= 0; --i)
   {
      initDR(temp.pivot_colNZ[i]);
      initDR(temp.pivot_rowNZ[i]);
   }

   for(i = 0; i < thedim; ++i)
   {
      if(rperm[i] < 0)
      {
         if(u.row.len[i] <= 0)
         {
            this->stat = SLinSolver<R>::SINGULAR;
            return;
         }

         ring = &(temp.pivot_rowNZ[u.row.len[i]]);

         init2DR(temp.pivot_row[i], *ring);
         temp.pivot_row[i].idx = i;
         temp.s_max[i] = -1;
      }

      if(cperm[i] < 0)
      {
         if(temp.s_cact[i] <= 0)
         {
            this->stat = SLinSolver<R>::SINGULAR;
            return;
         }

         ring = &(temp.pivot_colNZ[temp.s_cact[i]]);

         init2DR(temp.pivot_col[i], *ring);
         temp.pivot_col[i].idx = i;
         temp.s_mark[i] = 0;
      }
   }
}

template <class R>
void CLUFactor<R>::freeFactorRings(void)
{

   if(temp.pivot_col)
      spx_free(temp.pivot_col);

   if(temp.pivot_colNZ)
      spx_free(temp.pivot_colNZ);

   if(temp.pivot_row)
      spx_free(temp.pivot_row);

   if(temp.pivot_rowNZ)
      spx_free(temp.pivot_rowNZ);
}


/*
 *      Eliminate all row singletons from nucleus.
 *      A row singleton may well be column singleton at the same time!
 */
template <class R>
void CLUFactor<R>::eliminateRowSingletons()
{
   int i, j, k, ll, r;
   int len, lk;
   int pcol, prow;
   R pval;
   int* idx;
   CLUFactor<R>::Pring* sing;

   for(sing = temp.pivot_rowNZ[1].prev; sing != &(temp.pivot_rowNZ[1]); sing = sing->prev)
   {
      prow = sing->idx;
      i = u.row.start[prow];
      pcol = u.row.idx[i];
      pval = u.row.val[i];
      setPivot(temp.stage++, pcol, prow, pval);
      u.row.len[prow] = 0;
      removeDR(temp.pivot_col[pcol]);

      /*      Eliminate pivot column and build L VectorBase<R>.
       */
      i = temp.s_cact[pcol];

      if(i > 1)
      {
         idx = &(u.col.idx[u.col.start[pcol]]);
         len = u.col.len[pcol];
         lk = makeLvec(i - 1, prow);
         i = u.col.len[pcol] -= i;

         for(; (r = idx[i]) != prow; ++i)
         {
            /*      Find pivot column in row.
             */
            ll = --(u.row.len[r]);
            k = u.row.start[r] + ll;

            for(j = k; u.row.idx[j] != pcol; --j)
               ;

            assert(j >= u.row.start[r]);

            /*      Initialize L VectorBase<R>
             */
            l.idx[lk] = r;

            l.val[lk] = u.row.val[j] / pval;

            ++lk;

            /*      Remove pivot column from row.
             */
            u.row.idx[j] = u.row.idx[k];

            u.row.val[j] = u.row.val[k];

            /*      Move column to appropriate nonzero ring.
             */
            removeDR(temp.pivot_row[r]);

            init2DR(temp.pivot_row[r], temp.pivot_rowNZ[ll]);

            assert(row.perm[r] < 0);

            temp.s_max[r] = -1;
         }

         /* skip pivot element */
         assert(i < len && "ERROR: pivot column does not contain pivot row");

         for(++i; i < len; ++i)
         {
            /*      Find pivot column in row.
             */
            r = idx[i];
            ll = --(u.row.len[r]);
            k = u.row.start[r] + ll;

            for(j = k; u.row.idx[j] != pcol; --j)
               ;

            assert(j >= u.row.start[r]);

            /*      Initialize L VectorBase<R>
             */
            l.idx[lk] = r;

            l.val[lk] = u.row.val[j] / pval;

            ++lk;

            /*      Remove pivot column from row.
             */
            u.row.idx[j] = u.row.idx[k];

            u.row.val[j] = u.row.val[k];

            /*      Move column to appropriate nonzero ring.
             */
            removeDR(temp.pivot_row[r]);

            init2DR(temp.pivot_row[r], temp.pivot_rowNZ[ll]);

            assert(row.perm[r] < 0);

            temp.s_max[r] = -1;
         }
      }
      else
         u.col.len[pcol] -= i;
   }

   initDR(temp.pivot_rowNZ[1]);    /* Remove all row singletons from list */
}



/*
 *      Eliminate all column singletons from nucleus.
 *      A column singleton must not be row singleton at the same time!
 */
template <class R>
void CLUFactor<R>::eliminateColSingletons()
{
   int i, j, k, m, c;
   int pcol, prow;
   CLUFactor<R>::Pring* sing;

   for(sing = temp.pivot_colNZ[1].prev;
         sing != &(temp.pivot_colNZ[1]);
         sing = sing->prev)
   {
      /*      Find pivot value
       */
      pcol = sing->idx;
      j = --(u.col.len[pcol]) + u.col.start[pcol];   /* remove pivot column */
      prow = u.col.idx[j];
      removeDR(temp.pivot_row[prow]);

      j = --(u.row.len[prow]) + u.row.start[prow];

      for(i = j; (c = u.row.idx[i]) != pcol; --i)
      {
         m = u.col.len[c] + u.col.start[c] - (temp.s_cact[c])--;

         for(k = m; u.col.idx[k] != prow; ++k)
            ;

         u.col.idx[k] = u.col.idx[m];

         u.col.idx[m] = prow;

         m = temp.s_cact[c];

         removeDR(temp.pivot_col[c]);

         init2DR(temp.pivot_col[c], temp.pivot_colNZ[m]);

         assert(col.perm[c] < 0);
      }

      /*      remove pivot element from pivot row
       */
      setPivot(temp.stage++, pcol, prow, u.row.val[i]);

      u.row.idx[i] = u.row.idx[j];

      u.row.val[i] = u.row.val[j];

      j = u.row.start[prow];

      for(--i; i >= j; --i)
      {
         c = u.row.idx[i];
         m = u.col.len[c] + u.col.start[c] - (temp.s_cact[c])--;

         for(k = m; u.col.idx[k] != prow; ++k)
            ;

         u.col.idx[k] = u.col.idx[m];

         u.col.idx[m] = prow;

         m = temp.s_cact[c];

         removeDR(temp.pivot_col[c]);

         init2DR(temp.pivot_col[c], temp.pivot_colNZ[m]);

         assert(col.perm[c] < 0);
      }
   }

   initDR(temp.pivot_colNZ[1]);    /* Remove all column singletons from list */
}

/*
 * No singletons available: Select pivot elements.
 */
template <class R>
void CLUFactor<R>::selectPivots(R threshold)
{
   int ii;
   int i;
   int j;
   int k;
   int ll = -1; // This value should never be used.
   int kk;
   int m;
   int count;
   int num;
   int rw = -1; // This value should never be used.
   int cl = -1; // This value should never be used.
   int len;
   int beg;
   R l_maxabs;
   R x = R(0.0);   // This value should never be used.
   int mkwtz;
   int candidates;

   candidates = thedim - temp.stage - 1;

   if(candidates > 4)
      candidates = 4;

   num = 0;

   count = 2;

   for(;;)
   {
      ii = -1;

      if(temp.pivot_rowNZ[count].next != &(temp.pivot_rowNZ[count]))
      {
         rw = temp.pivot_rowNZ[count].next->idx;
         beg = u.row.start[rw];
         len = u.row.len[rw] + beg - 1;

         /*  set l_maxabs to maximum absolute value in row
          *  (compute it if necessary).
          */

         if((l_maxabs = temp.s_max[rw]) < 0)
         {
            l_maxabs = spxAbs(u.row.val[len]);

            for(i = len - 1; i >= beg; --i)
               if(l_maxabs < spxAbs(u.row.val[i]))
                  l_maxabs = spxAbs(u.row.val[i]);

            temp.s_max[rw] = l_maxabs;               /* ##### */
         }

         l_maxabs *= threshold;

         /*  select pivot element with lowest markowitz number in row
          */
         mkwtz = thedim + 1;

         for(i = len; i >= beg; --i)
         {
            k = u.row.idx[i];
            j = temp.s_cact[k];
            x = u.row.val[i];

            if(j < mkwtz && spxAbs(x) > l_maxabs)
            {
               mkwtz = j;
               cl = k;
               ii = i;

               if(j <= count)               /* ##### */
                  break;
            }
         }
      }
      else if(temp.pivot_colNZ[count].next != &(temp.pivot_colNZ[count]))
      {
         cl = temp.pivot_colNZ[count].next->idx;
         beg = u.col.start[cl];
         len = u.col.len[cl] + beg - 1;
         beg = len - temp.s_cact[cl] + 1;
         assert(count == temp.s_cact[cl]);

         /*  select pivot element with lowest markowitz number in column
          */
         mkwtz = thedim + 1;

         for(i = len; i >= beg; --i)
         {
            k = u.col.idx[i];
            j = u.row.len[k];

            if(j < mkwtz)
            {
               /*  ensure that element (cl,k) is stable.
                */
               if(temp.s_max[k] > 0)
               {
                  /*  case 1: l_maxabs is known
                   */
                  for(m = u.row.start[k], kk = m + u.row.len[k] - 1;
                        kk >= m; --kk)
                  {
                     if(u.row.idx[kk] == cl)
                     {
                        x = u.row.val[kk];
                        ll = kk;
                        break;
                     }
                  }

                  l_maxabs = temp.s_max[k];
               }
               else
               {
                  /*  case 2: l_maxabs needs to be computed
                   */
                  m = u.row.start[k];
                  l_maxabs = spxAbs(u.row.val[m]);

                  for(kk = m + u.row.len[k] - 1; kk >= m; --kk)
                  {
                     if(l_maxabs < spxAbs(u.row.val[kk]))
                        l_maxabs = spxAbs(u.row.val[kk]);

                     if(u.row.idx[kk] == cl)
                     {
                        x = u.row.val[kk];
                        ll = kk;
                        break;
                     }
                  }

                  for(--kk; kk > m; --kk)
                  {
                     if(l_maxabs < spxAbs(u.row.val[kk]))
                        l_maxabs = spxAbs(u.row.val[kk]);
                  }

                  temp.s_max[k] = l_maxabs;
               }

               l_maxabs *= threshold;

               if(spxAbs(x) > l_maxabs)
               {
                  mkwtz = j;
                  rw = k;
                  ii = ll;

                  if(j <= count + 1)
                     break;
               }
            }
         }
      }
      else
      {
         ++count;
         continue;
      }

      assert(cl >= 0);

      removeDR(temp.pivot_col[cl]);
      initDR(temp.pivot_col[cl]);

      if(ii >= 0)
      {
         /*  Initialize selected pivot element
          */
         CLUFactor<R>::Pring* pr;
         temp.pivot_row[rw].pos = ii - u.row.start[rw];
         temp.pivot_row[rw].mkwtz = mkwtz = (mkwtz - 1) * (count - 1);
         // ??? mkwtz originally was long,
         // maybe to avoid an overflow in this instruction?

         for(pr = temp.pivots.next; pr->idx >= 0; pr = pr->next)
         {
            if(pr->idx == rw || pr->mkwtz >= mkwtz)
               break;
         }

         pr = pr->prev;

         if(pr->idx != rw)
         {
            removeDR(temp.pivot_row[rw]);
            init2DR(temp.pivot_row[rw], *pr);
         }

         num++;

         if(num >= candidates)
            break;
      }
   }

   /*
    *     while(temp.temp.next->mkwtz < temp.temp.prev->mkwtz)
    *     {
    *     Pring   *pr;
    *     pr = temp.temp.prev;
    *     removeDR(*pr);
    *     init2DR (*pr, rowNZ[u.row.len[pr->idx]]);
    }
    */

   assert(row.perm[rw] < 0);

   assert(col.perm[cl] < 0);
}


/*
 *      Perform L and update loop for row r
 */
template <class R>
int CLUFactor<R>::updateRow(int r,
                            int lv,
                            int prow,
                            int pcol,
                            R pval,
                            R eps)
{
   int fill;
   R x, lx;
   int c, i, j, k, ll, m, n;

   n = u.row.start[r];
   m = --(u.row.len[r]) + n;

   /*  compute L VectorBase<R> entry and
    *  and remove pivot column form row file
    */

   for(j = m; u.row.idx[j] != pcol; --j)
      ;

   lx = u.row.val[j] / pval;

   l.val[lv] = lx;

   l.idx[lv] = r;

   ++lv;

   u.row.idx[j] = u.row.idx[m];

   u.row.val[j] = u.row.val[m];


   /*  update loop (I) and
    *  computing expected fill
    */
   fill = u.row.len[prow];

   for(j = m - 1; j >= n; --j)
   {
      c = u.row.idx[j];

      if(temp.s_mark[c])
      {
         /*  count fill elements.
          */
         temp.s_mark[c] = 0;
         --fill;

         /*  update row values
          */
         x = u.row.val[j] -= work[c] * lx;

         if(isZero(x, eps))
         {
            /* Eliminate zero from row r
             */
            --u.row.len[r];
            --m;
            u.row.val[j] = u.row.val[m];
            u.row.idx[j] = u.row.idx[m];

            /* Eliminate zero from column c
             */
            --(temp.s_cact[c]);
            k = --(u.col.len[c]) + u.col.start[c];

            for(i = k; u.col.idx[i] != r; --i)
               ;

            u.col.idx[i] = u.col.idx[k];
         }
      }
   }


   /*  create space for fill in row file
    */
   ll = u.row.len[r];

   if(ll + fill > u.row.max[r])
      remaxRow(r, ll + fill);

   ll += u.row.start[r];

   /*  fill creating update loop (II)
    */
   for(j = u.row.start[prow], m = j + u.row.len[prow]; j < m; ++j)
   {
      c = u.row.idx[j];

      if(temp.s_mark[c])
      {
         x = - work[c] * lx;

         if(isNotZero(x, eps))
         {
            /* produce fill element in row r
             */
            u.row.val[ll] = x;
            u.row.idx[ll] = c;
            ll++;
            u.row.len[r]++;

            /* produce fill element in column c
             */

            if(u.col.len[c] >= u.col.max[c])
               remaxCol(c, u.col.len[c] + 1);

            u.col.idx[u.col.start[c] + (u.col.len[c])++] = r;

            temp.s_cact[c]++;
         }
      }
      else
         temp.s_mark[c] = 1;
   }

   /*  move row to appropriate list.
    */
   removeDR(temp.pivot_row[r]);

   init2DR(temp.pivot_row[r], temp.pivot_rowNZ[u.row.len[r]]);

   assert(row.perm[r] < 0);

   temp.s_max[r] = -1;

   return lv;
}

/*
 *      Eliminate pivot element
 */
template <class R>
void CLUFactor<R>::eliminatePivot(int prow, int pos, R eps)
{
   int i, j, k, m = -1;
   int lv = -1;  // This value should never be used.
   int pcol;
   R pval;
   int pbeg = u.row.start[prow];
   int plen = --(u.row.len[prow]);
   int pend = pbeg + plen;


   /*  extract pivot element   */
   i = pbeg + pos;
   pcol = u.row.idx[i];
   pval = u.row.val[i];
   removeDR(temp.pivot_col[pcol]);
   initDR(temp.pivot_col[pcol]);

   /*  remove pivot from pivot row     */
   u.row.idx[i] = u.row.idx[pend];
   u.row.val[i] = u.row.val[pend];

   /*  set pivot element and construct L VectorBase<R> */
   setPivot(temp.stage++, pcol, prow, pval);

   /**@todo If this test failes, lv has no value. I suppose that in this
    *       case none of the loops below that uses lv is executed.
    *       But this is unproven.
    */

   if(temp.s_cact[pcol] - 1 > 0)
      lv = makeLvec(temp.s_cact[pcol] - 1, prow);

   /*  init working VectorBase<R>,
    *  remove pivot row from working matrix
    *  and remove columns from list.
    */
   for(i = pbeg; i < pend; ++i)
   {
      j = u.row.idx[i];
      temp.s_mark[j] = 1;
      work[j] = u.row.val[i];
      removeDR(temp.pivot_col[j]);
      m = u.col.start[j] + u.col.len[j] - temp.s_cact[j];

      for(k = m; u.col.idx[k] != prow; ++k)
         ;

      u.col.idx[k] = u.col.idx[m];

      u.col.idx[m] = prow;

      temp.s_cact[j]--;
   }

   /*  perform L and update loop
    */
   for(i = u.col.len[pcol] - temp.s_cact[pcol];
         (m = u.col.idx[u.col.start[pcol] + i]) != prow;
         ++i)
   {
      assert(row.perm[m] < 0);
      assert(lv >= 0);
      /* coverity[negative_returns] */
      updateRow(m, lv++, prow, pcol, pval, eps);
   }

   /*  skip pivot row  */

   m = u.col.len[pcol];

   for(++i; i < m; ++i)
   {
      assert(lv >= 0);
      /* coverity[negative_returns] */
      updateRow(u.col.idx[u.col.start[pcol] + i], lv++, prow, pcol, pval, eps);
   }

   /*  remove pivot column from column file.
    */
   u.col.len[pcol] -= temp.s_cact[pcol];

   /*  clear working VectorBase<R> and reinsert columns to lists
    */
   for(i = u.row.start[prow], pend = i + plen; i < pend; ++i)
   {
      j = u.row.idx[i];
      work[j] = 0;
      temp.s_mark[j] = 0;
      init2DR(temp.pivot_col[j], temp.pivot_colNZ[temp.s_cact[j]]);
      assert(col.perm[j] < 0);
   }
}


/*
 *      Factorize nucleus.
 */
template <class R>
void CLUFactor<R>::eliminateNucleus(const R eps,
                                    const R threshold)
{
   int r, c;
   CLUFactor<R>::Pring* pivot;

   if(this->stat == SLinSolver<R>::SINGULAR)
      return;

   temp.pivots.mkwtz = -1;

   temp.pivots.idx = -1;

   temp.pivots.pos = -1;

   while(temp.stage < thedim - 1)
   {
#ifndef NDEBUG
      int i;
      // CLUFactorIsConsistent(fac);

      for(i = 0; i < thedim; ++i)
         if(col.perm[i] < 0)
            assert(temp.s_mark[i] == 0);

#endif

      if(temp.pivot_rowNZ[1].next != &(temp.pivot_rowNZ[1]))
         /* row singleton available */
         eliminateRowSingletons();
      else if(temp.pivot_colNZ[1].next != &(temp.pivot_colNZ[1]))
         /* column singleton available */
         eliminateColSingletons();
      else
      {
         initDR(temp.pivots);
         selectPivots(threshold);

         assert(temp.pivots.next != &temp.pivots &&
                "ERROR: no pivot element selected");

         for(pivot = temp.pivots.next; pivot != &temp.pivots;
               pivot = pivot->next)
         {
            eliminatePivot(pivot->idx, pivot->pos, eps);
         }
      }

      if(temp.pivot_rowNZ->next != temp.pivot_rowNZ ||
            temp.pivot_colNZ->next != temp.pivot_colNZ)
      {
         this->stat = SLinSolver<R>::SINGULAR;
         return;
      }
   }

   if(temp.stage < thedim)
   {
      /*      Eliminate remaining element.
       *      Note, that this must be both, column and row singleton.
       */
      assert(temp.pivot_rowNZ[1].next != &(temp.pivot_rowNZ[1]) &&
             "ERROR: one row must be left");
      assert(temp.pivot_colNZ[1].next != &(temp.pivot_colNZ[1]) &&
             "ERROR: one col must be left");
      r = temp.pivot_rowNZ[1].next->idx;
      c = temp.pivot_colNZ[1].next->idx;
      u.row.len[r] = 0;
      u.col.len[c]--;
      setPivot(temp.stage, c, r, u.row.val[u.row.start[r]]);
   }
}

/*****************************************************************************/

template <class R>
int CLUFactor<R>::setupColVals()
{
   int i;
   int n = thedim;

   if(!u.col.val.empty())
      u.col.val.clear();

   u.col.val.reserve(u.col.size); // small performance improvement before the insertion
   u.col.val.insert(u.col.val.begin(), u.col.size, 0);

   for(i = 0; i < thedim; i++)
      u.col.len[i] = 0;

   maxabs = R(0.0);

   for(i = 0; i < thedim; i++)
   {
      int     k   = u.row.start[i];
      int*    idx = &u.row.idx[k];
      R*   val = &u.row.val[k];
      int     len = u.row.len[i];

      n += len;

      while(len-- > 0)
      {
         assert((*idx >= 0) && (*idx < thedim));

         k = u.col.start[*idx] + u.col.len[*idx];

         assert((k >= 0) && (k < u.col.size));

         u.col.len[*idx]++;

         assert(u.col.len[*idx] <= u.col.max[*idx]);

         u.col.idx[k] = i;
         u.col.val[k] = *val;

         if(spxAbs(*val) > maxabs)
            maxabs = spxAbs(*val);

         idx++;

         val++;
      }
   }

   return n;
}

/*****************************************************************************/

#ifdef WITH_L_ROWS
template <class R>
void CLUFactor<R>::setupRowVals()
{
   int   i, j, k, m;
   int   vecs, mem;
   int*  l_row;
   int*  idx;
   R* val;
   int*  beg;
   int*  l_ridx;
   R* l_rval;
   int*  l_rbeg;
   int*  rorig;
   int*  rrorig;
   int*  rperm;
   int*  rrperm;

   vecs  = l.firstUpdate;
   l_row = l.row;
   idx   = l.idx;
   val   = l.val.data();
   beg   = l.start;
   mem   = beg[vecs];

   if(!l.rval.empty())
   {
      l.rval.clear();
   }


   if(l.ridx)
      spx_free(l.ridx);

   if(l.rbeg)
      spx_free(l.rbeg);

   if(l.rorig)
      spx_free(l.rorig);

   if(l.rperm)
      spx_free(l.rperm);

   l.rval.reserve(mem);         // small performance improvement before the insertion
   // Insert mem number of zeros.
   l.rval.insert(l.rval.begin(), mem, 0);

   spx_alloc(l.ridx, mem);

   spx_alloc(l.rbeg, thedim + 1);

   spx_alloc(l.rorig, thedim);

   spx_alloc(l.rperm, thedim);

   l_ridx = l.ridx;

   l_rval = l.rval.data();

   l_rbeg = l.rbeg;

   rorig  = l.rorig;

   rrorig = row.orig;

   rperm  = l.rperm;

   rrperm = row.perm;

   for(i = thedim; i--; *l_rbeg++ = 0)
   {
      *rorig++ = *rrorig++;
      *rperm++ = *rrperm++;
   }

   *l_rbeg = 0;

   l_rbeg = l.rbeg + 1;

   for(i = mem; i--;)
      l_rbeg[*idx++]++;

   idx = l.idx;

   for(m = 0, i = thedim; i--; l_rbeg++)
   {
      j = *l_rbeg;
      *l_rbeg = m;
      m += j;
   }

   assert(m == mem);

   l_rbeg = l.rbeg + 1;

   for(i = j = 0; i < vecs; ++i)
   {
      m = l_row[i];
      assert(idx == &l.idx[l.start[i]]);

      for(; j < beg[i + 1]; j++)
      {
         k = l_rbeg[*idx++]++;
         assert(k < mem);
         l_ridx[k] = m;
         l_rval[k] = *val++;
      }
   }

   assert(l.rbeg[thedim] == mem);

   assert(l.rbeg[0] == 0);
}

#endif

/*****************************************************************************/

template <class R>
void CLUFactor<R>::factor(const SVectorBase<R>**
                          vec,          ///< Array of column VectorBase<R> pointers
                          R            threshold,    ///< pivoting threshold
                          R            eps)          ///< epsilon for zero detection
{

   factorTime->start();

   this->stat = SLinSolver<R>::OK;

   l.start[0]    = 0;
   l.firstUpdate = 0;
   l.firstUnused = 0;

   temp.init(thedim);
   initPerm();

   initFactorMatrix(vec, eps);

   if(this->stat)
      goto TERMINATE;

   //   initMaxabs = initMaxabs;

   colSingletons();

   if(this->stat != SLinSolver<R>::OK)
      goto TERMINATE;

   rowSingletons();

   if(this->stat != SLinSolver<R>::OK)
      goto TERMINATE;

   if(temp.stage < thedim)
   {
      initFactorRings();
      eliminateNucleus(eps, threshold);
      freeFactorRings();
   }

TERMINATE:

   l.firstUpdate = l.firstUnused;

   if(this->stat == SLinSolver<R>::OK)
   {
#ifdef WITH_L_ROWS
      setupRowVals();
#endif
      nzCnt = setupColVals();
   }

   factorTime->stop();

   factorCount++;
}

template <class R>
void CLUFactor<R>::dump() const
{
   int i, j, k;

   // Dump regardless of the verbosity level if this method is called;

   /*  Dump U:
    */

   for(i = 0; i < thedim; ++i)
   {
      if(row.perm[i] >= 0)
         std::cout << "DCLUFA01 diag[" << i << "]: [" << col.orig[row.perm[i]]
                   << "] = " << diag[i] << std::endl;

      for(j = 0; j < u.row.len[i]; ++j)
         std::cout << "DCLUFA02   u[" << i << "]: ["
                   << u.row.idx[u.row.start[i] + j] << "] = "
                   << u.row.val[u.row.start[i] + j] << std::endl;
   }

   /*  Dump L:
    */
   for(i = 0; i < thedim; ++i)
   {
      for(j = 0; j < l.firstUnused; ++j)
         if(col.orig[row.perm[l.row[j]]] == i)
         {
            std::cout << "DCLUFA03 l[" << i << "]" << std::endl;

            for(k = l.start[j]; k < l.start[j + 1]; ++k)
               std::cout << "DCLUFA04   l[" << k - l.start[j]
                         << "]:  [" << l.idx[k]
                         << "] = "  << l.val[k] << std::endl;

            break;
         }
   }

   return;
}

/*****************************************************************************/
/*
 *      Ensure that row memory is at least size.
 */
template <class R>
void CLUFactor<R>::minRowMem(int size)
{

   if(u.row.size < size)
   {
      u.row.size = size;
      u.row.val.resize(size);
      spx_realloc(u.row.idx, size);
   }
}

/*****************************************************************************/
/*
 *      Ensure that column memory is at least size.
 */
template <class R>
void CLUFactor<R>::minColMem(int size)
{

   if(u.col.size < size)
   {
      u.col.size = size;
      spx_realloc(u.col.idx, size);
   }
}

template <class R>
void CLUFactor<R>::forestMinColMem(int size)
{

   if(u.col.size < size)
   {
      u.col.size = size;
      spx_realloc(u.col.idx, size);
      u.col.val.resize(size);
   }
}

template <class R>
void CLUFactor<R>::minLMem(int size)
{

   if(size > l.size)
   {
      l.size = int(0.2 * l.size + size);
      l.val.resize(l.size);
      spx_realloc(l.idx, l.size);
   }
}


template <class R>
int CLUFactor<R>::makeLvec(int p_len, int p_row)
{

   if(l.firstUnused >= l.startSize)
   {
      l.startSize += 100;
      spx_realloc(l.start, l.startSize);
   }

   int* p_lrow = l.row;

   int* p_lbeg = l.start;
   int first   = p_lbeg[l.firstUnused];

   assert(p_len > 0 && "ERROR: no empty columns allowed in L vectors");

   minLMem(first + p_len);
   p_lrow[l.firstUnused] = p_row;
   l.start[++(l.firstUnused)] = first + p_len;

   assert(l.start[l.firstUnused] <= l.size);
   assert(l.firstUnused <= l.startSize);
   return first;
}


/*****************************************************************************/

template <class R>
bool CLUFactor<R>::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS
   int              i, j, k, ll;
   Dring*            ring;
   CLUFactor<R>::Pring* pring;

   /*  Consistency only relevant for R factorizations
    */

   if(this->stat)
      return true;

   /*  Test column ring list consistency.
    */
   i = 0;

   for(ring = u.col.list.next; ring != &(u.col.list); ring = ring->next)
   {
      assert(ring->idx >= 0);
      assert(ring->idx < thedim);
      assert(ring->prev->next == ring);

      if(ring != u.col.list.next)
      {
         assert(u.col.start[ring->prev->idx] + u.col.max[ring->prev->idx]
                == u.col.start[ring->idx]);
      }

      ++i;
   }

   assert(i == thedim);

   assert(u.col.start[ring->prev->idx] + u.col.max[ring->prev->idx]
          == u.col.used);


   /*  Test row ring list consistency.
    */
   i = 0;

   for(ring = u.row.list.next; ring != &(u.row.list); ring = ring->next)
   {
      assert(ring->idx >= 0);
      assert(ring->idx < thedim);
      assert(ring->prev->next == ring);
      assert(u.row.start[ring->prev->idx] + u.row.max[ring->prev->idx]
             == u.row.start[ring->idx]);
      ++i;
   }

   assert(i == thedim);

   assert(u.row.start[ring->prev->idx] + u.row.max[ring->prev->idx]
          == u.row.used);


   /*  Test consistency of individual svectors.
    */

   for(i = 0; i < thedim; ++i)
   {
      assert(u.row.max[i] >= u.row.len[i]);
      assert(u.col.max[i] >= u.col.len[i]);
   }


   /*  Test consistency of column file to row file of U
    */
   for(i = 0; i < thedim; ++i)
   {
      for(j = u.row.start[i] + u.row.len[i] - 1; j >= u.row.start[i]; j--)
      {
         k = u.row.idx[j];

         for(ll = u.col.start[k] + u.col.len[k] - 1; ll >= u.col.start[k]; ll--)
         {
            if(u.col.idx[ll] == i)
               break;
         }

         assert(u.col.idx[ll] == i);

         if(row.perm[i] < 0)
         {
            assert(col.perm[k] < 0);
         }
         else
         {
            assert(col.perm[k] < 0 || col.perm[k] > row.perm[i]);
         }
      }
   }

   /*  Test consistency of row file to column file of U
    */
   for(i = 0; i < thedim; ++i)
   {
      for(j = u.col.start[i] + u.col.len[i] - 1; j >= u.col.start[i]; j--)
      {
         k = u.col.idx[j];

         for(ll = u.row.start[k] + u.row.len[k] - 1; ll >= u.row.start[k]; ll--)
         {
            if(u.row.idx[ll] == i)
               break;
         }

         assert(u.row.idx[ll] == i);

         assert(col.perm[i] < 0 || row.perm[k] < col.perm[i]);
      }
   }

   /*  Test consistency of nonzero count lists
    */
   if(temp.pivot_colNZ && temp.pivot_rowNZ)
   {
      for(i = 0; i < thedim - temp.stage; ++i)
      {
         for(pring = temp.pivot_rowNZ[i].next; pring != &(temp.pivot_rowNZ[i]); pring = pring->next)
         {
            assert(row.perm[pring->idx] < 0);
         }

         for(pring = temp.pivot_colNZ[i].next; pring != &(temp.pivot_colNZ[i]); pring = pring->next)
         {
            assert(col.perm[pring->idx] < 0);
         }
      }
   }

#endif // CONSISTENCY_CHECKS

   return true;
}

template <class R>
void CLUFactor<R>::solveUright(R* wrk, R* vec) const
{

   for(int i = thedim - 1; i >= 0; i--)
   {
      int  r = row.orig[i];
      int  c = col.orig[i];
      R x = wrk[c] = diag[r] * vec[r];

      vec[r] = 0.0;

      if(x != 0.0)
         //if (isNotZero(x))
      {
         for(int j = u.col.start[c]; j < u.col.start[c] + u.col.len[c]; j++)
            vec[u.col.idx[j]] -= x * u.col.val[j];
      }
   }
}

template <class R>
int CLUFactor<R>::solveUrightEps(R* vec, int* nonz, R eps, R* rhs)
{
   int i, j, r, c, n;
   int* rorig, *corig;
   int* cidx, *clen, *cbeg;
   R* cval;
   R x;

   int* idx;
   R* val;

   rorig = row.orig;
   corig = col.orig;

   cidx = u.col.idx;
   cval = u.col.val.data();
   clen = u.col.len;
   cbeg = u.col.start;

   n = 0;

   for(i = thedim - 1; i >= 0; --i)
   {
      r = rorig[i];
      x = diag[r] * rhs[r];

      if(isNotZero(x, eps))
      {
         c = corig[i];
         vec[c] = x;
         nonz[n++] = c;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         while(j-- > 0)
            rhs[*idx++] -= x * (*val++);
      }
   }

   return n;
}

template <class R>
void CLUFactor<R>::solveUright2(
   R* p_work1, R* vec1, R* p_work2, R* vec2)
{
   int i, j, r, c;
   int* rorig, *corig;
   int* cidx, *clen, *cbeg;
   R* cval;
   R x1, x2;

   int* idx;
   R* val;

   rorig = row.orig;
   corig = col.orig;

   cidx = u.col.idx;
   cval = u.col.val.data();
   clen = u.col.len;
   cbeg = u.col.start;

   for(i = thedim - 1; i >= 0; --i)
   {
      r = rorig[i];
      c = corig[i];
      p_work1[c] = x1 = diag[r] * vec1[r];
      p_work2[c] = x2 = diag[r] * vec2[r];
      vec1[r] = vec2[r] = 0;

      if(x1 != 0.0 && x2 != 0.0)
      {
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         while(j-- > 0)
         {
            vec1[*idx] -= x1 * (*val);
            vec2[*idx++] -= x2 * (*val++);
         }
      }
      else if(x1 != 0.0)
      {
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         while(j-- > 0)
            vec1[*idx++] -= x1 * (*val++);
      }
      else if(x2 != 0.0)
      {
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         while(j-- > 0)
            vec2[*idx++] -= x2 * (*val++);
      }
   }
}

template <class R>
int CLUFactor<R>::solveUright2eps(
   R* p_work1, R* vec1, R* p_work2, R* vec2,
   int* nonz, R eps)
{
   int i, j, r, c, n;
   int* rorig, *corig;
   int* cidx, *clen, *cbeg;
   bool notzero1, notzero2;
   R* cval;
   R x1, x2;

   int* idx;
   R* val;

   rorig = row.orig;
   corig = col.orig;

   cidx = u.col.idx;
   cval = u.col.val.data();
   clen = u.col.len;
   cbeg = u.col.start;

   n = 0;

   for(i = thedim - 1; i >= 0; --i)
   {
      c = corig[i];
      r = rorig[i];
      p_work1[c] = x1 = diag[r] * vec1[r];
      p_work2[c] = x2 = diag[r] * vec2[r];
      vec1[r] = vec2[r] = 0;
      notzero1 = isNotZero(x1, eps);
      notzero2 = isNotZero(x2, eps);

      if(notzero1 && notzero2)
      {
         *nonz++ = c;
         n++;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         while(j-- > 0)
         {
            vec1[*idx] -= x1 * (*val);
            vec2[*idx++] -= x2 * (*val++);
         }
      }
      else if(notzero1)
      {
         p_work2[c] = 0.0;
         *nonz++ = c;
         n++;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         while(j-- > 0)
            vec1[*idx++] -= x1 * (*val++);
      }
      else if(notzero2)
      {
         p_work1[c] = 0.0;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         while(j-- > 0)
            vec2[*idx++] -= x2 * (*val++);
      }
      else
      {
         p_work1[c] = 0.0;
         p_work2[c] = 0.0;
      }
   }

   return n;
}

template <class R>
void CLUFactor<R>::solveLright(R* vec)
{
   int i, j, k;
   int end;
   R x;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;

   for(i = 0; i < end; ++i)
   {
      if((x = vec[lrow[i]]) != 0.0)
      {
         MSG_DEBUG(std::cout << "y" << lrow[i] << "=" << vec[lrow[i]] << std::endl;)

         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
         {
            MSG_DEBUG(std::cout << "                         -> y" << *idx << " -= " << x << " * " << *val <<
                      " = " << x * (*val) << "    -> " << vec[*idx] - x * (*val) << std::endl;)
            vec[*idx++] -= x * (*val++);
         }
      }
   }

   if(l.updateType)                      /* Forest-Tomlin Updates */
   {
      MSG_DEBUG(std::cout << "performing FT updates..." << std::endl;)

      end = l.firstUnused;

      for(; i < end; ++i)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         StableSum<R> tmp(-vec[lrow[i]]);

         for(j = lbeg[i + 1]; j > k; --j)
            tmp += vec[*idx++] * (*val++);

         vec[lrow[i]] = -R(tmp);

         MSG_DEBUG(std::cout << "y" << lrow[i] << "=" << vec[lrow[i]] << std::endl;)
      }

      MSG_DEBUG(std::cout << "finished FT updates." << std::endl;)
   }
}

template <class R>
void CLUFactor<R>::solveLright2(R* vec1, R* vec2)
{
   int i, j, k;
   int end;
   R x2;
   R x1;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;

   for(i = 0; i < end; ++i)
   {
      x1 = vec1[lrow[i]];
      x2 = vec2[lrow[i]];

      if(x1 != 0 && x2 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
         {
            vec1[*idx] -= x1 * (*val);
            vec2[*idx++] -= x2 * (*val++);
         }
      }
      else if(x1 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
            vec1[*idx++] -= x1 * (*val++);
      }
      else if(x2 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
            vec2[*idx++] -= x2 * (*val++);
      }
   }

   if(l.updateType)                      /* Forest-Tomlin Updates */
   {
      end = l.firstUnused;

      for(; i < end; ++i)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         StableSum<Real> tmp1(-vec1[lrow[i]]);
         StableSum<Real> tmp2(-vec2[lrow[i]]);

         for(j = lbeg[i + 1]; j > k; --j)
         {
            tmp1 += vec1[*idx] * (*val);
            tmp2 += vec2[*idx++] * (*val++);
         }

         vec1[lrow[i]] = -tmp1;

         vec2[lrow[i]] = -tmp2;
      }
   }
}

template <class R>
void CLUFactor<R>::solveUpdateRight(R* vec)
{
   int i, j, k;
   int end;
   R x;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   assert(!l.updateType);               /* no Forest-Tomlin Updates */

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUnused;

   for(i = l.firstUpdate; i < end; ++i)
   {
      if((x = vec[lrow[i]]) != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
            vec[*idx++] -= x * (*val++);
      }
   }
}

template <class R>
void CLUFactor<R>::solveUpdateRight2(R* vec1, R* vec2)
{
   int i, j, k;
   int end;
   R x1, x2;
   R* lval;
   int* lrow, *lidx;
   int* lbeg;

   int* idx;
   R* val;

   assert(!l.updateType);               /* no Forest-Tomlin Updates */

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUnused;

   for(i = l.firstUpdate; i < end; ++i)
   {
      x1 = vec1[lrow[i]];
      x2 = vec2[lrow[i]];

      if(x1 != 0.0 && x2 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
         {
            vec1[*idx] -= x1 * (*val);
            vec2[*idx++] -= x2 * (*val++);
         }
      }
      else if(x1 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
            vec1[*idx++] -= x1 * (*val++);
      }
      else if(x2 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
            vec2[*idx++] -= x2 * (*val++);
      }
   }
}

template <class R>
int CLUFactor<R>::solveRight4update(R* vec, int* nonz, R eps,
                                    R* rhs, R* forest, int* forestNum, int* forestIdx)
{
   solveLright(rhs);

   if(forest)
   {
      int n = 0;

      for(int i = 0; i < thedim; i++)
      {
         forestIdx[n] = i;
         forest[i]    = rhs[i];
         n           += rhs[i] != 0.0 ? 1 : 0;
      }

      *forestNum = n;
   }

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      solveUright(vec, rhs);
      solveUpdateRight(vec);
      return 0;
   }
   else
      return solveUrightEps(vec, nonz, eps, rhs);
}

template <class R>
void CLUFactor<R>::solveRight(R* vec, R* rhs)
{
   solveLright(rhs);
   solveUright(vec, rhs);

   if(!l.updateType)             /* no Forest-Tomlin Updates */
      solveUpdateRight(vec);
}

template <class R>
int CLUFactor<R>::solveRight2update(R* vec1,
                                    R* vec2,
                                    R* rhs1,
                                    R* rhs2,
                                    int* nonz,
                                    R eps,
                                    R* forest,
                                    int* forestNum,
                                    int* forestIdx)
{
   solveLright2(rhs1, rhs2);

   if(forest)
   {
      int n = 0;

      for(int i = 0; i < thedim; i++)
      {
         forestIdx[n] = i;
         forest[i]    = rhs1[i];
         n           += rhs1[i] != 0.0 ? 1 : 0;
      }

      *forestNum = n;
   }

   if(!l.updateType)            /* no Forest-Tomlin Updates */
   {
      solveUright2(vec1, rhs1, vec2, rhs2);
      solveUpdateRight2(vec1, vec2);
      return 0;
   }
   else
      return solveUright2eps(vec1, rhs1, vec2, rhs2, nonz, eps);
}

template <class R>
void CLUFactor<R>::solveRight2(
   R* vec1,
   R* vec2,
   R* rhs1,
   R* rhs2)
{
   solveLright2(rhs1, rhs2);

   if(l.updateType)              /* Forest-Tomlin Updates */
      solveUright2(vec1, rhs1, vec2, rhs2);
   else
   {
      solveUright2(vec1, rhs1, vec2, rhs2);
      solveUpdateRight2(vec1, vec2);
   }
}

/*****************************************************************************/
template <class R>
void CLUFactor<R>::solveUleft(R* p_work, R* vec)
{
   for(int i = 0; i < thedim; ++i)
   {
      int  c  = col.orig[i];
      int  r  = row.orig[i];

      assert(c >= 0);    // Inna/Tobi: otherwise, vec[c] would be strange...
      assert(r >= 0);    // Inna/Tobi: otherwise, diag[r] would be strange...

      R x  = vec[c];


      vec[c]  = 0.0;

      if(x != 0.0)
      {
         DEBUG_CHECK_HUGE_VALUE("WSOLVE01", x);
         DEBUG_CHECK_HUGE_VALUE("WSOLVE02", diag[r]);

         x        *= diag[r];
         p_work[r] = x;

         int end = u.row.start[r] + u.row.len[r];

         for(int m = u.row.start[r]; m < end; m++)
         {
            vec[u.row.idx[m]] -= x * u.row.val[m];
            DEBUG_CHECK_HUGE_VALUE("WSOLVE03", vec[u.row.idx[m]]);
         }
      }
   }
}

template <class R>
void CLUFactor<R>::solveUleft2(
   R* p_work1, R* vec1, R* p_work2, R* vec2)
{
   R x1;
   R x2;
   int i, k, r, c;
   int* rorig, *corig;
   int* ridx, *rlen, *rbeg, *idx;
   R* rval, *val;

   rorig = row.orig;
   corig = col.orig;

   ridx = u.row.idx;
   rval = u.row.val.data();
   rlen = u.row.len;
   rbeg = u.row.start;

   for(i = 0; i < thedim; ++i)
   {
      c = corig[i];
      r = rorig[i];
      x1 = vec1[c];
      x2 = vec2[c];

      if((x1 != 0.0) && (x2 != 0.0))
      {
         DEBUG_CHECK_HUGE_VALUE("WSOLVE04", x1);
         DEBUG_CHECK_HUGE_VALUE("WSOLVE05", x2);
         DEBUG_CHECK_HUGE_VALUE("WSOLVE06", diag[r]);
         x1 *= diag[r];
         x2 *= diag[r];
         p_work1[r] = x1;
         p_work2[r] = x2;
         k = rbeg[r];
         idx = &ridx[k];
         val = &rval[k];

         for(int m = rlen[r]; m != 0; --m)
         {
            vec1[*idx] -= x1 * (*val);
            vec2[*idx] -= x2 * (*val++);
            DEBUG_CHECK_HUGE_VALUE("WSOLVE07", vec1[*idx]);
            DEBUG_CHECK_HUGE_VALUE("WSOLVE08", vec2[*idx]);
            idx++;
         }
      }
      else if(x1 != 0.0)
      {
         DEBUG_CHECK_HUGE_VALUE("WSOLVE09", x1);
         DEBUG_CHECK_HUGE_VALUE("WSOLVE10", diag[r]);
         x1 *= diag[r];
         p_work1[r] = x1;
         k = rbeg[r];
         idx = &ridx[k];
         val = &rval[k];

         for(int m = rlen[r]; m != 0; --m)
         {
            vec1[*idx] -= x1 * (*val++);
            DEBUG_CHECK_HUGE_VALUE("WSOLVE11", vec1[*idx]);
            idx++;
         }
      }
      else if(x2 != 0.0)
      {
         DEBUG_CHECK_HUGE_VALUE("WSOLVE12", x2);
         DEBUG_CHECK_HUGE_VALUE("WSOLVE13", diag[r]);
         x2 *= diag[r];
         p_work2[r] = x2;
         k = rbeg[r];
         idx = &ridx[k];
         val = &rval[k];

         for(int m = rlen[r]; m != 0; --m)
         {
            vec2[*idx] -= x2 * (*val++);
            DEBUG_CHECK_HUGE_VALUE("WSOLVE14", vec2[*idx]);
            idx++;
         }
      }
   }
}

template <class R>
int CLUFactor<R>::solveLleft2forest(
   R* vec1,
   int* /* nonz */,
   R* vec2,
   R /* eps */)
{
   int i;
   int j;
   int k;
   int end;
   R x1, x2;
   R* lval, *val;
   int* lidx, *idx, *lrow;
   int* lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;

   for(i = l.firstUnused - 1; i >= end; --i)
   {
      j = lrow[i];
      x1 = vec1[j];
      x2 = vec2[j];

      if(x1 != 0.0)
      {
         if(x2 != 0.0)
         {
            k = lbeg[i];
            val = &lval[k];
            idx = &lidx[k];

            for(j = lbeg[i + 1]; j > k; --j)
            {
               vec1[*idx] -= x1 * (*val);
               vec2[*idx++] -= x2 * (*val++);
            }
         }
         else
         {
            k = lbeg[i];
            val = &lval[k];
            idx = &lidx[k];

            for(j = lbeg[i + 1]; j > k; --j)
               vec1[*idx++] -= x1 * (*val++);
         }
      }
      else if(x2 != 0.0)
      {
         k = lbeg[i];
         val = &lval[k];
         idx = &lidx[k];

         for(j = lbeg[i + 1]; j > k; --j)
            vec2[*idx++] -= x2 * (*val++);
      }
   }

   return 0;
}

template <class R>
void CLUFactor<R>::solveLleft2(
   R* vec1,
   int* /* nonz */,
   R* vec2,
   R /* eps */)
{
   int i, j, k, r;
   int x1not0, x2not0;
   R x1, x2;

   R* rval, *val;
   int* ridx, *idx;
   int* rbeg;
   int* rorig;

   ridx  = l.ridx;
   rval  = l.rval.data();
   rbeg  = l.rbeg;
   rorig = l.rorig;

#ifndef WITH_L_ROWS
   R*   lval  = l.val.data();
   int*    lidx  = l.idx;
   int*    lrow  = l.row;
   int*    lbeg  = l.start;

   i = l.firstUpdate - 1;

   for(; i >= 0; --i)
   {
      k = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];
      x1 = 0;
      x2 = 0;

      for(j = lbeg[i + 1]; j > k; --j)
      {
         x1 += vec1[*idx] * (*val);
         x2 += vec2[*idx++] * (*val++);
      }

      vec1[lrow[i]] -= x1;

      vec2[lrow[i]] -= x2;
   }

#else

   for(i = thedim; i--;)
   {
      r = rorig[i];
      x1 = vec1[r];
      x2 = vec2[r];
      x1not0 = (x1 != 0);
      x2not0 = (x2 != 0);

      if(x1not0 && x2not0)
      {
         k = rbeg[r];
         j = rbeg[r + 1] - k;
         val = &rval[k];
         idx = &ridx[k];

         while(j-- > 0)
         {
            assert(row.perm[*idx] < i);
            vec1[*idx] -= x1 * *val;
            vec2[*idx++] -= x2 * *val++;
         }
      }
      else if(x1not0)
      {
         k = rbeg[r];
         j = rbeg[r + 1] - k;
         val = &rval[k];
         idx = &ridx[k];

         while(j-- > 0)
         {
            assert(row.perm[*idx] < i);
            vec1[*idx++] -= x1 * *val++;
         }
      }
      else if(x2not0)
      {
         k = rbeg[r];
         j = rbeg[r + 1] - k;
         val = &rval[k];
         idx = &ridx[k];

         while(j-- > 0)
         {
            assert(row.perm[*idx] < i);
            vec2[*idx++] -= x2 * *val++;
         }
      }
   }

#endif
}

template <class R>
int CLUFactor<R>::solveLleftForest(R* vec, int* /* nonz */, R /* eps */)
{
   int i, j, k, end;
   R x;
   R* val, *lval;
   int* idx, *lidx, *lrow, *lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;

   for(i = l.firstUnused - 1; i >= end; --i)
   {
      if((x = vec[lrow[i]]) != 0.0)
      {
         k = lbeg[i];
         val = &lval[k];
         idx = &lidx[k];

         for(j = lbeg[i + 1]; j > k; --j)
            vec[*idx++] -= x * (*val++);
      }
   }

   return 0;
}

template <class R>
void CLUFactor<R>::solveLleft(R* vec) const
{

#ifndef WITH_L_ROWS
   int*  idx;
   R* val;
   R* lval  = l.val.data();
   int*  lidx  = l.idx;
   int*  lrow  = l.row;
   int*  lbeg  = l.start;

   for(int i = l.firstUpdate - 1; i >= 0; --i)
   {
      int k = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];
      x = 0;

      for(int j = lbeg[i + 1]; j > k; --j)
         x += vec[*idx++] * (*val++);

      vec[lrow[i]] -= x;
   }

#else

   for(int i = thedim - 1; i >= 0; --i)
   {
      int  r = l.rorig[i];
      R x = vec[r];

      if(x != 0.0)
      {
         for(int k = l.rbeg[r]; k < l.rbeg[r + 1]; k++)
         {
            int j = l.ridx[k];

            assert(l.rperm[j] < i);

            vec[j] -= x * l.rval[k];
         }
      }
   }

#endif // WITH_L_ROWS
}

template <class R>
int CLUFactor<R>::solveLleftEps(R* vec, int* nonz, R eps)
{
   int i, j, k, n;
   int r;
   R x;
   R* rval, *val;
   int* ridx, *idx;
   int* rbeg;
   int* rorig;

   ridx = l.ridx;
   rval = l.rval.data();
   rbeg = l.rbeg;
   rorig = l.rorig;
   n = 0;
#ifndef WITH_L_ROWS
   R* lval = l.val.data();
   int*  lidx = l.idx;
   int*  lrow = l.row;
   int*  lbeg = l.start;

   for(i = l.firstUpdate - 1; i >= 0; --i)
   {
      k = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];
      x = 0;

      for(j = lbeg[i + 1]; j > k; --j)
         x += vec[*idx++] * (*val++);

      vec[lrow[i]] -= x;
   }

#else

   for(i = thedim; i--;)
   {
      r = rorig[i];
      x = vec[r];

      if(isNotZero(x, eps))
      {
         *nonz++ = r;
         n++;
         k = rbeg[r];
         j = rbeg[r + 1] - k;
         val = &rval[k];
         idx = &ridx[k];

         while(j-- > 0)
         {
            assert(row.perm[*idx] < i);
            vec[*idx++] -= x * *val++;
         }
      }
      else
         vec[r] = 0;
   }

#endif

   return n;
}

template <class R>
void CLUFactor<R>::solveUpdateLeft(R* vec)
{
   int i, j, k, end;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   assert(!l.updateType);               /* Forest-Tomlin Updates */

   end = l.firstUpdate;

   for(i = l.firstUnused - 1; i >= end; --i)
   {
      k = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];
      StableSum<R> tmp(-vec[lrow[i]]);

      for(j = lbeg[i + 1]; j > k; --j)
         tmp += vec[*idx++] * (*val++);

      vec[lrow[i]] = -R(tmp);
   }
}

template <class R>
void CLUFactor<R>::solveUpdateLeft2(R* vec1, R* vec2)
{
   int i, j, k, end;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   assert(!l.updateType);               /* Forest-Tomlin Updates */

   end = l.firstUpdate;

   for(i = l.firstUnused - 1; i >= end; --i)
   {
      k = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];

      StableSum<R> tmp1(-vec1[lrow[i]]);
      StableSum<R> tmp2(-vec2[lrow[i]]);

      for(j = lbeg[i + 1]; j > k; --j)
      {
         tmp1 += vec1[*idx] * (*val);
         tmp2 += vec2[*idx++] * (*val++);
      }

      vec1[lrow[i]] = -tmp1;
      vec2[lrow[i]] = -tmp2;
   }
}

template <class R>
int CLUFactor<R>::solveUpdateLeft(R eps, R* vec, int* nonz, int n)
{
   int i, j, k, end;
   R y;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   assert(!l.updateType);               /* no Forest-Tomlin Updates! */

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;

   for(i = l.firstUnused - 1; i >= end; --i)
   {
      k = lbeg[i];
      assert(k >= 0 && k < l.size);
      val = &lval[k];
      idx = &lidx[k];

      k = lrow[i];

      y = vec[k];
      StableSum<R> tmp(-y);

      for(j = lbeg[i + 1]; j > k; --j)
      {
         assert(*idx >= 0 && *idx < thedim);
         tmp += vec[*idx++] * (*val++);
      }

      if(y == 0)
      {
         y = -R(tmp);

         if(isNotZero(y, eps))
         {
            nonz[n++] = k;
            vec[k] = y;
         }
      }
      else
      {
         y = -R(tmp);
         vec[k] = (y != 0) ? y : SOPLEX_FACTOR_MARKER;
      }
   }

   return n;
}

template <class R>
void CLUFactor<R>::solveLeft(R* vec, R* rhs)
{

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      solveUpdateLeft(rhs);
      solveUleft(vec, rhs);
      solveLleft(vec);
   }
   else
   {
      solveUleft(vec, rhs);
      //@todo is 0.0 the right value for eps here ?
      solveLleftForest(vec, 0, 0.0);
      solveLleft(vec);
   }
}

template <class R>
int CLUFactor<R>::solveLeftEps(R* vec, R* rhs, int* nonz, R eps)
{

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      solveUpdateLeft(rhs);
      solveUleft(vec, rhs);
      return solveLleftEps(vec, nonz, eps);
   }
   else
   {
      solveUleft(vec, rhs);
      solveLleftForest(vec, nonz, eps);
      return solveLleftEps(vec, nonz, eps);
   }
}

template <class R>
int CLUFactor<R>::solveLeft2(
   R* vec1,
   int* nonz,
   R* vec2,
   R eps,
   R* rhs1,
   R* rhs2)
{

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      solveUpdateLeft2(rhs1, rhs2);
      solveUleft2(vec1, rhs1, vec2, rhs2);
      solveLleft2(vec1, nonz, vec2, eps);
      return 0;
   }
   else
   {
      solveUleft2(vec1, rhs1, vec2, rhs2);
      solveLleft2forest(vec1, nonz, vec2, eps);
      solveLleft2(vec1, nonz, vec2, eps);
      return 0;
   }
}

template <class R>
int CLUFactor<R>::solveUleft(R eps,
                             R* vec, int* vecidx,
                             R* rhs, int* rhsidx, int rhsn)
{
   R x, y;
   int i, j, k, n, r, c;
   int* rorig, *corig, *cperm;
   int* ridx, *rlen, *rbeg, *idx;
   R* rval, *val;

   rorig = row.orig;
   corig = col.orig;
   cperm = col.perm;

   /*  move rhsidx to a heap
    */

   for(i = 0; i < rhsn;)
      enQueueMin(rhsidx, &i, cperm[rhsidx[i]]);

   ridx = u.row.idx;

   rval = u.row.val.data();

   rlen = u.row.len;

   rbeg = u.row.start;

   n = 0;

   while(rhsn > 0)
   {
      i = deQueueMin(rhsidx, &rhsn);
      assert(i >= 0 && i < thedim);
      c = corig[i];
      assert(c >= 0 && c < thedim);
      x = rhs[c];
      rhs[c] = 0;

      if(isNotZero(x, eps))
      {
         r = rorig[i];
         assert(r >= 0 && r < thedim);
         vecidx[n++] = r;
         x *= diag[r];
         vec[r] = x;
         k = rbeg[r];
         assert(k >= 0 && k < u.row.size);
         idx = &ridx[k];
         val = &rval[k];

         for(int m = rlen[r]; m; --m)
         {
            j = *idx++;
            assert(j >= 0 && j < thedim);
            y = rhs[j];

            if(y == 0)
            {
               y = -x * (*val++);

               if(isNotZero(y, eps))
               {
                  rhs[j] = y;
                  enQueueMin(rhsidx, &rhsn, cperm[j]);
               }
            }
            else
            {
               y -= x * (*val++);
               rhs[j] = (y != 0) ? y : SOPLEX_FACTOR_MARKER;
            }
         }
      }
   }

   return n;
}


template <class R>
void CLUFactor<R>::solveUleftNoNZ(R eps, R* vec,
                                  R* rhs, int* rhsidx, int rhsn)
{
   R x, y;
   int i, j, k, r, c;
   int* rorig, *corig, *cperm;
   int* ridx, *rlen, *rbeg, *idx;
   R* rval, *val;

   rorig = row.orig;
   corig = col.orig;
   cperm = col.perm;

   /*  move rhsidx to a heap
    */

   for(i = 0; i < rhsn;)
      enQueueMin(rhsidx, &i, cperm[rhsidx[i]]);

   ridx = u.row.idx;

   rval = u.row.val.data();

   rlen = u.row.len;

   rbeg = u.row.start;

   while(rhsn > 0)
   {
      i = deQueueMin(rhsidx, &rhsn);
      assert(i >= 0 && i < thedim);
      c = corig[i];
      assert(c >= 0 && c < thedim);
      x = rhs[c];
      rhs[c] = 0;

      if(isNotZero(x, eps))
      {
         r = rorig[i];
         assert(r >= 0 && r < thedim);
         x *= diag[r];
         vec[r] = x;
         k = rbeg[r];
         assert(k >= 0 && k < u.row.size);
         idx = &ridx[k];
         val = &rval[k];

         for(int m = rlen[r]; m; --m)
         {
            j = *idx++;
            assert(j >= 0 && j < thedim);
            y = rhs[j];

            if(y == 0)
            {
               y = -x * (*val++);

               if(isNotZero(y, eps))
               {
                  rhs[j] = y;
                  enQueueMin(rhsidx, &rhsn, cperm[j]);
               }
            }
            else
            {
               y -= x * (*val++);
               rhs[j] = (y != 0) ? y : SOPLEX_FACTOR_MARKER;
            }
         }
      }
   }
}


template <class R>
int CLUFactor<R>::solveLleftForest(R eps, R* vec, int* nonz, int n)
{
   int i, j, k, end;
   R x, y;
   R* val, *lval;
   int* idx, *lidx, *lrow, *lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;
   end = l.firstUpdate;

   for(i = l.firstUnused - 1; i >= end; --i)
   {
      assert(i >= 0 && i < l.size);

      if((x = vec[lrow[i]]) != 0.0)
      {
         k = lbeg[i];
         assert(k >= 0 && k < l.size);
         val = &lval[k];
         idx = &lidx[k];

         for(j = lbeg[i + 1]; j > k; --j)
         {
            int m = *idx++;
            assert(m >= 0 && m < thedim);
            y = vec[m];

            if(y == 0)
            {
               y = -x * (*val++);

               if(isNotZero(y, eps))
               {
                  vec[m] = y;
                  nonz[n++] = m;
               }
            }
            else
            {
               y -= x * (*val++);
               vec[m] = (y != 0) ? y : SOPLEX_FACTOR_MARKER;
            }
         }
      }
   }

   return n;
}


template <class R>
void CLUFactor<R>::solveLleftForestNoNZ(R* vec)
{
   int i, j, k, end;
   R x;
   R* val, *lval;
   int* idx, *lidx, *lrow, *lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;
   end = l.firstUpdate;

   for(i = l.firstUnused - 1; i >= end; --i)
   {
      if((x = vec[lrow[i]]) != 0.0)
      {
         assert(i >= 0 && i < l.size);
         k = lbeg[i];
         assert(k >= 0 && k < l.size);
         val = &lval[k];
         idx = &lidx[k];

         for(j = lbeg[i + 1]; j > k; --j)
         {
            assert(*idx >= 0 && *idx < thedim);
            vec[*idx++] -= x * (*val++);
         }
      }
   }
}


template <class R>
int CLUFactor<R>::solveLleft(R eps, R* vec, int* nonz, int rn)
{
   int i, j, k, n;
   int r;
   R x, y;
   R* rval, *val;
   int* ridx, *idx;
   int* rbeg;
   int* rorig, *rperm;
   int* last;

   ridx  = l.ridx;
   rval  = l.rval.data();
   rbeg  = l.rbeg;
   rorig = l.rorig;
   rperm = l.rperm;
   n     = 0;

   i = l.firstUpdate - 1;
#ifndef WITH_L_ROWS
#pragma warn "Not yet implemented, define WITH_L_ROWS"
   R*   lval = l.val.data();
   int*    lidx = l.idx;
   int*    lrow = l.row;
   int*    lbeg = l.start;

   for(; i >= 0; --i)
   {
      k   = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];
      x   = 0;

      for(j = lbeg[i + 1]; j > k; --j)
         x += vec[*idx++] * (*val++);

      vec[lrow[i]] -= x;
   }

#else

   /*  move rhsidx to a heap
    */
   for(i = 0; i < rn;)
      enQueueMax(nonz, &i, rperm[nonz[i]]);

   last = nonz + thedim;

   while(rn > 0)
   {
      i = deQueueMax(nonz, &rn);
      r = rorig[i];
      x = vec[r];

      if(isNotZero(x, eps))
      {
         *(--last) = r;
         n++;
         k = rbeg[r];
         j = rbeg[r + 1] - k;
         val = &rval[k];
         idx = &ridx[k];

         while(j-- > 0)
         {
            assert(l.rperm[*idx] < i);
            int m = *idx++;
            y = vec[m];

            if(y == 0)
            {
               y = -x * *val++;

               if(isNotZero(y, eps))
               {
                  vec[m] = y;
                  enQueueMax(nonz, &rn, rperm[m]);
               }
            }
            else
            {
               y -= x * *val++;
               vec[m] = (y != 0) ? y : SOPLEX_FACTOR_MARKER;
            }
         }
      }
      else
         vec[r] = 0;
   }

   for(i = 0; i < n; ++i)
      *nonz++ = *last++;

#endif

   return n;
}


template <class R>
void CLUFactor<R>::solveLleftNoNZ(R* vec)
{
   int i, j, k;
   int r;
   R x;
   R* rval, *val;
   int* ridx, *idx;
   int* rbeg;
   int* rorig;

   ridx = l.ridx;
   rval = l.rval.data();
   rbeg = l.rbeg;
   rorig = l.rorig;

#ifndef WITH_L_ROWS
   R* lval = l.val.data();
   int*    lidx = l.idx;
   int*    lrow = l.row;
   int*    lbeg = l.start;

   i = l.firstUpdate - 1;
   assert(i < thedim);

   for(; i >= 0; --i)
   {
      k = lbeg[i];
      assert(k >= 0 && k < l.size);
      val = &lval[k];
      idx = &lidx[k];
      x = 0;

      for(j = lbeg[i + 1]; j > k; --j)
      {
         assert(*idx >= 0 && *idx < thedim);
         x += vec[*idx++] * (*val++);
      }

      vec[lrow[i]] -= x;
   }

#else

   for(i = thedim; i--;)
   {
      r = rorig[i];
      x = vec[r];

      if(x != 0.0)
      {
         k = rbeg[r];
         j = rbeg[r + 1] - k;
         val = &rval[k];
         idx = &ridx[k];

         while(j-- > 0)
         {
            assert(l.rperm[*idx] < i);
            vec[*idx++] -= x * *val++;
         }
      }
   }

#endif
}

template <class R>
void inline CLUFactor<R>::updateSolutionVectorLright(R change, int j, R& vec, int* idx, int& nnz)
{
   // create a new entry in #ridx
   if(vec == 0.0)
   {
      assert(nnz < thedim);
      idx[nnz] = j;
      ++nnz;
   }

   vec -= change;

   // mark the entry where exact eliminiation occurred
   if(vec == 0.0)
      vec = SOPLEX_FACTOR_MARKER;
}

// solve Lz = b, inplace, using and preserving sparisity structure in the rhs and solution VectorBase<R>
// arrays #vec and #ridx must be large enough to hold #thedim entries!
template <class R>
void CLUFactor<R>::vSolveLright(R* vec, int* ridx, int& rn, R eps)
{
   int i, j, k, n;
   int end;
   R x;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;

   // loop through columns of L
   for(i = 0; i < end; ++i)
   {
      x = vec[lrow[i]];

      // check whether there is a corresponding value in the rhs VectorBase<R>; skipping/ignoring FACTOR_MARKER
      if(isNotZero(x, eps))
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         // apply \f$- x * L_{k,i}\f$ to all corresponding values in rhs/solution VectorBase<R>
         for(j = lbeg[i + 1]; j > k; --j)
         {
            assert(*idx >= 0 && *idx < thedim);
            n = *idx++;
            updateSolutionVectorLright(x * (*val), n, vec[n], ridx, rn);
            ++val;
         }
      }
   }

   if(l.updateType)                     /* Forest-Tomlin Updates */
   {
      end = l.firstUnused;

      for(; i < end; ++i)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         StableSum<R> tmp;

         for(j = lbeg[i + 1]; j > k; --j)
         {
            assert(*idx >= 0 && *idx < thedim);
            tmp += vec[*idx++] * (*val++);
         }

         j = lrow[i];
         x = R(tmp);

         if(isNotZero(x, eps))
            updateSolutionVectorLright(x, j, vec[j], ridx, rn);
      }
   }
}

// solve with L for two right hand sides
// see above methods for documentation
template <class R>
void CLUFactor<R>::vSolveLright2(
   R* vec, int* ridx, int& rn, R eps,
   R* vec2, int* ridx2, int& rn2, R eps2)
{
   int i, j, k, n;
   int end;
   R x, x2;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;

   // loop through columns of L
   for(i = 0; i < end; ++i)
   {
      j = lrow[i];
      x2 = vec2[j];
      x = vec[j];

      // check whether there is a corresponding value in the first rhs VectorBase<R>; skipping/ignoring FACTOR_MARKER
      if(isNotZero(x, eps))
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         // check whether there is  also a corresponding value in the second rhs VectorBase<R>; skipping/ignoring FACTOR_MARKER
         if(isNotZero(x2, eps2))
         {
            for(j = lbeg[i + 1]; j > k; --j)
            {
               assert(*idx >= 0 && *idx < thedim);
               n = *idx++;
               updateSolutionVectorLright(x * (*val), n, vec[n], ridx, rn);
               updateSolutionVectorLright(x2 * (*val), n, vec2[n], ridx2, rn2);
               ++val;
            }
         }
         // only the first VectorBase<R> needs to be modified
         else
         {
            for(j = lbeg[i + 1]; j > k; --j)
            {
               assert(*idx >= 0 && *idx < thedim);
               n = *idx++;
               updateSolutionVectorLright(x * (*val), n, vec[n], ridx, rn);
               ++val;
            }
         }
      }
      // only the second VectorBase<R> needs to be modified
      else if(isNotZero(x2, eps2))
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
         {
            assert(*idx >= 0 && *idx < thedim);
            n = *idx++;
            updateSolutionVectorLright(x2 * (*val), n, vec2[n], ridx2, rn2);
            ++val;
         }
      }
   }

   if(l.updateType)                     /* Forest-Tomlin Updates */
   {
      end = l.firstUnused;

      for(; i < end; ++i)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         StableSum<R> tmp1, tmp2;

         for(j = lbeg[i + 1]; j > k; --j)
         {
            assert(*idx >= 0 && *idx < thedim);
            tmp1 += vec[*idx] * (*val);
            tmp2 += vec2[*idx++] * (*val++);
         }

         x = R(tmp1);
         x2 = R(tmp2);

         j = lrow[i];

         if(isNotZero(x, eps))
            updateSolutionVectorLright(x, j, vec[j], ridx, rn);

         if(isNotZero(x2, eps2))
            updateSolutionVectorLright(x2, j, vec2[j], ridx2, rn2);
      }
   }
}

// solve with L for three right hand sides
// see above methods for documentation
template <class R>
void CLUFactor<R>::vSolveLright3(
   R* vec, int* ridx, int& rn, R eps,
   R* vec2, int* ridx2, int& rn2, R eps2,
   R* vec3, int* ridx3, int& rn3, R eps3)
{
   int i, j, k, n;
   int end;
   R x, x2, x3;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;

   for(i = 0; i < end; ++i)
   {
      j = lrow[i];
      x = vec[j];
      x2 = vec2[j];
      x3 = vec3[j];

      if(isNotZero(x, eps))
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         if(isNotZero(x2, eps2))
         {
            if(isNotZero(x3, eps3))
            {
               // case 1: all three vectors are nonzero at j
               for(j = lbeg[i + 1]; j > k; --j)
               {
                  assert(*idx >= 0 && *idx < thedim);
                  n = *idx++;
                  updateSolutionVectorLright(x * (*val), n, vec[n], ridx, rn);
                  updateSolutionVectorLright(x2 * (*val), n, vec2[n], ridx2, rn2);
                  updateSolutionVectorLright(x3 * (*val), n, vec3[n], ridx3, rn3);
                  ++val;
               }
            }
            else
            {
               // case 2: 1 and 2 are nonzero at j
               for(j = lbeg[i + 1]; j > k; --j)
               {
                  assert(*idx >= 0 && *idx < thedim);
                  n = *idx++;
                  updateSolutionVectorLright(x * (*val), n, vec[n], ridx, rn);
                  updateSolutionVectorLright(x2 * (*val), n, vec2[n], ridx2, rn2);
                  ++val;
               }
            }
         }
         else if(isNotZero(x3, eps3))
         {
            // case 3: 1 and 3 are nonzero at j
            for(j = lbeg[i + 1]; j > k; --j)
            {
               assert(*idx >= 0 && *idx < thedim);
               n = *idx++;
               updateSolutionVectorLright(x * (*val), n, vec[n], ridx, rn);
               updateSolutionVectorLright(x3 * (*val), n, vec3[n], ridx3, rn3);
               ++val;
            }
         }
         else
         {
            // case 4: only 1 is nonzero at j
            for(j = lbeg[i + 1]; j > k; --j)
            {
               assert(*idx >= 0 && *idx < thedim);
               n = *idx++;
               updateSolutionVectorLright(x * (*val), n, vec[n], ridx, rn);
               ++val;
            }
         }
      }
      else if(isNotZero(x2, eps2))
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         if(isNotZero(x3, eps3))
         {
            // case 5: 2 and 3 are nonzero at j
            for(j = lbeg[i + 1]; j > k; --j)
            {
               assert(*idx >= 0 && *idx < thedim);
               n = *idx++;
               updateSolutionVectorLright(x2 * (*val), n, vec2[n], ridx2, rn2);
               updateSolutionVectorLright(x3 * (*val), n, vec3[n], ridx3, rn3);
               ++val;
            }
         }
         else
         {
            // case 6: only 2 is nonzero at j
            for(j = lbeg[i + 1]; j > k; --j)
            {
               assert(*idx >= 0 && *idx < thedim);
               n = *idx++;
               updateSolutionVectorLright(x2 * (*val), n, vec2[n], ridx2, rn2);
               ++val;
            }
         }
      }
      else if(isNotZero(x3, eps3))
      {
         // case 7: only 3 is nonzero at j
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
         {
            assert(*idx >= 0 && *idx < thedim);
            n = *idx++;
            updateSolutionVectorLright(x3 * (*val), n, vec3[n], ridx3, rn3);
            ++val;
         }
      }
   }

   if(l.updateType)                      /* Forest-Tomlin Updates */
   {
      end = l.firstUnused;

      for(; i < end; ++i)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);

         StableSum<R> tmp1, tmp2, tmp3;

         for(j = lbeg[i + 1]; j > k; --j)
         {
            assert(*idx >= 0 && *idx < thedim);
            tmp1 += vec[*idx] * (*val);
            tmp2 += vec2[*idx] * (*val);
            tmp3 += vec3[*idx++] * (*val++);
         }

         x = R(tmp1);
         x2 = R(tmp2);
         x3 = R(tmp3);

         j = lrow[i];

         if(isNotZero(x, eps))
            updateSolutionVectorLright(x, j, vec[j], ridx, rn);

         if(isNotZero(x2, eps2))
            updateSolutionVectorLright(x2, j, vec2[j], ridx2, rn2);

         if(isNotZero(x3, eps3))
            updateSolutionVectorLright(x3, j, vec3[j], ridx3, rn3);
      }
   }
}

template <class R>
int CLUFactor<R>::vSolveUright(R* vec, int* vidx,
                               R* rhs, int* ridx, int rn, R eps)
{
   int i, j, k, r, c, n;
   int* rorig, *corig;
   int* rperm;
   int* cidx, *clen, *cbeg;
   R* cval;
   R x, y;

   int* idx;
   R* val;

   rorig = row.orig;
   corig = col.orig;
   rperm = row.perm;

   cidx = u.col.idx;
   cval = u.col.val.data();
   clen = u.col.len;
   cbeg = u.col.start;

   n = 0;

   while(rn > 0)
   {
      /*      Find nonzero with highest permuted row index and setup i and r
       */
      i = deQueueMax(ridx, &rn);
      assert(i >= 0 && i < thedim);
      r = rorig[i];
      assert(r >= 0 && r < thedim);

      x = diag[r] * rhs[r];
      rhs[r] = 0;

      if(isNotZero(x, eps))
      {
         c = corig[i];
         assert(c >= 0 && c < thedim);
         vidx[n++] = c;
         vec[c] = x;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         while(j-- > 0)
         {
            assert(*idx >= 0 && *idx < thedim);
            k = *idx++;
            assert(k >= 0 && k < thedim);
            y = rhs[k];

            if(y == 0)
            {
               y = -x * (*val++);

               if(isNotZero(y, eps))
               {
                  rhs[k] = y;
                  enQueueMax(ridx, &rn, rperm[k]);
               }
            }
            else
            {
               y -= x * (*val++);
               y += (y == 0) ? SOPLEX_FACTOR_MARKER : 0;
               rhs[k] = y;
            }
         }

         if(rn > i * verySparseFactor4right)
         {
            /* continue with dense case */
            for(i = *ridx; i >= 0; --i)
            {
               r = rorig[i];
               assert(r >= 0 && r < thedim);
               x = diag[r] * rhs[r];
               rhs[r] = 0;

               if(isNotZero(x, eps))
               {
                  c = corig[i];
                  assert(c >= 0 && c < thedim);
                  vidx[n++] = c;
                  vec[c] = x;
                  val = &cval[cbeg[c]];
                  idx = &cidx[cbeg[c]];
                  j = clen[c];

                  while(j-- > 0)
                  {
                     assert(*idx >= 0 && *idx < thedim);
                     rhs[*idx++] -= x * (*val++);
                  }
               }
            }

            break;
         }
      }
   }

   return n;
}

template <class R>
void CLUFactor<R>::vSolveUrightNoNZ(R* vec,
                                    R* rhs, int* ridx, int rn, R eps)
{
   int i, j, k, r, c;
   int* rorig, *corig;
   int* rperm;
   int* cidx, *clen, *cbeg;
   R* cval;
   R x, y;

   int* idx;
   R* val;

   rorig = row.orig;
   corig = col.orig;
   rperm = row.perm;

   cidx = u.col.idx;
   cval = u.col.val.data();
   clen = u.col.len;
   cbeg = u.col.start;

   while(rn > 0)
   {
      if(rn > *ridx * verySparseFactor4right)
      {
         /* continue with dense case */
         for(i = *ridx; i >= 0; --i)
         {
            assert(i >= 0 && i < thedim);
            r = rorig[i];
            assert(r >= 0 && r < thedim);
            x = diag[r] * rhs[r];
            rhs[r] = 0;

            if(isNotZero(x, eps))
            {
               c = corig[i];
               vec[c] = x;
               val = &cval[cbeg[c]];
               idx = &cidx[cbeg[c]];
               j = clen[c];

               while(j-- > 0)
               {
                  assert(*idx >= 0 && *idx < thedim);
                  rhs[*idx++] -= x * (*val++);
               }
            }
         }

         break;
      }

      /*      Find nonzero with highest permuted row index and setup i and r
       */
      i = deQueueMax(ridx, &rn);

      assert(i >= 0 && i < thedim);

      r = rorig[i];

      assert(r >= 0 && r < thedim);

      x = diag[r] * rhs[r];

      rhs[r] = 0;

      if(isNotZero(x, eps))
      {
         c = corig[i];
         vec[c] = x;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         while(j-- > 0)
         {
            k = *idx++;
            assert(k >= 0 && k < thedim);
            y = rhs[k];

            if(y == 0)
            {
               y = -x * (*val++);

               if(isNotZero(y, eps))
               {
                  rhs[k] = y;
                  enQueueMax(ridx, &rn, rperm[k]);
               }
            }
            else
            {
               y -= x * (*val++);
               y += (y == 0) ? SOPLEX_FACTOR_MARKER : 0;
               rhs[k] = y;
            }
         }
      }
   }
}


template <class R>
int CLUFactor<R>::vSolveUright2(
   R* vec, int* vidx, R* rhs, int* ridx, int rn, R eps,
   R* vec2, R* rhs2, int* ridx2, int rn2, R eps2)
{
   int i, j, k, r, c, n;
   int* rorig, *corig;
   int* rperm;
   int* cidx, *clen, *cbeg;
   R* cval;
   R x, y;
   R x2, y2;

   int* idx;
   R* val;

   rorig = row.orig;
   corig = col.orig;
   rperm = row.perm;

   cidx = u.col.idx;
   cval = u.col.val.data();
   clen = u.col.len;
   cbeg = u.col.start;

   n = 0;

   while(rn + rn2 > 0)
   {
      /*      Find nonzero with highest permuted row index and setup i and r
       */
      if(rn <= 0)
         i = deQueueMax(ridx2, &rn2);
      else if(rn2 <= 0)
         i = deQueueMax(ridx, &rn);
      else if(*ridx2 > *ridx)
         i = deQueueMax(ridx2, &rn2);
      else if(*ridx2 < *ridx)
         i = deQueueMax(ridx, &rn);
      else
      {
         (void) deQueueMax(ridx, &rn);
         i = deQueueMax(ridx2, &rn2);
      }

      assert(i >= 0 && i < thedim);

      r = rorig[i];
      assert(r >= 0 && r < thedim);

      x = diag[r] * rhs[r];
      x2 = diag[r] * rhs2[r];
      rhs[r] = 0;
      rhs2[r] = 0;

      if(isNotZero(x, eps))
      {
         c = corig[i];
         vidx[n++] = c;
         vec[c] = x;
         vec2[c] = x2;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         if(isNotZero(x2, eps2))
         {
            while(j-- > 0)
            {
               k = *idx++;
               assert(k >= 0 && k < thedim);
               y2 = rhs2[k];

               if(y2 == 0)
               {
                  y2 = -x2 * (*val);

                  if(isNotZero(y2, eps2))
                  {
                     rhs2[k] = y2;
                     enQueueMax(ridx2, &rn2, rperm[k]);
                  }
               }
               else
               {
                  y2 -= x2 * (*val);
                  rhs2[k] = (y2 != 0) ? y2 : SOPLEX_FACTOR_MARKER;
               }

               y = rhs[k];

               if(y == 0)
               {
                  y = -x * (*val++);

                  if(isNotZero(y, eps))
                  {
                     rhs[k] = y;
                     enQueueMax(ridx, &rn, rperm[k]);
                  }
               }
               else
               {
                  y -= x * (*val++);
                  y += (y == 0) ? SOPLEX_FACTOR_MARKER : 0;
                  rhs[k] = y;
               }
            }
         }
         else
         {
            while(j-- > 0)
            {
               k = *idx++;
               assert(k >= 0 && k < thedim);
               y = rhs[k];

               if(y == 0)
               {
                  y = -x * (*val++);

                  if(isNotZero(y, eps))
                  {
                     rhs[k] = y;
                     enQueueMax(ridx, &rn, rperm[k]);
                  }
               }
               else
               {
                  y -= x * (*val++);
                  y += (y == 0) ? SOPLEX_FACTOR_MARKER : 0;
                  rhs[k] = y;
               }
            }
         }
      }
      else if(isNotZero(x2, eps2))
      {
         c = corig[i];
         assert(c >= 0 && c < thedim);
         vec2[c] = x2;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];

         while(j-- > 0)
         {
            k = *idx++;
            assert(k >= 0 && k < thedim);
            y2 = rhs2[k];

            if(y2 == 0)
            {
               y2 = -x2 * (*val++);

               if(isNotZero(y2, eps2))
               {
                  rhs2[k] = y2;
                  enQueueMax(ridx2, &rn2, rperm[k]);
               }
            }
            else
            {
               y2 -= x2 * (*val++);
               rhs2[k] = (y2 != 0) ? y2 : SOPLEX_FACTOR_MARKER;
            }
         }
      }

      if(rn + rn2 > i * verySparseFactor4right)
      {
         /* continue with dense case */
         if(*ridx > *ridx2)
            i = *ridx;
         else
            i = *ridx2;

         for(; i >= 0; --i)
         {
            assert(i < thedim);
            r = rorig[i];
            assert(r >= 0 && r < thedim);
            x = diag[r] * rhs[r];
            x2 = diag[r] * rhs2[r];
            rhs[r] = 0;
            rhs2[r] = 0;

            if(isNotZero(x2, eps2))
            {
               c = corig[i];
               assert(c >= 0 && c < thedim);
               vec2[c] = x2;
               val = &cval[cbeg[c]];
               idx = &cidx[cbeg[c]];
               j = clen[c];

               if(isNotZero(x, eps))
               {
                  vidx[n++] = c;
                  vec[c] = x;

                  while(j-- > 0)
                  {
                     assert(*idx >= 0 && *idx < thedim);
                     rhs[*idx] -= x * (*val);
                     rhs2[*idx++] -= x2 * (*val++);
                  }
               }
               else
               {
                  while(j-- > 0)
                  {
                     assert(*idx >= 0 && *idx < thedim);
                     rhs2[*idx++] -= x2 * (*val++);
                  }
               }
            }
            else if(isNotZero(x, eps))
            {
               c = corig[i];
               assert(c >= 0 && c < thedim);
               vidx[n++] = c;
               vec[c] = x;
               val = &cval[cbeg[c]];
               idx = &cidx[cbeg[c]];
               j = clen[c];

               while(j-- > 0)
               {
                  assert(*idx >= 0 && *idx < thedim);
                  rhs[*idx++] -= x * (*val++);
               }
            }
         }

         break;
      }
   }

   return n;
}

template <class R>
int CLUFactor<R>::vSolveUpdateRight(R* vec, int* ridx, int n, R eps)
{
   int i, j, k;
   int end;
   R x, y;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   assert(!l.updateType);               /* no Forest-Tomlin Updates */

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;
   end = l.firstUnused;

   for(i = l.firstUpdate; i < end; ++i)
   {
      assert(i >= 0 && i < thedim);
      x = vec[lrow[i]];

      if(isNotZero(x, eps))
      {
         k = lbeg[i];
         assert(k >= 0 && k < l.size);
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
         {
            int m = ridx[n] = *idx++;
            assert(m >= 0 && m < thedim);
            y = vec[m];
            n += (y == 0) ? 1 : 0;
            y = y - x * (*val++);
            vec[m] = (y != 0) ? y : SOPLEX_FACTOR_MARKER;
         }
      }
   }

   return n;
}

template <class R>
void CLUFactor<R>::vSolveUpdateRightNoNZ(R* vec, R /*eps*/)
{
   int i, j, k;
   int end;
   R x;
   R* lval, *val;
   int* lrow, *lidx, *idx;
   int* lbeg;

   assert(!l.updateType);               /* no Forest-Tomlin Updates */

   lval = l.val.data();
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;
   end = l.firstUnused;

   for(i = l.firstUpdate; i < end; ++i)
   {
      assert(i >= 0 && i < thedim);

      if((x = vec[lrow[i]]) != 0.0)
      {
         k = lbeg[i];
         assert(k >= 0 && k < l.size);
         idx = &(lidx[k]);
         val = &(lval[k]);

         for(j = lbeg[i + 1]; j > k; --j)
         {
            assert(*idx >= 0 && *idx < thedim);
            vec[*idx++] -= x * (*val++);
         }
      }
   }
}


template <class R>
int CLUFactor<R>::vSolveRight4update(R eps,
                                     R* vec, int* idx,                       /* result */
                                     R* rhs, int* ridx, int rn,              /* rhs    */
                                     R* forest, int* forestNum, int* forestIdx)
{
   vSolveLright(rhs, ridx, rn, eps);
   assert(rn >= 0 && rn <= thedim);

   /*  turn index list into a heap
    */

   if(forest)
   {
      R x;
      int i, j, k;
      int* rperm;
      int* it = forestIdx;

      rperm = row.perm;

      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(isNotZero(x, eps))
         {
            enQueueMax(ridx, &j, rperm[*it++ = k]);
            forest[k] = x;
         }
         else
            rhs[k] = 0;
      }

      *forestNum = rn = j;
   }
   else
   {
      R x;
      int i, j, k;
      int* rperm;

      rperm = row.perm;

      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(isNotZero(x, eps))
            enQueueMax(ridx, &j, rperm[k]);
         else
            rhs[k] = 0;
      }

      rn = j;
   }

   rn = vSolveUright(vec, idx, rhs, ridx, rn, eps);

   if(!l.updateType)             /* no Forest-Tomlin Updates */
      rn = vSolveUpdateRight(vec, idx, rn, eps);

   return rn;
}

template <class R>
int CLUFactor<R>::vSolveRight4update2(R eps,
                                      R* vec, int* idx,                  /* result1 */
                                      R* rhs, int* ridx, int rn,         /* rhs1    */
                                      R* vec2, R eps2,              /* result2 */
                                      R* rhs2, int* ridx2, int rn2,      /* rhs2    */
                                      R* forest, int* forestNum, int* forestIdx)
{
   vSolveLright2(rhs, ridx, rn, eps, rhs2, ridx2, rn2, eps2);
   assert(rn >= 0 && rn <= thedim);
   assert(rn2 >= 0 && rn2 <= thedim);

   /*  turn index list into a heap
    */

   if(forest)
   {
      R x;
      int i, j, k;
      int* rperm;
      int* it = forestIdx;

      rperm = row.perm;

      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(isNotZero(x, eps))
         {
            enQueueMax(ridx, &j, rperm[*it++ = k]);
            forest[k] = x;
         }
         else
            rhs[k] = 0;
      }

      *forestNum = rn = j;
   }
   else
   {
      R x;
      int i, j, k;
      int* rperm;

      rperm = row.perm;

      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(isNotZero(x, eps))
            enQueueMax(ridx, &j, rperm[k]);
         else
            rhs[k] = 0;
      }

      rn = j;
   }

   if(rn2 > thedim * verySparseFactor4right)
   {
      ridx2[0] = thedim - 1;
      /* ridx2[1] = thedim - 2; */
   }
   else
   {
      R x;
      /*      R  maxabs; */
      int i, j, k;
      int* rperm;

      /*      maxabs = 1;    */
      rperm = row.perm;

      for(i = j = 0; i < rn2; ++i)
      {
         k = ridx2[i];
         assert(k >= 0 && k < thedim);
         x = rhs2[k];

         if(x < -eps2)
         {
            /*              maxabs = (maxabs < -x) ? -x : maxabs;  */
            enQueueMax(ridx2, &j, rperm[k]);
         }
         else if(x > eps2)
         {
            /*              maxabs = (maxabs < x) ? x : maxabs;    */
            enQueueMax(ridx2, &j, rperm[k]);
         }
         else
            rhs2[k] = 0;
      }

      rn2 = j;

      /*      eps2 = maxabs * eps2;  */
   }

   rn = vSolveUright(vec, idx, rhs, ridx, rn, eps);

   vSolveUrightNoNZ(vec2, rhs2, ridx2, rn2, eps2);

   /*
    *  rn = vSolveUright2(vec, idx, rhs, ridx, rn, eps, vec2, rhs2, ridx2, rn2, eps2);
    */

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      rn = vSolveUpdateRight(vec, idx, rn, eps);
      vSolveUpdateRightNoNZ(vec2, eps2);
   }

   return rn;
}

template <class R>
void CLUFactor<R>::vSolveRight4update2sparse(R eps, R* vec, int* idx,         /* result1 */
      R* rhs, int* ridx, int& rn,        /* rhs1    */
      R eps2, R* vec2, int* idx2,     /* result2 */
      R* rhs2, int* ridx2, int& rn2,     /* rhs2    */
      R* forest, int* forestNum, int* forestIdx)
{
   /* solve with L */
   vSolveLright2(rhs, ridx, rn, eps, rhs2, ridx2, rn2, eps2);
   assert(rn >= 0 && rn <= thedim);
   assert(rn2 >= 0 && rn2 <= thedim);

   R x;
   int i, j, k;
   int* rperm = row.perm;

   /*  turn index list into a heap for both ridx and ridx2 */
   if(forest)
   {
      int* it = forestIdx;

      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(isNotZero(x, eps))
         {
            enQueueMax(ridx, &j, rperm[*it++ = k]);
            forest[k] = x;
         }
         else
            rhs[k] = 0;
      }

      *forestNum = rn = j;
   }
   else
   {
      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(isNotZero(x, eps))
            enQueueMax(ridx, &j, rperm[k]);
         else
            rhs[k] = 0;
      }

      rn = j;
   }

   for(i = j = 0; i < rn2; ++i)
   {
      k = ridx2[i];
      assert(k >= 0 && k < thedim);
      x = rhs2[k];

      if(isNotZero(x, eps2))
         enQueueMax(ridx2, &j, rperm[k]);
      else
         rhs2[k] = 0;
   }

   rn2 = j;

   /* solve with U */
   rn = vSolveUright(vec, idx, rhs, ridx, rn, eps);
   rn2 = vSolveUright(vec2, idx2, rhs2, ridx2, rn2, eps2);

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      rn = vSolveUpdateRight(vec, idx, rn, eps);
      rn2 = vSolveUpdateRight(vec2, idx2, rn2, eps2);
   }
}


template <class R>
int CLUFactor<R>::vSolveRight4update3(R eps,
                                      R* vec, int* idx,                 /* result1 */
                                      R* rhs, int* ridx, int rn,        /* rhs1    */
                                      R* vec2, R eps2,               /* result2 */
                                      R* rhs2, int* ridx2, int rn2,     /* rhs2    */
                                      R* vec3, R eps3,               /* result3 */
                                      R* rhs3, int* ridx3, int rn3,     /* rhs3    */
                                      R* forest, int* forestNum, int* forestIdx)
{

   vSolveLright3(rhs, ridx, rn, eps, rhs2, ridx2, rn2, eps2, rhs3, ridx3, rn3, eps3);
   assert(rn >= 0 && rn <= thedim);
   assert(rn2 >= 0 && rn2 <= thedim);
   assert(rn3 >= 0 && rn3 <= thedim);

   /*  turn index list into a heap
    */

   if(forest)
   {
      R x;
      int i, j, k;
      int* rperm;
      int* it = forestIdx;

      rperm = row.perm;

      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(isNotZero(x, eps))
         {
            enQueueMax(ridx, &j, rperm[*it++ = k]);
            forest[k] = x;
         }
         else
            rhs[k] = 0;
      }

      *forestNum = rn = j;
   }
   else
   {
      R x;
      int i, j, k;
      int* rperm;

      rperm = row.perm;

      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(isNotZero(x, eps))
            enQueueMax(ridx, &j, rperm[k]);
         else
            rhs[k] = 0;
      }

      rn = j;
   }

   if(rn2 > thedim * verySparseFactor4right)
   {
      ridx2[0] = thedim - 1;
   }
   else
   {
      R x;
      int i, j, k;
      int* rperm;

      rperm = row.perm;

      for(i = j = 0; i < rn2; ++i)
      {
         k = ridx2[i];
         assert(k >= 0 && k < thedim);
         x = rhs2[k];

         if(x < -eps2)
         {
            enQueueMax(ridx2, &j, rperm[k]);
         }
         else if(x > eps2)
         {
            enQueueMax(ridx2, &j, rperm[k]);
         }
         else
            rhs2[k] = 0;
      }

      rn2 = j;
   }

   if(rn3 > thedim * verySparseFactor4right)
   {
      ridx3[0] = thedim - 1;
   }
   else
   {
      R x;
      int i, j, k;
      int* rperm;

      rperm = row.perm;

      for(i = j = 0; i < rn3; ++i)
      {
         k = ridx3[i];
         assert(k >= 0 && k < thedim);
         x = rhs3[k];

         if(x < -eps3)
         {
            enQueueMax(ridx3, &j, rperm[k]);
         }
         else if(x > eps3)
         {
            enQueueMax(ridx3, &j, rperm[k]);
         }
         else
            rhs3[k] = 0;
      }

      rn3 = j;
   }

   rn = vSolveUright(vec, idx, rhs, ridx, rn, eps);

   vSolveUrightNoNZ(vec2, rhs2, ridx2, rn2, eps2);
   vSolveUrightNoNZ(vec3, rhs3, ridx3, rn3, eps3);

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      rn = vSolveUpdateRight(vec, idx, rn, eps);
      vSolveUpdateRightNoNZ(vec2, eps2);
      vSolveUpdateRightNoNZ(vec3, eps3);
   }

   return rn;
}

template <class R>
void CLUFactor<R>::vSolveRight4update3sparse(R eps, R* vec, int* idx,         /* result1 */
      R* rhs, int* ridx, int& rn,        /* rhs1    */
      R eps2, R* vec2, int* idx2,     /* result2 */
      R* rhs2, int* ridx2, int& rn2,     /* rhs2    */
      R eps3, R* vec3, int* idx3,     /* result3 */
      R* rhs3, int* ridx3, int& rn3,     /* rhs3    */
      R* forest, int* forestNum, int* forestIdx)
{
   vSolveLright3(rhs, ridx, rn, eps, rhs2, ridx2, rn2, eps2, rhs3, ridx3, rn3, eps3);
   assert(rn >= 0 && rn <= thedim);
   assert(rn2 >= 0 && rn2 <= thedim);
   assert(rn3 >= 0 && rn3 <= thedim);

   R x;
   int i, j, k;
   int* rperm = row.perm;

   /*  turn index list into a heap */
   if(forest)
   {
      int* it = forestIdx;

      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(isNotZero(x, eps))
         {
            enQueueMax(ridx, &j, rperm[*it++ = k]);
            forest[k] = x;
         }
         else
            rhs[k] = 0;
      }

      *forestNum = rn = j;
   }
   else
   {
      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(isNotZero(x, eps))
            enQueueMax(ridx, &j, rperm[k]);
         else
            rhs[k] = 0;
      }

      rn = j;
   }

   for(i = j = 0; i < rn2; ++i)
   {
      k = ridx2[i];
      assert(k >= 0 && k < thedim);
      x = rhs2[k];

      if(isNotZero(x, eps2))
         enQueueMax(ridx2, &j, rperm[k]);
      else
         rhs2[k] = 0;
   }

   rn2 = j;

   for(i = j = 0; i < rn3; ++i)
   {
      k = ridx3[i];
      assert(k >= 0 && k < thedim);
      x = rhs3[k];

      if(isNotZero(x, eps3))
         enQueueMax(ridx3, &j, rperm[k]);
      else
         rhs3[k] = 0;
   }

   rn3 = j;

   rn = vSolveUright(vec, idx, rhs, ridx, rn, eps);
   rn2 = vSolveUright(vec2, idx2, rhs2, ridx2, rn2, eps2);
   rn3 = vSolveUright(vec3, idx3, rhs3, ridx3, rn3, eps3);

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      rn = vSolveUpdateRight(vec, idx, rn, eps);
      rn2 = vSolveUpdateRight(vec2, idx2, rn2, eps2);
      rn3 = vSolveUpdateRight(vec3, idx3, rn3, eps3);
   }
}

template <class R>
void CLUFactor<R>::vSolveRightNoNZ(
   R* vec, R eps,             /* result */
   R* rhs, int* ridx, int rn)    /* rhs    */
{
   vSolveLright(rhs, ridx, rn, eps);
   assert(rn >= 0 && rn <= thedim);

   if(rn > thedim * verySparseFactor4right)
   {
      *ridx = thedim - 1;
   }
   else
   {
      R x;
      /*      R  maxabs; */
      int i, j, k;
      int* rperm;

      /*      maxabs = 1;    */
      rperm = row.perm;

      for(i = j = 0; i < rn; ++i)
      {
         k = ridx[i];
         assert(k >= 0 && k < thedim);
         x = rhs[k];

         if(x < -eps)
         {
            /*              maxabs = (maxabs < -x) ? -x : maxabs;  */
            enQueueMax(ridx, &j, rperm[k]);
         }
         else if(x > eps)
         {
            /*              maxabs = (maxabs < x) ? x : maxabs;    */
            enQueueMax(ridx, &j, rperm[k]);
         }
         else
            rhs[k] = 0;
      }

      rn = j;

      /*      eps2 = maxabs * eps2;  */
   }

   vSolveUrightNoNZ(vec, rhs, ridx, rn, eps);

   if(!l.updateType)             /* no Forest-Tomlin Updates */
      vSolveUpdateRightNoNZ(vec, eps);
}

template <class R>
int CLUFactor<R>::vSolveLeft(R eps,
                             R* vec, int* idx,                       /* result */
                             R* rhs, int* ridx, int rn)            /* rhs    */
{

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      rn = solveUpdateLeft(eps, rhs, ridx, rn);
      rn = solveUleft(eps, vec, idx, rhs, ridx, rn);
   }
   else
   {
      rn = solveUleft(eps, vec, idx, rhs, ridx, rn);
      rn = solveLleftForest(eps, vec, idx, rn);
   }

   // TODO verify the correctness of this check
   if(rn + l.firstUpdate > verySparseFactor4left * thedim)
   {
      // perform the dense solve
      solveLleftNoNZ(vec);
      // signal the caller that the nonzero pattern is lost
      return 0;
   }
   else
      return solveLleft(eps, vec, idx, rn);
}

template <class R>
int CLUFactor<R>::vSolveLeft2(R eps,
                              R* vec, int* idx,                      /* result */
                              R* rhs, int* ridx, int rn,             /* rhs    */
                              R* vec2,                               /* result2 */
                              R* rhs2, int* ridx2, int rn2)        /* rhs2    */
{

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      rn = solveUpdateLeft(eps, rhs, ridx, rn);
      rn = solveUleft(eps, vec, idx, rhs, ridx, rn);
      rn2 = solveUpdateLeft(eps, rhs2, ridx2, rn2);
      solveUleftNoNZ(eps, vec2, rhs2, ridx2, rn2);
   }
   else
   {
      rn = solveUleft(eps, vec, idx, rhs, ridx, rn);
      rn = solveLleftForest(eps, vec, idx, rn);
      solveUleftNoNZ(eps, vec2, rhs2, ridx2, rn2);
      solveLleftForestNoNZ(vec2);
   }

   rn = solveLleft(eps, vec, idx, rn);

   solveLleftNoNZ(vec2);

   return rn;
}

template <class R>
void CLUFactor<R>::vSolveLeft2sparse(R eps,
                                     R* vec, int* idx,                      /* result */
                                     R* rhs, int* ridx, int& rn,            /* rhs    */
                                     R* vec2, int* idx2,                    /* result2 */
                                     R* rhs2, int* ridx2, int& rn2)         /* rhs2    */
{
   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      rn = solveUpdateLeft(eps, rhs, ridx, rn);
      rn = solveUleft(eps, vec, idx, rhs, ridx, rn);
      rn2 = solveUpdateLeft(eps, rhs2, ridx2, rn2);
      rn2 = solveUleft(eps, vec2, idx2, rhs2, ridx2, rn2);
   }
   else
   {
      rn = solveUleft(eps, vec, idx, rhs, ridx, rn);
      rn = solveLleftForest(eps, vec, idx, rn);
      rn2 = solveUleft(eps, vec2, idx2, rhs2, ridx2, rn2);
      rn2 = solveLleftForest(eps, vec2, idx2, rn2);

   }

   rn = solveLleft(eps, vec, idx, rn);
   rn2 = solveLleft(eps, vec2, idx2, rn2);
}


template <class R>
int CLUFactor<R>::vSolveLeft3(R eps,
                              R* vec, int* idx,                      /* result */
                              R* rhs, int* ridx, int rn,             /* rhs    */
                              R* vec2,                               /* result2 */
                              R* rhs2, int* ridx2, int rn2,          /* rhs2    */
                              R* vec3,                               /* result3 */
                              R* rhs3, int* ridx3, int rn3)          /* rhs3    */
{

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      rn = solveUpdateLeft(eps, rhs, ridx, rn);
      rn = solveUleft(eps, vec, idx, rhs, ridx, rn);
      rn2 = solveUpdateLeft(eps, rhs2, ridx2, rn2);
      solveUleftNoNZ(eps, vec2, rhs2, ridx2, rn2);
      rn3 = solveUpdateLeft(eps, rhs3, ridx3, rn3);
      solveUleftNoNZ(eps, vec3, rhs3, ridx3, rn3);
   }
   else
   {
      rn = solveUleft(eps, vec, idx, rhs, ridx, rn);
      rn = solveLleftForest(eps, vec, idx, rn);
      solveUleftNoNZ(eps, vec2, rhs2, ridx2, rn2);
      solveLleftForestNoNZ(vec2);
      solveUleftNoNZ(eps, vec3, rhs3, ridx3, rn3);
      solveLleftForestNoNZ(vec3);
   }

   rn = solveLleft(eps, vec, idx, rn);

   solveLleftNoNZ(vec2);
   solveLleftNoNZ(vec3);

   return rn;
}

template <class R>
void CLUFactor<R>::vSolveLeft3sparse(R eps,
                                     R* vec, int* idx,                      /* result */
                                     R* rhs, int* ridx, int& rn,            /* rhs    */
                                     R* vec2, int* idx2,                    /* result2 */
                                     R* rhs2, int* ridx2, int& rn2,         /* rhs2    */
                                     R* vec3, int* idx3,                    /* result3 */
                                     R* rhs3, int* ridx3, int& rn3)         /* rhs3    */
{
   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      rn = solveUpdateLeft(eps, rhs, ridx, rn);
      rn = solveUleft(eps, vec, idx, rhs, ridx, rn);
      rn2 = solveUpdateLeft(eps, rhs2, ridx2, rn2);
      rn2 = solveUleft(eps, vec2, idx2, rhs2, ridx2, rn2);
      rn3 = solveUpdateLeft(eps, rhs3, ridx3, rn3);
      rn3 = solveUleft(eps, vec3, idx3, rhs3, ridx3, rn3);
   }
   else
   {
      rn = solveUleft(eps, vec, idx, rhs, ridx, rn);
      rn = solveLleftForest(eps, vec, idx, rn);
      rn2 = solveUleft(eps, vec2, idx2, rhs2, ridx2, rn2);
      rn2 = solveLleftForest(eps, vec2, idx2, rn2);
      rn3 = solveUleft(eps, vec3, idx3, rhs3, ridx3, rn3);
      rn3 = solveLleftForest(eps, vec3, idx3, rn3);
   }

   rn = solveLleft(eps, vec, idx, rn);
   rn2 = solveLleft(eps, vec2, idx2, rn2);
   rn3 = solveLleft(eps, vec3, idx3, rn3);
}

template <class R>
void CLUFactor<R>::vSolveLeftNoNZ(R eps,
                                  R* vec2,                            /* result2 */
                                  R* rhs2, int* ridx2, int rn2)     /* rhs2    */
{

   if(!l.updateType)             /* no Forest-Tomlin Updates */
   {
      rn2 = solveUpdateLeft(eps, rhs2, ridx2, rn2);
      solveUleftNoNZ(eps, vec2, rhs2, ridx2, rn2);
   }
   else
   {
      solveUleftNoNZ(eps, vec2, rhs2, ridx2, rn2);
      solveLleftForestNoNZ(vec2);
   }

   solveLleftNoNZ(vec2);
}
} // namespace soplex
