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
#include <iostream>

#include "soplex/spxdefines.h"
#include "soplex/svset.h"
#include "soplex/sorter.h"

namespace soplex
{
#define EPS     1e-6
#define STABLE  1e-3    // the sparsest row/column may only have a pivot of size STABLE*maxEntry


template <class R>
bool SPxWeightST<R>::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS
   return rowWeight.isConsistent()
          && colWeight.isConsistent()
          && rowRight.isConsistent()
          && colUp.isConsistent();
   // && SPxStarter::isConsistent();   // not yet implemented
#else
   return true;
#endif
}

/* Generating the Starting Basis
   The generation of a starting basis works as follows: After setting up the
   preference arrays #weight# and #coWeight#, #Id#s are selected to be dual in
   a greedy manner.  Initially, the first #Id# is taken. Then the next #Id# is
   checked wheter its vector is linearly dependend of the vectors of the #Id#s
   allready selected.  If not, it is added to them. This is iterated until a
   full matrix has been constructed.

   Testing for linear independence is done very much along the lines of LU
   factorization. A vector is taken, and updated with all previous L-vectors.
   Then the maximal absolut element is selected as pivot element for computing
   the L-vector. This is stored for further processing.
*/

/*
  The following two functions set the status of |id| to primal or dual,
  respectively.
*/
template <class R>
void SPxWeightST<R>::setPrimalStatus(
   typename SPxBasisBase<R>::Desc& desc,
   const SPxSolverBase<R>& base,
   const SPxId& id)
{
   if(id.isSPxRowId())
   {
      int n = base.number(SPxRowId(id));

      if(base.rhs(n) >= R(infinity))
      {
         if(base.lhs(n) <= R(-infinity))
            desc.rowStatus(n) = SPxBasisBase<R>::Desc::P_FREE;
         else
            desc.rowStatus(n) = SPxBasisBase<R>::Desc::P_ON_LOWER;
      }
      else
      {
         if(base.lhs(n) <= R(-infinity))
            desc.rowStatus(n) = SPxBasisBase<R>::Desc::P_ON_UPPER;
         else if(base.lhs(n) >= base.rhs(n) - base.epsilon())
            desc.rowStatus(n) = SPxBasisBase<R>::Desc::P_FIXED;
         else if(rowRight[n])
            desc.rowStatus(n) = SPxBasisBase<R>::Desc::P_ON_UPPER;
         else
            desc.rowStatus(n) = SPxBasisBase<R>::Desc::P_ON_LOWER;
      }
   }
   else
   {
      int n = base.number(SPxColId(id));

      if(base.SPxLPBase<R>::upper(n) >= R(infinity))
      {
         if(base.SPxLPBase<R>::lower(n) <= R(-infinity))
            desc.colStatus(n) = SPxBasisBase<R>::Desc::P_FREE;
         else
            desc.colStatus(n) = SPxBasisBase<R>::Desc::P_ON_LOWER;
      }
      else
      {
         if(base.SPxLPBase<R>::lower(n) <= R(-infinity))
            desc.colStatus(n) = SPxBasisBase<R>::Desc::P_ON_UPPER;
         else if(base.SPxLPBase<R>::lower(n) >= base.SPxLPBase<R>::upper(n) - base.epsilon())
            desc.colStatus(n) = SPxBasisBase<R>::Desc::P_FIXED;
         else if(colUp[n])
            desc.colStatus(n) = SPxBasisBase<R>::Desc::P_ON_UPPER;
         else
            desc.colStatus(n) = SPxBasisBase<R>::Desc::P_ON_LOWER;
      }
   }
}

// ----------------------------------------------------------------
template <class R>
static void setDualStatus(
   typename SPxBasisBase<R>::Desc& desc,
   const SPxSolverBase<R>& base,
   const SPxId& id)
{
   if(id.isSPxRowId())
   {
      int n = base.number(SPxRowId(id));
      desc.rowStatus(n) = base.basis().dualRowStatus(n);
   }
   else
   {
      int n = base.number(SPxColId(id));
      desc.colStatus(n) = base.basis().dualColStatus(n);
   }
}
// ----------------------------------------------------------------

/// Compare class for row weights, used for sorting.
template <typename T>
struct Compare
{
public:
   /// constructor
   Compare() : weight(0) {}
   //   const SPxSolverBase* base;     ///< the solver
   const T*      weight;   ///< the weights to compare

   /// compares the weights
   T operator()(int i1, int i2) const
   {
      return weight[i1] - weight[i2];
   }
};

// ----------------------------------------------------------------
/**
   The following method initializes \p pref such that it contains the
   set of \ref soplex::SPxId "SPxIds" ordered following \p rowWeight and
   \p colWeight. For the sorting we take the following approach: first
   we sort the rows, then the columns.  Finally we perform a mergesort
   of both.
*/
template <class R>
static void initPrefs(
   DataArray < SPxId >&      pref,
   const SPxSolverBase<R>&       base,
   const Array<R>& rowWeight,
   const Array<R>& colWeight)
{
   DataArray<int> row(base.nRows());
   DataArray<int> col(base.nCols());
   int i;
   int j;
   int k;

   Compare<R> compare;
   //   compare.base = &base;

   for(i = 0; i < base.nRows(); ++i)
      row[i] = i;

   compare.weight = rowWeight.get_const_ptr();

   SPxQuicksort(row.get_ptr(), row.size(), compare); // Sort rows

   for(i = 0; i < base.nCols(); ++i)
      col[i] = i;

   compare.weight = colWeight.get_const_ptr();

   SPxQuicksort(col.get_ptr(), col.size(), compare); // Sort column

   i = 0;
   j = 0;
   k = 0;

   while(k < pref.size())            // merge sort
   {
      if(rowWeight[row[i]] < colWeight[col[j]])
      {
         pref[k++] = base.rId(row[i++]);

         if(i >= base.nRows())
            while(k < pref.size())
               pref[k++] = base.cId(col[j++]);
      }
      else
      {
         pref[k++] = base.cId(col[j++]);

         if(j >= base.nCols())
            while(k < pref.size())
               pref[k++] = base.rId(row[i++]);
      }
   }

   assert(i == base.nRows());
   assert(j == base.nCols());
}

// ----------------------------------------------------------------
template <class R>
void SPxWeightST<R>::generate(SPxSolverBase<R>& base)
{
   SPxId tmpId;

   forbidden.reSize(base.dim());
   rowWeight.reSize(base.nRows());
   colWeight.reSize(base.nCols());
   rowRight.reSize(base.nRows());
   colUp.reSize(base.nCols());

   if(base.rep() == SPxSolverBase<R>::COLUMN)
   {
      weight   = &colWeight;
      coWeight = &rowWeight;
   }
   else
   {
      weight   = &rowWeight;
      coWeight = &colWeight;
   }

   assert(weight->size()   == base.coDim());
   assert(coWeight->size() == base.dim());

   setupWeights(base);

   typename SPxBasisBase<R>::Desc desc(base);
   //   desc.reSize(base.nRows(), base.nCols());

   DataArray < SPxId > pref(base.nRows() + base.nCols());
   initPrefs(pref, base, rowWeight, colWeight);

   int i;
   int stepi;
   int j;
   int sel;

   for(i = 0; i < base.dim(); ++i)
      forbidden[i] = 0;

   if(base.rep() == SPxSolverBase<R>::COLUMN)
   {
      // in COLUMN rep we scan from beginning to end
      i      = 0;
      stepi = 1;
   }
   else
   {
      // in ROW rep we scan from end to beginning
      i      = pref.size() - 1;
      stepi = -1;
   }

   int  dim = base.dim();
   R maxEntry = 0;

   for(; i >= 0 && i < pref.size(); i += stepi)
   {
      tmpId              = pref[i];
      const SVectorBase<R>& vec = base.vector(tmpId);
      sel                = -1;

      // column or row singleton ?
      if(vec.size() == 1)
      {
         int idx = vec.index(0);

         if(forbidden[idx] < 2)
         {
            sel  = idx;
            dim += (forbidden[idx] > 0) ? 1 : 0;
         }
      }
      else
      {
         maxEntry = vec.maxAbs();

         // initialize the nonzero counter
         int minRowEntries = base.nRows();

         // find a stable index with a sparse row/column
         for(j = vec.size(); --j >= 0;)
         {
            R x = vec.value(j);
            int  k = vec.index(j);
            int  nRowEntries = base.coVector(k).size();

            if(!forbidden[k] && (spxAbs(x) > STABLE * maxEntry) && (nRowEntries < minRowEntries))
            {
               minRowEntries = nRowEntries;
               sel  = k;
            }
         }
      }

      // we found a valid index
      if(sel >= 0)
      {
         MSG_DEBUG(

            if(pref[i].type() == SPxId::ROW_ID)
            std::cout << "DWEIST01 r" << base.number(pref[i]);
            else
               std::cout << "DWEIST02 c" << base.number(pref[i]);
            )

               forbidden[sel] = 2;

         // put current column/row into basis
         if(base.rep() == SPxSolverBase<R>::COLUMN)
            setDualStatus(desc, base, pref[i]);
         else
            setPrimalStatus(desc, base, pref[i]);

         for(j = vec.size(); --j >= 0;)
         {
            R x = vec.value(j);
            int  k = vec.index(j);

            if(!forbidden[k] && (x > EPS * maxEntry || -x > EPS * maxEntry))
            {
               forbidden[k] = 1;
               --dim;
            }
         }

         if(--dim == 0)
         {
            //@ for(++i; i < pref.size(); ++i)
            if(base.rep() == SPxSolverBase<R>::COLUMN)
            {
               // set all remaining indeces to nonbasic status
               for(i += stepi; i >= 0 && i < pref.size(); i += stepi)
                  setPrimalStatus(desc, base, pref[i]);

               // fill up the basis wherever linear independence is assured
               for(i = forbidden.size(); --i >= 0;)
               {
                  if(forbidden[i] < 2)
                     setDualStatus(desc, base, base.coId(i));
               }
            }
            else
            {
               for(i += stepi; i >= 0 && i < pref.size(); i += stepi)
                  setDualStatus(desc, base, pref[i]);

               for(i = forbidden.size(); --i >= 0;)
               {
                  if(forbidden[i] < 2)
                     setPrimalStatus(desc, base, base.coId(i));
               }
            }

            break;
         }
      }
      // sel == -1
      else if(base.rep() == SPxSolverBase<R>::COLUMN)
         setPrimalStatus(desc, base, pref[i]);
      else
         setDualStatus(desc, base, pref[i]);

#ifndef NDEBUG
      {
         int n, m;

         for(n = 0, m = forbidden.size(); n < forbidden.size(); ++n)
            m -= (forbidden[n] != 0) ? 1 : 0;

         assert(m == dim);
      }
#endif  // NDEBUG
   }

   assert(dim == 0);

   base.loadBasis(desc);
#ifdef  TEST
   base.init();

   int changed = 0;
   const VectorBase<R>& pvec = base.pVec();

   for(i = pvec.dim() - 1; i >= 0; --i)
   {
      if(desc.colStatus(i) == SPxBasisBase<R>::Desc::P_ON_UPPER
            && base.lower(i) > R(-infinity) && pvec[i] > base.maxObj(i))
      {
         changed = 1;
         desc.colStatus(i) = SPxBasisBase<R>::Desc::P_ON_LOWER;
      }
      else if(desc.colStatus(i) == SPxBasisBase<R>::Desc::P_ON_LOWER
              && base.upper(i) < R(infinity) && pvec[i] < base.maxObj(i))
      {
         changed = 1;
         desc.colStatus(i) = SPxBasisBase<R>::Desc::P_ON_UPPER;
      }
   }

   if(changed)
   {
      std::cout << "changed basis\n";
      base.loadBasis(desc);
   }
   else
      std::cout << "nothing changed\n";

#endif  // TEST
}

// ----------------------------------------------------------------

/* Computation of Weights
 */
template <class R>
void SPxWeightST<R>::setupWeights(SPxSolverBase<R>& base)
{
   const VectorBase<R>& obj  = base.maxObj();
   const VectorBase<R>& low  = base.lower();
   const VectorBase<R>& up   = base.upper();
   const VectorBase<R>& rhs  = base.rhs();
   const VectorBase<R>& lhs  = base.lhs();
   int    i;

   R eps    = base.epsilon();
   R maxabs = 1.0;

   // find absolut biggest entry in bounds and left-/right hand side
   for(i = 0; i < base.nCols(); i++)
   {
      if((up[i] < R(infinity)) && (spxAbs(up[i]) > maxabs))
         maxabs = spxAbs(up[i]);

      if((low[i] > R(-infinity)) && (spxAbs(low[i]) > maxabs))
         maxabs = spxAbs(low[i]);
   }

   for(i = 0; i < base.nRows(); i++)
   {
      if((rhs[i] < R(infinity)) && (spxAbs(rhs[i]) > maxabs))
         maxabs = spxAbs(rhs[i]);

      if((lhs[i] > R(-infinity)) && (spxAbs(lhs[i]) > maxabs))
         maxabs = spxAbs(lhs[i]);
   }

   /**@todo The comments are wrong. The first is for dual simplex and
    *       the secound for primal one. Is anything else wrong?
    *       Also the values are nearly the same for both cases.
    *       Should this be ? Changed the values for
    *       r_fixed to 0 because of maros-r7. It is not clear why
    *       this makes a difference because all constraints in that
    *       instance are of equality type.
    *       Why is rowRight sometimes not set?
    */
   if(base.rep() * base.type() > 0)             // primal simplex
   {
      const R ax            = 1e-3 / obj.maxAbs();
      const R bx            = 1.0 / maxabs;
      const R nne           = ax / base.nRows();  // 1e-4 * ax;
      const R c_fixed       = 1e+5;
      const R r_fixed       = 0; // TK20010103: was 1e+4 (maros-r7)
      const R c_dbl_bounded = 1e+1;
      const R r_dbl_bounded = 0;
      const R c_bounded     = 1e+1;
      const R r_bounded     = 0;
      const R c_free        = -1e+4;
      const R r_free        = -1e+5;

      for(i = base.nCols() - 1; i >= 0; i--)
      {
         R n = nne * (base.colVector(i).size() - 1); // very small value that is zero for col singletons
         R x = ax * obj[i]; // this is at most 1e-3, probably a lot smaller
         R u = bx * up [i]; // this is at most 1, probably a lot smaller
         R l = bx * low[i]; // this is at most 1, probably a lot smaller

         if(up[i] < R(infinity))
         {
            if(spxAbs(low[i] - up[i]) < eps)
               colWeight[i] = c_fixed + n + spxAbs(x);
            else if(low[i] > R(-infinity))
            {
               colWeight[i] = c_dbl_bounded + l - u + n;

               l = spxAbs(l);
               u = spxAbs(u);

               if(u < l)
               {
                  colUp[i]      = true;
                  colWeight[i] += x;
               }
               else
               {
                  colUp[i]      = false;
                  colWeight[i] -= x;
               }
            }
            else
            {
               colWeight[i] = c_bounded - u + x + n;
               colUp[i]     = true;
            }
         }
         else
         {
            if(low[i] > R(-infinity))
            {
               colWeight[i] = c_bounded + l + n - x;
               colUp[i]     = false;
            }
            else
            {
               colWeight[i] = c_free + n - spxAbs(x);
            }
         }
      }

      for(i = base.nRows() - 1; i >= 0; i--)
      {
         if(rhs[i] < R(infinity))
         {
            if(spxAbs(lhs[i] - rhs[i]) < eps)
            {
               rowWeight[i] = r_fixed;
            }
            else if(lhs[i] > R(-infinity))
            {
               R u = bx * rhs[i];
               R l = bx * lhs[i];

               rowWeight[i] = r_dbl_bounded + l - u;
               rowRight[i]  = spxAbs(u) < spxAbs(l);
            }
            else
            {
               rowWeight[i] = r_bounded - bx * rhs[i];
               rowRight[i]  = true;
            }
         }
         else
         {
            if(lhs[i] > R(-infinity))
            {
               rowWeight[i] = r_bounded + bx * lhs[i];
               rowRight[i]  = false;
            }
            else
            {
               rowWeight[i] = r_free;
            }
         }
      }
   }
   else
   {
      assert(base.rep() * base.type() < 0);           // dual simplex

      const R ax            = 1.0  / obj.maxAbs();
      const R bx            = 1e-2 / maxabs;
      const R nne           = 1e-4 * bx;
      const R c_fixed       = 1e+5;
      const R r_fixed       = 1e+4;
      const R c_dbl_bounded = 1;
      const R r_dbl_bounded = 0;
      const R c_bounded     = 0;
      const R r_bounded     = 0;
      const R c_free        = -1e+4;
      const R r_free        = -1e+5;

      for(i = base.nCols() - 1; i >= 0; i--)
      {
         R n = nne * (base.colVector(i).size() - 1);
         R x = ax  * obj[i];
         R u = bx  * up [i];
         R l = bx  * low[i];

         if(up[i] < R(infinity))
         {
            if(spxAbs(low[i] - up[i]) < eps)
               colWeight[i] = c_fixed + n + spxAbs(x);
            else if(low[i] > R(-infinity))
            {
               if(x > 0)
               {
                  colWeight[i] = c_dbl_bounded + x - u + n;
                  colUp[i]     = true;
               }
               else
               {
                  colWeight[i] = c_dbl_bounded - x + l + n;
                  colUp[i]     = false;
               }
            }
            else
            {
               colWeight[i] = c_bounded - u + x + n;
               colUp[i]     = true;
            }
         }
         else
         {
            if(low[i] > R(-infinity))
            {
               colWeight[i] = c_bounded - x + l + n;
               colUp[i]     = false;
            }
            else
               colWeight[i] = c_free + n - spxAbs(x);
         }
      }

      for(i = base.nRows() - 1; i >= 0; i--)
      {
         const R len1 = 1; // (base.rowVector(i).length() + base.epsilon());
         R n    = 0;  // nne * (base.rowVector(i).size() - 1);
         R u    = bx * len1 * rhs[i];
         R l    = bx * len1 * lhs[i];
         R x    = ax * len1 * (obj * base.rowVector(i));

         if(rhs[i] < R(infinity))
         {
            if(spxAbs(lhs[i] - rhs[i]) < eps)
               rowWeight[i] = r_fixed + n + spxAbs(x);
            else if(lhs[i] > R(-infinity))
            {
               if(x > 0)
               {
                  rowWeight[i] = r_dbl_bounded + x - u + n;
                  rowRight[i]  = true;
               }
               else
               {
                  rowWeight[i] = r_dbl_bounded - x + l + n;
                  rowRight[i]  = false;
               }
            }
            else
            {
               rowWeight[i] = r_bounded - u + n + x;
               rowRight[i]  = true;
            }
         }
         else
         {
            if(lhs[i] > R(-infinity))
            {
               rowWeight[i] = r_bounded + l + n - x;
               rowRight[i]  = false;
            }
            else
            {
               rowWeight[i] = r_free + n - spxAbs(x);
            }
         }
      }
   }

   MSG_DEBUG(
   {
      for(i = 0; i < base.nCols(); i++)
         std::cout << "C i= " << i
                   << " up= " << colUp[i]
                   << " w= " << colWeight[i]
                   << std::endl;
      for(i = 0; i < base.nRows(); i++)
         std::cout << "R i= " << i
                   << " rr= " << rowRight[i]
                   << " w= " << rowWeight[i]
                   << std::endl;
   })
}
} // namespace soplex
