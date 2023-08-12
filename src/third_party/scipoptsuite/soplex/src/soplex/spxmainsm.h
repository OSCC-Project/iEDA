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

/**@file  spxmainsm.h
 * @brief General methods in LP preprocessing.
 */
#ifndef _SPXMAINSM_H_
#define _SPXMAINSM_H_

#include <assert.h>
#include <memory>

#include "soplex/spxdefines.h"
#include "soplex/spxsimplifier.h"
#include "soplex/array.h"
#include "soplex/exceptions.h"

namespace soplex
{
//---------------------------------------------------------------------
//  class SPxMainSM
//---------------------------------------------------------------------

/**@brief   LP simplifier for removing uneccessary row/columns.
   @ingroup Algo

   This #SPxSimplifier is mainly based on the paper "Presolving in
   linear programming" by E. Andersen and K. Andersen (Mathematical
   Programming, 1995).  It implements all proposed methods and some
   other preprocessing techniques for removing redundant rows and
   columns and bounds.  Also infeasibility and unboundedness may be
   detected.

   Removed are:
   - empty rows / columns
   - unconstraint rows
   - row singletons
   - forcing rows
   - zero objective column singletons
   - (implied) free column singletons
   - doubleton equations combined with a column singleton
   - (implicitly) fixed columns
   - redundant lhs / rhs
   - redundant variable bounds
   - variables that are free in one direction
   - (weakly) dominated columns
   - duplicate rows / columns
*/
template <class R>
class SPxMainSM : public SPxSimplifier<R>
{
private:
   //---------------------------------------------------------------------
   //  class PostsolveStep
   //---------------------------------------------------------------------

   /**@brief   Base class for postsolving operations.
      @ingroup Algo

      Class #PostStep is an abstract base class providing the
      interface for operations in the postsolving process.
   */
   class PostStep
   {
   private:
      /// name of the simplifier
      const char* m_name;
      /// number of cols
      int nCols;
      /// number of rows
      int nRows;

   public:
      /// constructor.
      PostStep(const char* p_name, int nR = 0, int nC = 0)
         : m_name(p_name)
         , nCols(nC)
         , nRows(nR)
      {}
      /// copy constructor.
      PostStep(const PostStep& old)
         : m_name(old.m_name)
         , nCols(old.nCols)
         , nRows(old.nRows)
      {}
      /// assignment operator
      PostStep& operator=(const PostStep& /*rhs*/)
      {
         return *this;
      }
      /// destructor.
      virtual ~PostStep()
      {
         m_name = 0;
      }
      /// get name of simplifying step.
      virtual const char* getName() const
      {
         return m_name;
      }
      /// clone function for polymorphism
      virtual PostStep* clone() const = 0;
      /// executes the postsolving.
      virtual void execute(
         VectorBase<R>& x,                                 //*< Primal solution VectorBase<R> */
         VectorBase<R>& y,                                 //*< Dual solution VectorBase<R> */
         VectorBase<R>& s,                                 //*< VectorBase<R> of slacks */
         VectorBase<R>& r,                                 //*< Reduced cost VectorBase<R> */
         DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,    //*< Basis status of column basis */
         DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis,    //*< Basis status of row basis */
         bool isOptimal
      ) const = 0;

      virtual bool checkBasisDim(DataArray<typename SPxSolverBase<R>::VarStatus> rows,
                                 DataArray<typename SPxSolverBase<R>::VarStatus> cols) const;

      static R eps()
      {
         return 1e-6;
      }
   };

   /**@brief   Postsolves row objectives.
      @ingroup Algo
   */
   class RowObjPS : public PostStep
   {
   private:
      int m_i; ///< row index
      int m_j; ///< slack column index

   public:
      ///
      RowObjPS(const SPxLPBase<R>& lp, int _i, int _j)
         : PostStep("RowObj", lp.nRows(), lp.nCols())
         , m_i(_i)
         , m_j(_j)
      {}
      /// copy constructor
      RowObjPS(const RowObjPS& old)
         : PostStep(old)
         , m_i(old.m_i)
         , m_j(old.m_j)
      {}
      /// assignment operator
      RowObjPS& operator=(const RowObjPS& rhs)
      {
         if(this != &rhs)
         {
            m_i = rhs.m_i;
            m_j = rhs.m_j;
         }

         return *this;
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         return new RowObjPS(*this);
      }
   };

   /**@brief   Postsolves unconstraint constraints.
      @ingroup Algo
   */
   class FreeConstraintPS : public PostStep
   {
   private:
      int m_i;
      int m_old_i;
      DSVectorBase<R>  m_row;
      R m_row_obj;

   public:
      ///
      FreeConstraintPS(const SPxLPBase<R>& lp, int _i)
         : PostStep("FreeConstraint", lp.nRows(), lp.nCols())
         , m_i(_i)
         , m_old_i(lp.nRows() - 1)
         , m_row(lp.rowVector(_i))
         , m_row_obj(lp.rowObj(_i))
      {}
      /// copy constructor
      FreeConstraintPS(const FreeConstraintPS& old)
         : PostStep(old)
         , m_i(old.m_i)
         , m_old_i(old.m_old_i)
         , m_row(old.m_row)
         , m_row_obj(old.m_row_obj)
      {}
      /// assignment operator
      FreeConstraintPS& operator=(const FreeConstraintPS& rhs)
      {
         if(this != &rhs)
         {
            m_i = rhs.m_i;
            m_old_i = rhs.m_old_i;
            m_row = rhs.m_row;
            m_row_obj = rhs.m_row_obj;
         }

         return *this;
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         return new FreeConstraintPS(*this);
      }
   };

   /**@brief   Postsolves empty constraints.
      @ingroup Algo
   */
   class EmptyConstraintPS : public PostStep
   {
   private:
      int m_i;
      int m_old_i;
      R m_row_obj;

   public:
      ///
      EmptyConstraintPS(const SPxLPBase<R>& lp, int _i)
         : PostStep("EmptyConstraint", lp.nRows(), lp.nCols())
         , m_i(_i)
         , m_old_i(lp.nRows() - 1)
         , m_row_obj(lp.rowObj(_i))
      {}
      /// copy constructor
      EmptyConstraintPS(const EmptyConstraintPS& old)
         : PostStep(old)
         , m_i(old.m_i)
         , m_old_i(old.m_old_i)
         , m_row_obj(old.m_row_obj)
      {}
      /// assignment operator
      EmptyConstraintPS& operator=(const EmptyConstraintPS& rhs)
      {
         if(this != &rhs)
         {
            m_i = rhs.m_i;
            m_old_i = rhs.m_old_i;
            m_row_obj = rhs.m_row_obj;
         }

         return *this;
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         return new EmptyConstraintPS(*this);
      }
   };

   /**@brief   Postsolves row singletons.
      @ingroup Algo
   */
   class RowSingletonPS : public PostStep
   {
   private:
      const int  m_i;
      const int  m_old_i;
      const int  m_j;
      const R m_lhs;
      const R m_rhs;
      const bool m_strictLo;
      const bool m_strictUp;
      const bool m_maxSense;
      const R m_obj;
      DSVectorBase<R>   m_col;
      const R m_newLo;
      const R m_newUp;
      const R m_oldLo;
      const R m_oldUp;
      const R m_row_obj;

   public:
      ///
      RowSingletonPS(const SPxLPBase<R>& lp, int _i, int _j, bool strictLo, bool strictUp,
                     R newLo, R newUp, R oldLo, R oldUp)
         : PostStep("RowSingleton", lp.nRows(), lp.nCols())
         , m_i(_i)
         , m_old_i(lp.nRows() - 1)
         , m_j(_j)
         , m_lhs(lp.lhs(_i))
         , m_rhs(lp.rhs(_i))
         , m_strictLo(strictLo)
         , m_strictUp(strictUp)
         , m_maxSense(lp.spxSense() == SPxLPBase<R>::MAXIMIZE)
         , m_obj(lp.spxSense() == SPxLPBase<R>::MINIMIZE ? lp.obj(_j) : -lp.obj(_j))
         , m_col(lp.colVector(_j))
         , m_newLo(newLo)
         , m_newUp(newUp)
         , m_oldLo(oldLo)
         , m_oldUp(oldUp)
         , m_row_obj(lp.rowObj(_i))
      {}
      /// copy constructor
      RowSingletonPS(const RowSingletonPS& old)
         : PostStep(old)
         , m_i(old.m_i)
         , m_old_i(old.m_old_i)
         , m_j(old.m_j)
         , m_lhs(old.m_lhs)
         , m_rhs(old.m_rhs)
         , m_strictLo(old.m_strictLo)
         , m_strictUp(old.m_strictUp)
         , m_maxSense(old.m_maxSense)
         , m_obj(old.m_obj)
         , m_col(old.m_col)
         , m_newLo(old.m_newLo)
         , m_newUp(old.m_newUp)
         , m_oldLo(old.m_oldLo)
         , m_oldUp(old.m_oldUp)
         , m_row_obj(old.m_row_obj)
      {}
      /// assignment operator
      RowSingletonPS& operator=(const RowSingletonPS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_col = rhs.m_col;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         return new RowSingletonPS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves forcing constraints.
      @ingroup Algo
   */
   class ForceConstraintPS : public PostStep
   {
   private:
      const int       m_i;
      const int       m_old_i;
      const R      m_lRhs;
      DSVectorBase<R>        m_row;
      Array<R> m_objs;
      DataArray<bool> m_fixed;
      Array<DSVectorBase<R>> m_cols;
      const bool      m_lhsFixed;
      const bool      m_maxSense;
      Array<R> m_oldLowers;
      Array<R> m_oldUppers;
      const R      m_lhs;
      const R      m_rhs;
      const R      m_rowobj;

   public:
      ///
      ForceConstraintPS(const SPxLPBase<R>& lp, int _i, bool lhsFixed, DataArray<bool>& fixCols,
                        Array<R>& lo, Array<R>& up)
         : PostStep("ForceConstraint", lp.nRows(), lp.nCols())
         , m_i(_i)
         , m_old_i(lp.nRows() - 1)
         , m_lRhs(lhsFixed ? lp.lhs(_i) : lp.rhs(_i))
         , m_row(lp.rowVector(_i))
         , m_objs(lp.rowVector(_i).size())
         , m_fixed(fixCols)
         , m_cols(lp.rowVector(_i).size())
         , m_lhsFixed(lhsFixed)
         , m_maxSense(lp.spxSense() == SPxLPBase<R>::MAXIMIZE)
         , m_oldLowers(lo)
         , m_oldUppers(up)
         , m_lhs(lp.lhs(_i))
         , m_rhs(lp.rhs(_i))
         , m_rowobj(lp.rowObj(_i))
      {
         for(int k = 0; k < m_row.size(); ++k)
         {
            m_objs[k] = (lp.spxSense() == SPxLPBase<R>::MINIMIZE ? lp.obj(m_row.index(k)) : -lp.obj(m_row.index(
                            k)));
            m_cols[k] = lp.colVector(m_row.index(k));
         }
      }
      /// copy constructor
      ForceConstraintPS(const ForceConstraintPS& old)
         : PostStep(old)
         , m_i(old.m_i)
         , m_old_i(old.m_old_i)
         , m_lRhs(old.m_lRhs)
         , m_row(old.m_row)
         , m_objs(old.m_objs)
         , m_fixed(old.m_fixed)
         , m_cols(old.m_cols)
         , m_lhsFixed(old.m_lhsFixed)
         , m_maxSense(old.m_maxSense)
         , m_oldLowers(old.m_oldLowers)
         , m_oldUppers(old.m_oldUppers)
         , m_lhs(old.m_lhs)
         , m_rhs(old.m_rhs)
         , m_rowobj(old.m_rowobj)
      {}
      /// assignment operator
      ForceConstraintPS& operator=(const ForceConstraintPS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_row = rhs.m_row;
            m_objs = rhs.m_objs;
            m_fixed = rhs.m_fixed;
            m_cols = rhs.m_cols;
            m_oldLowers = rhs.m_oldLowers;
            m_oldUppers = rhs.m_oldUppers;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         return new ForceConstraintPS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves variable fixing.
      @ingroup Algo
   */
   class FixVariablePS : public PostStep
   {
   private:
      const int  m_j;
      const int  m_old_j;
      const R m_val;
      const R m_obj;
      const R m_lower;
      const R m_upper;
      const bool m_correctIdx; /// does the index mapping have to be updated in postsolving?
      DSVectorBase<R>   m_col;

   public:
      ///
      FixVariablePS(const SPxLPBase<R>& lp, SPxMainSM& simplifier, int _j, const R val,
                    bool correctIdx = true)
         : PostStep("FixVariable", lp.nRows(), lp.nCols())
         , m_j(_j)
         , m_old_j(lp.nCols() - 1)
         , m_val(val)
         , m_obj(lp.spxSense() == SPxLPBase<R>::MINIMIZE ? lp.obj(_j) : -lp.obj(_j))
         , m_lower(lp.lower(_j))
         , m_upper(lp.upper(_j))
         , m_correctIdx(correctIdx)
         , m_col(lp.colVector(_j))
      {
         simplifier.addObjoffset(m_val * lp.obj(m_j));
      }
      /// copy constructor
      FixVariablePS(const FixVariablePS& old)
         : PostStep(old)
         , m_j(old.m_j)
         , m_old_j(old.m_old_j)
         , m_val(old.m_val)
         , m_obj(old.m_obj)
         , m_lower(old.m_lower)
         , m_upper(old.m_upper)
         , m_correctIdx(old.m_correctIdx)
         , m_col(old.m_col)
      {}
      /// assignment operator
      FixVariablePS& operator=(const FixVariablePS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_col = rhs.m_col;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         return new FixVariablePS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves variable bound fixing.
      @ingroup Algo
   */
   class FixBoundsPS : public PostStep
   {
   private:
      const int            m_j;
      typename SPxSolverBase<R>::VarStatus m_status;

   public:
      ///
      FixBoundsPS(const SPxLPBase<R>& lp, int j, R val)
         : PostStep("FixBounds", lp.nRows(), lp.nCols())
         , m_j(j)
      {
         if(EQrel(lp.lower(j), lp.upper(j), this->eps()))
            m_status = SPxSolverBase<R>::FIXED;
         else if(EQrel(val, lp.lower(j), this->eps()))
            m_status = SPxSolverBase<R>::ON_LOWER;
         else if(EQrel(val, lp.upper(j), this->eps()))
            m_status = SPxSolverBase<R>::ON_UPPER;
         else if(lp.lower(j) <= R(-infinity) && lp.upper(j) >= R(infinity))
            m_status = SPxSolverBase<R>::ZERO;
         else
         {
            throw SPxInternalCodeException("XMAISM14 This should never happen.");
         }
      }
      /// copy constructor
      FixBoundsPS(const FixBoundsPS& old)
         : PostStep(old)
         , m_j(old.m_j)
         , m_status(old.m_status)
      {}
      /// assignment operator
      FixBoundsPS& operator=(const FixBoundsPS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_status = rhs.m_status;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         FixBoundsPS* FixBoundsPSptr = 0;
         spx_alloc(FixBoundsPSptr);
         return new(FixBoundsPSptr) FixBoundsPS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief Postsolves the case when constraints are removed due to a
             variable with zero objective that is free in one direction.
      @ingroup Algo
   */
   class FreeZeroObjVariablePS : public PostStep
   {
   private:
      const int       m_j;
      const int       m_old_j;
      const int       m_old_i;
      const R      m_bnd;
      DSVectorBase<R>        m_col;
      DSVectorBase<R>        m_lRhs;
      DSVectorBase<R>        m_rowObj;
      Array<DSVectorBase<R>> m_rows;
      const bool      m_loFree;

   public:
      ///
      FreeZeroObjVariablePS(const SPxLPBase<R>& lp, int _j, bool loFree, DSVectorBase<R> col_idx_sorted)
         : PostStep("FreeZeroObjVariable", lp.nRows(), lp.nCols())
         , m_j(_j)
         , m_old_j(lp.nCols() - 1)
         , m_old_i(lp.nRows() - 1)
         , m_bnd(loFree ? lp.upper(_j) : lp.lower(_j))
         , m_col(col_idx_sorted)
         , m_lRhs(lp.colVector(_j).size())
         , m_rowObj(lp.colVector(_j).size())
         , m_rows(lp.colVector(_j).size())
         , m_loFree(loFree)
      {
         for(int k = 0; k < m_col.size(); ++k)
         {
            int r = m_col.index(k);

            if((m_loFree  && m_col.value(k) > 0) ||
                  (!m_loFree && m_col.value(k) < 0))
               m_lRhs.add(k, lp.rhs(r));
            else
               m_lRhs.add(k, lp.lhs(r));

            m_rows[k] = lp.rowVector(r);
            m_rowObj.add(k, lp.rowObj(r));
         }
      }
      /// copy constructor
      FreeZeroObjVariablePS(const FreeZeroObjVariablePS& old)
         : PostStep(old)
         , m_j(old.m_j)
         , m_old_j(old.m_old_j)
         , m_old_i(old.m_old_i)
         , m_bnd(old.m_bnd)
         , m_col(old.m_col)
         , m_lRhs(old.m_lRhs)
         , m_rowObj(old.m_rowObj)
         , m_rows(old.m_rows)
         , m_loFree(old.m_loFree)
      {}
      /// assignment operator
      FreeZeroObjVariablePS& operator=(const FreeZeroObjVariablePS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_col = rhs.m_col;
            m_lRhs = rhs.m_lRhs;
            m_rowObj = rhs.m_rowObj;
            m_rows = rhs.m_rows;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         FreeZeroObjVariablePS* FreeZeroObjVariablePSptr = 0;
         spx_alloc(FreeZeroObjVariablePSptr);
         return new(FreeZeroObjVariablePSptr) FreeZeroObjVariablePS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves column singletons with zero objective.
      @ingroup Algo
   */
   class ZeroObjColSingletonPS : public PostStep
   {
   private:
      const int  m_j;
      const int  m_i;
      const int  m_old_j;
      const R m_lhs;
      const R m_rhs;
      const R m_lower;
      const R m_upper;
      DSVectorBase<R>   m_row;

   public:
      ///
      ZeroObjColSingletonPS(const SPxLPBase<R>& lp, const SPxMainSM&, int _j, int _i)
         : PostStep("ZeroObjColSingleton", lp.nRows(), lp.nCols())
         , m_j(_j)
         , m_i(_i)
         , m_old_j(lp.nCols() - 1)
         , m_lhs(lp.lhs(_i))
         , m_rhs(lp.rhs(_i))
         , m_lower(lp.lower(_j))
         , m_upper(lp.upper(_j))
         , m_row(lp.rowVector(_i))
      {}
      /// copy constructor
      ZeroObjColSingletonPS(const ZeroObjColSingletonPS& old)
         : PostStep(old)
         , m_j(old.m_j)
         , m_i(old.m_i)
         , m_old_j(old.m_old_j)
         , m_lhs(old.m_lhs)
         , m_rhs(old.m_rhs)
         , m_lower(old.m_lower)
         , m_upper(old.m_upper)
         , m_row(old.m_row)
      {}
      /// assignment operator
      ZeroObjColSingletonPS& operator=(const ZeroObjColSingletonPS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_row = rhs.m_row;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         ZeroObjColSingletonPS* ZeroObjColSingletonPSptr = 0;
         spx_alloc(ZeroObjColSingletonPSptr);
         return new(ZeroObjColSingletonPSptr) ZeroObjColSingletonPS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves free column singletons.
      @ingroup Algo
   */
   class FreeColSingletonPS : public PostStep
   {
   private:
      const int  m_j;
      const int  m_i;
      const int  m_old_j;
      const int  m_old_i;
      const R m_obj;
      const R m_lRhs;
      const bool m_onLhs;
      const bool m_eqCons;
      DSVectorBase<R>   m_row;

   public:
      ///
      FreeColSingletonPS(const SPxLPBase<R>& lp, SPxMainSM& simplifier, int _j, int _i, R slackVal)
         : PostStep("FreeColSingleton", lp.nRows(), lp.nCols())
         , m_j(_j)
         , m_i(_i)
         , m_old_j(lp.nCols() - 1)
         , m_old_i(lp.nRows() - 1)
         , m_obj(lp.spxSense() == SPxLPBase<R>::MINIMIZE ? lp.obj(_j) : -lp.obj(_j))
         , m_lRhs(slackVal)
         , m_onLhs(EQ(slackVal, lp.lhs(_i)))
         , m_eqCons(EQ(lp.lhs(_i), lp.rhs(_i)))
         , m_row(lp.rowVector(_i))
      {
         assert(m_row[m_j] != 0.0);
         simplifier.addObjoffset(m_lRhs * (lp.obj(m_j) / m_row[m_j]));
      }
      /// copy constructor
      FreeColSingletonPS(const FreeColSingletonPS& old)
         : PostStep(old)
         , m_j(old.m_j)
         , m_i(old.m_i)
         , m_old_j(old.m_old_j)
         , m_old_i(old.m_old_i)
         , m_obj(old.m_obj)
         , m_lRhs(old.m_lRhs)
         , m_onLhs(old.m_onLhs)
         , m_eqCons(old.m_eqCons)
         , m_row(old.m_row)
      {}
      /// assignment operator
      FreeColSingletonPS& operator=(const FreeColSingletonPS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_row = rhs.m_row;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         FreeColSingletonPS* FreeColSingletonPSptr = 0;
         spx_alloc(FreeColSingletonPSptr);
         return new(FreeColSingletonPSptr) FreeColSingletonPS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves doubleton equations combined with a column singleton.
      @ingroup Algo
   */
   class DoubletonEquationPS : public PostStep
   {
   private:
      const int  m_j;
      const int  m_k;
      const int  m_i;
      const bool m_maxSense;
      const bool m_jFixed;
      const R m_jObj;
      const R m_kObj;
      const R m_aij;
      const bool m_strictLo;
      const bool m_strictUp;
      const R m_newLo;
      const R m_newUp;
      const R m_oldLo;
      const R m_oldUp;
      const R m_Lo_j;
      const R m_Up_j;
      const R m_lhs;
      const R m_rhs;
      DSVectorBase<R>   m_col;

   public:
      ///
      DoubletonEquationPS(const SPxLPBase<R>& lp, int _j, int _k, int _i, R oldLo, R oldUp)
         : PostStep("DoubletonEquation", lp.nRows(), lp.nCols())
         , m_j(_j)
         , m_k(_k)
         , m_i(_i)
         , m_maxSense(lp.spxSense() == SPxLPBase<R>::MAXIMIZE)
         , m_jFixed(EQ(lp.lower(_j), lp.upper(_j)))
         , m_jObj(lp.spxSense() == SPxLPBase<R>::MINIMIZE ? lp.obj(_j) : -lp.obj(_j))
         , m_kObj(lp.spxSense() == SPxLPBase<R>::MINIMIZE ? lp.obj(_k) : -lp.obj(_k))
         , m_aij(lp.colVector(_j).value(0))
         , m_strictLo(lp.lower(_k) > oldLo)
         , m_strictUp(lp.upper(_k) < oldUp)
         , m_newLo(lp.lower(_k))
         , m_newUp(lp.upper(_k))
         , m_oldLo(oldLo)
         , m_oldUp(oldUp)
         , m_Lo_j(lp.lower(_j))
         , m_Up_j(lp.upper(_j))
         , m_lhs(lp.lhs(_i))
         , m_rhs(lp.rhs(_i))
         , m_col(lp.colVector(_k))
      {}
      /// copy constructor
      DoubletonEquationPS(const DoubletonEquationPS& old)
         : PostStep(old)
         , m_j(old.m_j)
         , m_k(old.m_k)
         , m_i(old.m_i)
         , m_maxSense(old.m_maxSense)
         , m_jFixed(old.m_jFixed)
         , m_jObj(old.m_jObj)
         , m_kObj(old.m_kObj)
         , m_aij(old.m_aij)
         , m_strictLo(old.m_strictLo)
         , m_strictUp(old.m_strictUp)
         , m_newLo(old.m_newLo)
         , m_newUp(old.m_newUp)
         , m_oldLo(old.m_oldLo)
         , m_oldUp(old.m_oldUp)
         , m_Lo_j(old.m_Lo_j)
         , m_Up_j(old.m_Up_j)
         , m_lhs(old.m_lhs)
         , m_rhs(old.m_rhs)
         , m_col(old.m_col)
      {}
      /// assignment operator
      DoubletonEquationPS& operator=(const DoubletonEquationPS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_col = rhs.m_col;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         DoubletonEquationPS* DoubletonEquationPSptr = 0;
         spx_alloc(DoubletonEquationPSptr);
         return new(DoubletonEquationPSptr) DoubletonEquationPS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves duplicate rows.
      @ingroup Algo
   */
   class DuplicateRowsPS : public PostStep
   {
   private:
      const int       m_i;
      const R      m_i_rowObj;
      const int       m_maxLhsIdx;
      const int       m_minRhsIdx;
      const bool      m_maxSense;
      const bool      m_isFirst;
      const bool      m_isLast;
      const bool      m_fixed;
      const int       m_nCols;
      DSVectorBase<R>        m_scale;
      DSVectorBase<R>        m_rowObj;
      DataArray<int>  m_rIdxLocalOld;
      DataArray<int>  m_perm;
      DataArray<bool> m_isLhsEqualRhs;

   public:
      DuplicateRowsPS(const SPxLPBase<R>& lp, int _i,
                      int maxLhsIdx, int minRhsIdx, const DSVectorBase<R>& dupRows,
                      const Array<R> scale, const DataArray<int> perm, const DataArray<bool> isLhsEqualRhs,
                      bool isTheLast, bool isFixedRow, bool isFirst = false)
         : PostStep("DuplicateRows", lp.nRows(), lp.nCols())
         , m_i(_i)
         , m_i_rowObj(lp.rowObj(_i))
         , m_maxLhsIdx((maxLhsIdx == -1) ? -1 : maxLhsIdx)
         , m_minRhsIdx((minRhsIdx == -1) ? -1 : minRhsIdx)
         , m_maxSense(lp.spxSense() == SPxLPBase<R>::MAXIMIZE)
         , m_isFirst(isFirst)
         , m_isLast(isTheLast)
         , m_fixed(isFixedRow)
         , m_nCols(lp.nCols())
         , m_scale(dupRows.size())
         , m_rowObj(dupRows.size())
         , m_rIdxLocalOld(dupRows.size())
         , m_perm(perm)
         , m_isLhsEqualRhs(isLhsEqualRhs)
      {
         R rowScale = scale[_i];

         for(int k = 0; k < dupRows.size(); ++k)
         {
            m_scale.add(dupRows.index(k), rowScale / scale[dupRows.index(k)]);
            m_rowObj.add(dupRows.index(k), lp.rowObj(dupRows.index(k)));
            m_rIdxLocalOld[k] = dupRows.index(k);
         }
      }
      /// copy constructor
      DuplicateRowsPS(const DuplicateRowsPS& old)
         : PostStep(old)
         , m_i(old.m_i)
         , m_i_rowObj(old.m_i_rowObj)
         , m_maxLhsIdx(old.m_maxLhsIdx)
         , m_minRhsIdx(old.m_minRhsIdx)
         , m_maxSense(old.m_maxSense)
         , m_isFirst(old.m_isFirst)
         , m_isLast(old.m_isLast)
         , m_fixed(old.m_fixed)
         , m_nCols(old.m_nCols)
         , m_scale(old.m_scale)
         , m_rowObj(old.m_rowObj)
         , m_rIdxLocalOld(old.m_rIdxLocalOld)
         , m_perm(old.m_perm)
         , m_isLhsEqualRhs(old.m_isLhsEqualRhs)
      {}
      /// assignment operator
      DuplicateRowsPS& operator=(const DuplicateRowsPS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_scale = rhs.m_scale;
            m_rowObj = rhs.m_rowObj;
            m_rIdxLocalOld = rhs.m_rIdxLocalOld;
            m_perm = rhs.m_perm;
            m_isLhsEqualRhs = rhs.m_isLhsEqualRhs;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         DuplicateRowsPS* DuplicateRowsPSptr = 0;
         spx_alloc(DuplicateRowsPSptr);
         return new(DuplicateRowsPSptr) DuplicateRowsPS(*this);
      }
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves duplicate columns.
      @ingroup Algo
   */
   class DuplicateColsPS : public PostStep
   {
   private:
      const int            m_j;
      const int            m_k;
      const R           m_loJ;
      const R           m_upJ;
      const R           m_loK;
      const R           m_upK;
      const R           m_scale;
      const bool           m_isFirst;
      const bool           m_isLast;
      DataArray<int>       m_perm;

   public:
      DuplicateColsPS(const SPxLPBase<R>& lp, int _j, int _k, R scale, DataArray<int>  perm,
                      bool isFirst = false, bool isTheLast = false)
         : PostStep("DuplicateCols", lp.nRows(), lp.nCols())
         , m_j(_j)
         , m_k(_k)
         , m_loJ(lp.lower(_j))
         , m_upJ(lp.upper(_j))
         , m_loK(lp.lower(_k))
         , m_upK(lp.upper(_k))
         , m_scale(scale)
         , m_isFirst(isFirst)
         , m_isLast(isTheLast)
         , m_perm(perm)
      {}
      /// copy constructor
      DuplicateColsPS(const DuplicateColsPS& old)
         : PostStep(old)
         , m_j(old.m_j)
         , m_k(old.m_k)
         , m_loJ(old.m_loJ)
         , m_upJ(old.m_upJ)
         , m_loK(old.m_loK)
         , m_upK(old.m_upK)
         , m_scale(old.m_scale)
         , m_isFirst(old.m_isFirst)
         , m_isLast(old.m_isLast)
         , m_perm(old.m_perm)
      {}
      /// assignment operator
      DuplicateColsPS& operator=(const DuplicateColsPS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         DuplicateColsPS* DuplicateColsPSptr = 0;
         spx_alloc(DuplicateColsPSptr);
         return new(DuplicateColsPSptr) DuplicateColsPS(*this);
      }
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves aggregation.
      @ingroup Algo
   */
   class AggregationPS : public PostStep
   {
   private:
      const int  m_j;
      const int  m_i;
      const int  m_old_j;
      const int  m_old_i;
      const R m_upper;
      const R m_lower;
      const R m_obj;
      const R m_oldupper;
      const R m_oldlower;
      const R m_rhs;
      DSVectorBase<R>   m_row;
      DSVectorBase<R>   m_col;

   public:
      ///
      AggregationPS(const SPxLPBase<R>& lp, int _i, int _j, R rhs, R oldupper, R oldlower)
         : PostStep("Aggregation", lp.nRows(), lp.nCols())
         , m_j(_j)
         , m_i(_i)
         , m_old_j(lp.nCols() - 1)
         , m_old_i(lp.nRows() - 1)
         , m_upper(lp.upper(_j))
         , m_lower(lp.lower(_j))
         , m_obj(lp.spxSense() == SPxLPBase<R>::MINIMIZE ? lp.obj(_j) : -lp.obj(_j))
         , m_oldupper(oldupper)
         , m_oldlower(oldlower)
         , m_rhs(rhs)
         , m_row(lp.rowVector(_i))
         , m_col(lp.colVector(_j))
      {
         assert(m_row[m_j] != 0.0);
      }
      /// copy constructor
      AggregationPS(const AggregationPS& old)
         : PostStep(old)
         , m_j(old.m_j)
         , m_i(old.m_i)
         , m_old_j(old.m_old_j)
         , m_old_i(old.m_old_i)
         , m_upper(old.m_upper)
         , m_lower(old.m_lower)
         , m_obj(old.m_obj)
         , m_oldupper(old.m_oldupper)
         , m_oldlower(old.m_oldlower)
         , m_rhs(old.m_rhs)
         , m_row(old.m_row)
         , m_col(old.m_col)
      {}
      /// assignment operator
      AggregationPS& operator=(const AggregationPS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_row = rhs.m_row;
            m_col = rhs.m_col;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         AggregationPS* AggregationPSptr = 0;
         spx_alloc(AggregationPSptr);
         return new(AggregationPSptr) AggregationPS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves multi aggregation.
      @ingroup Algo
   */
   class MultiAggregationPS : public PostStep
   {
   private:
      const int  m_j;
      const int  m_i;
      const int  m_old_j;
      const int  m_old_i;
      const R m_upper;
      const R m_lower;
      const R m_obj;
      const R m_const;
      const bool m_onLhs;
      const bool m_eqCons;
      DSVectorBase<R>   m_row;
      DSVectorBase<R>   m_col;

   public:
      ///
      MultiAggregationPS(const SPxLPBase<R>& lp, SPxMainSM& simplifier, int _i, int _j, R constant)
         : PostStep("MultiAggregation", lp.nRows(), lp.nCols())
         , m_j(_j)
         , m_i(_i)
         , m_old_j(lp.nCols() - 1)
         , m_old_i(lp.nRows() - 1)
         , m_upper(lp.upper(_j))
         , m_lower(lp.lower(_j))
         , m_obj(lp.spxSense() == SPxLPBase<R>::MINIMIZE ? lp.obj(_j) : -lp.obj(_j))
         , m_const(constant)
         , m_onLhs(EQ(constant, lp.lhs(_i)))
         , m_eqCons(EQ(lp.lhs(_i), lp.rhs(_i)))
         , m_row(lp.rowVector(_i))
         , m_col(lp.colVector(_j))
      {
         assert(m_row[m_j] != 0.0);
         simplifier.addObjoffset(m_obj * m_const / m_row[m_j]);
      }
      /// copy constructor
      MultiAggregationPS(const MultiAggregationPS& old)
         : PostStep(old)
         , m_j(old.m_j)
         , m_i(old.m_i)
         , m_old_j(old.m_old_j)
         , m_old_i(old.m_old_i)
         , m_upper(old.m_upper)
         , m_lower(old.m_lower)
         , m_obj(old.m_obj)
         , m_const(old.m_const)
         , m_onLhs(old.m_onLhs)
         , m_eqCons(old.m_eqCons)
         , m_row(old.m_row)
         , m_col(old.m_col)
      {}
      /// assignment operator
      MultiAggregationPS& operator=(const MultiAggregationPS& rhs)
      {
         if(this != &rhs)
         {
            PostStep::operator=(rhs);
            m_row = rhs.m_row;
            m_col = rhs.m_col;
         }

         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         MultiAggregationPS* MultiAggregationPSptr = 0;
         spx_alloc(MultiAggregationPSptr);
         return new(MultiAggregationPSptr) MultiAggregationPS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };

   /**@brief   Postsolves variable bound tightening from pseudo objective propagation.
      @ingroup Algo
   */
   class TightenBoundsPS : public PostStep
   {
   private:
      const int            m_j;
      const R           m_origupper;
      const R           m_origlower;

   public:
      ///
      TightenBoundsPS(const SPxLPBase<R>& lp, int j, R origupper, R origlower)
         : PostStep("TightenBounds", lp.nRows(), lp.nCols())
         , m_j(j)
         , m_origupper(origupper)
         , m_origlower(origlower)
      {
      }
      /// copy constructor
      TightenBoundsPS(const TightenBoundsPS& old)
         : PostStep(old)
         , m_j(old.m_j)
         , m_origupper(old.m_origupper)
         , m_origlower(old.m_origlower)
      {}
      /// assignment operator
      TightenBoundsPS& operator=(const TightenBoundsPS& rhs)
      {
         return *this;
      }
      /// clone function for polymorphism
      inline virtual PostStep* clone() const
      {
         TightenBoundsPS* TightenBoundsPSptr = 0;
         spx_alloc(TightenBoundsPSptr);
         return new(TightenBoundsPSptr) TightenBoundsPS(*this);
      }
      ///
      virtual void execute(VectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& s, VectorBase<R>& r,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& cBasis,
                           DataArray<typename SPxSolverBase<R>::VarStatus>& rBasis, bool isOptimal) const;
   };
   // friends
   friend class FreeConstraintPS;
   friend class EmptyConstraintPS;
   friend class RowSingletonPS;
   friend class ForceConstraintPS;
   friend class FixVariablePS;
   friend class FixBoundsPS;
   friend class FreeZeroObjVariablePS;
   friend class ZeroObjColSingletonPS;
   friend class FreeColSingletonPS;
   friend class DoubletonEquationPS;
   friend class DuplicateRowsPS;
   friend class DuplicateColsPS;
   friend class AggregationPS;

private:
   //------------------------------------
   ///@name Types
   ///@{
   /// Different simplification steps.
   enum SimpleStep
   {
      EMPTY_ROW            =  0,
      FREE_ROW             =  1,
      SINGLETON_ROW        =  2,
      FORCE_ROW            =  3,
      EMPTY_COL            =  4,
      FIX_COL              =  5,
      FREE_ZOBJ_COL        =  6,
      ZOBJ_SINGLETON_COL   =  7,
      DOUBLETON_ROW        =  8,
      FREE_SINGLETON_COL   =  9,
      DOMINATED_COL        = 10,
      WEAKLY_DOMINATED_COL = 11,
      DUPLICATE_ROW        = 12,
      FIX_DUPLICATE_COL    = 13,
      SUB_DUPLICATE_COL    = 14,
      AGGREGATION          = 15,
      MULTI_AGG            = 16
   };
   ///@}

   //------------------------------------
   ///@name Data
   ///@{
   ///
   VectorBase<R>                         m_prim;       ///< unsimplified primal solution VectorBase<R>.
   VectorBase<R>                         m_slack;      ///< unsimplified slack VectorBase<R>.
   VectorBase<R>                         m_dual;       ///< unsimplified dual solution VectorBase<R>.
   VectorBase<R>                         m_redCost;    ///< unsimplified reduced cost VectorBase<R>.
   DataArray<typename SPxSolverBase<R>::VarStatus> m_cBasisStat; ///< basis status of columns.
   DataArray<typename SPxSolverBase<R>::VarStatus> m_rBasisStat; ///< basis status of rows.
   DataArray<int>                  m_cIdx;       ///< column index VectorBase<R> in original LP.
   DataArray<int>                  m_rIdx;       ///< row index VectorBase<R> in original LP.
   Array<std::shared_ptr<PostStep>>m_hist;       ///< VectorBase<R> of presolve history.
   Array<DSVectorBase<R>>
                       m_classSetRows; ///< stores parallel classes with non-zero colum entry
   Array<DSVectorBase<R>>
                       m_classSetCols; ///< stores parallel classes with non-zero row entry
   Array<DSVectorBase<R>>
                       m_dupRows;    ///< arrange duplicate rows using bucket sort w.r.t. their pClass values
   Array<DSVectorBase<R>>
                       m_dupCols;    ///< arrange duplicate columns w.r.t. their pClass values
   bool                            m_postsolved; ///< status of postsolving.
   R                            m_epsilon;    ///< epsilon zero.
   R                            m_feastol;    ///< primal feasibility tolerance.
   R                            m_opttol;     ///< dual feasibility tolerance.
   DataArray<int>                  m_stat;       ///< preprocessing history.
   typename SPxLPBase<R>::SPxSense                 m_thesense;   ///< optimization sense.
   bool                            m_keepbounds;  ///< keep some bounds (for boundflipping)
   int                             m_addedcols;  ///< columns added by handleRowObjectives()
   typename SPxSimplifier<R>::Result m_result;     ///< result of the simplification.
   R                            m_cutoffbound;  ///< the cutoff bound that is found by heuristics
   R                            m_pseudoobj;    ///< the pseudo objective function value
   ///@}

private:
   //------------------------------------
   ///@name Private helpers
   ///@{
   /// handle row objectives
   void handleRowObjectives(SPxLPBase<R>& lp);

   /// handles extreme values by setting them to zero or R(infinity).
   void handleExtremes(SPxLPBase<R>& lp);

   /// computes the minimum and maximum residual activity for a given row and column. If colNumber is set to -1, then
   //  the activity of the row is returned.
   void computeMinMaxResidualActivity(SPxLPBase<R>& lp, int rowNumber, int colNumber, R& minAct,
                                      R& maxAct);

   /// calculate min/max value for the multi aggregated variables
   void computeMinMaxValues(SPxLPBase<R>& lp, R side, R val, R minRes, R maxRes, R& minVal, R& maxVal);

   /// tries to find good lower bound solutions by applying some trivial heuristics
   void trivialHeuristic(SPxLPBase<R>& lp);

   /// checks a solution for feasibility
   bool checkSolution(SPxLPBase<R>& lp, VectorBase<R> sol);

   /// tightens variable bounds by propagating the pseudo objective function value.
   void propagatePseudoobj(SPxLPBase<R>& lp);

   /// removed empty rows and empty columns.
   typename SPxSimplifier<R>::Result removeEmpty(SPxLPBase<R>& lp);

   /// remove row singletons.
   typename SPxSimplifier<R>::Result removeRowSingleton(SPxLPBase<R>& lp, const SVectorBase<R>& row,
         int& i);

   /// aggregate two variables that appear in an equation.
   typename SPxSimplifier<R>::Result aggregateVars(SPxLPBase<R>& lp, const SVectorBase<R>& row,
         int& i);

   /// performs simplification steps on the rows of the LP.
   typename SPxSimplifier<R>::Result simplifyRows(SPxLPBase<R>& lp, bool& again);

   /// performs simplification steps on the columns of the LP.
   typename SPxSimplifier<R>::Result simplifyCols(SPxLPBase<R>& lp, bool& again);

   /// performs simplification steps on the LP based on dual concepts.
   typename SPxSimplifier<R>::Result simplifyDual(SPxLPBase<R>& lp, bool& again);

   /// performs multi-aggregations of variable based upon constraint activitu.
   typename SPxSimplifier<R>::Result multiaggregation(SPxLPBase<R>& lp, bool& again);

   /// removes duplicate rows.
   typename SPxSimplifier<R>::Result duplicateRows(SPxLPBase<R>& lp, bool& again);

   /// removes duplicate columns
   typename SPxSimplifier<R>::Result duplicateCols(SPxLPBase<R>& lp, bool& again);

   /// handles the fixing of a variable. correctIdx is true iff the index mapping has to be updated.
   void fixColumn(SPxLPBase<R>& lp, int i, bool correctIdx = true);

   /// removes a row in the LP.
   void removeRow(SPxLPBase<R>& lp, int i)
   {
      m_rIdx[i] = m_rIdx[lp.nRows() - 1];
      lp.removeRow(i);
   }
   /// removes a column in the LP.
   void removeCol(SPxLPBase<R>& lp, int j)
   {
      m_cIdx[j] = m_cIdx[lp.nCols() - 1];
      lp.removeCol(j);
   }
   /// returns for a given row index of the (reduced) LP the corresponding row index in the unsimplified LP.
   int rIdx(int i) const
   {
      return m_rIdx[i];
   }
   /// returns for a given column index of the (reduced) LP the corresponding column index in the unsimplified LP.
   int cIdx(int j) const
   {
      return m_cIdx[j];
   }
   ///@}

protected:

   ///
   R epsZero() const
   {
      return m_epsilon;
   }
   ///
   R feastol() const
   {
      return m_feastol;
   }
   ///
   R opttol() const
   {
      return m_opttol;
   }

public:

   //------------------------------------
   ///@name Constructors / destructors
   ///@{
   /// default constructor.
   SPxMainSM(Timer::TYPE ttype = Timer::USER_TIME)
      : SPxSimplifier<R>("MainSM", ttype)
      , m_postsolved(0)
      , m_epsilon(DEFAULT_EPS_ZERO)
      , m_feastol(DEFAULT_BND_VIOL)
      , m_opttol(DEFAULT_BND_VIOL)
      , m_stat(16)
      , m_thesense(SPxLPBase<R>::MAXIMIZE)
      , m_keepbounds(false)
      , m_addedcols(0)
      , m_result(this->OKAY)
      , m_cutoffbound(R(-infinity))
      , m_pseudoobj(R(-infinity))
   {}
   /// copy constructor.
   SPxMainSM(const SPxMainSM& old)
      : SPxSimplifier<R>(old)
      , m_prim(old.m_prim)
      , m_slack(old.m_slack)
      , m_dual(old.m_dual)
      , m_redCost(old.m_redCost)
      , m_cBasisStat(old.m_cBasisStat)
      , m_rBasisStat(old.m_rBasisStat)
      , m_cIdx(old.m_cIdx)
      , m_rIdx(old.m_rIdx)
      , m_hist(old.m_hist)
      , m_postsolved(old.m_postsolved)
      , m_epsilon(old.m_epsilon)
      , m_feastol(old.m_feastol)
      , m_opttol(old.m_opttol)
      , m_stat(old.m_stat)
      , m_thesense(old.m_thesense)
      , m_keepbounds(old.m_keepbounds)
      , m_addedcols(old.m_addedcols)
      , m_result(old.m_result)
      , m_cutoffbound(old.m_cutoffbound)
      , m_pseudoobj(old.m_pseudoobj)
   {
      ;
   }
   /// assignment operator
   SPxMainSM& operator=(const SPxMainSM& rhs)
   {
      if(this != &rhs)
      {
         SPxSimplifier<R>::operator=(rhs);
         m_prim = rhs.m_prim;
         m_slack = rhs.m_slack;
         m_dual = rhs.m_dual;
         m_redCost = rhs.m_redCost;
         m_cBasisStat = rhs.m_cBasisStat;
         m_rBasisStat = rhs.m_rBasisStat;
         m_cIdx = rhs.m_cIdx;
         m_rIdx = rhs.m_rIdx;
         m_postsolved = rhs.m_postsolved;
         m_epsilon = rhs.m_epsilon;
         m_feastol = rhs.m_feastol;
         m_opttol = rhs.m_opttol;
         m_stat = rhs.m_stat;
         m_thesense = rhs.m_thesense;
         m_keepbounds = rhs.m_keepbounds;
         m_addedcols = rhs.m_addedcols;
         m_result = rhs.m_result;
         m_cutoffbound = rhs.m_cutoffbound;
         m_pseudoobj = rhs.m_pseudoobj;
         m_hist = rhs.m_hist;
      }


      return *this;
   }
   /// destructor.
   virtual ~SPxMainSM()
   {
      ;
   }
   /// clone function for polymorphism
   inline virtual SPxSimplifier<R>* clone() const
   {
      return new SPxMainSM(*this);
   }
   ///@}

   //------------------------------------
   //**@name LP simplification */
   ///@{
   /// simplify SPxLPBase<R> \p lp with identical primal and dual feasibility tolerance.
   virtual typename SPxSimplifier<R>::Result simplify(SPxLPBase<R>& lp, R eps, R delta,
         Real remainingTime)
   {
      return simplify(lp, eps, delta, delta, remainingTime);
   }
   /// simplify SPxLPBase<R> \p lp with independent primal and dual feasibility tolerance.
   virtual typename SPxSimplifier<R>::Result simplify(SPxLPBase<R>& lp, R eps, R ftol, R otol,
         Real remainingTime,
         bool keepbounds = false, uint32_t seed = 0);

   /// reconstructs an optimal solution for the unsimplified LP.
   virtual void unsimplify(const VectorBase<R>& x, const VectorBase<R>& y, const VectorBase<R>& s,
                           const VectorBase<R>& r,
                           const typename SPxSolverBase<R>::VarStatus rows[],
                           const typename SPxSolverBase<R>::VarStatus cols[], bool isOptimal = true);

   /// returns result status of the simplification
   virtual typename SPxSimplifier<R>::Result result() const
   {
      return m_result;
   }

   /// specifies whether an optimal solution has already been unsimplified.
   virtual bool isUnsimplified() const
   {
      return m_postsolved;
   }
   /// returns a reference to the unsimplified primal solution.
   virtual const VectorBase<R>& unsimplifiedPrimal()
   {
      assert(m_postsolved);
      return m_prim;
   }
   /// returns a reference to the unsimplified dual solution.
   virtual const VectorBase<R>& unsimplifiedDual()
   {
      assert(m_postsolved);
      return m_dual;
   }
   /// returns a reference to the unsimplified slack values.
   virtual const VectorBase<R>& unsimplifiedSlacks()
   {
      assert(m_postsolved);
      return m_slack;
   }
   /// returns a reference to the unsimplified reduced costs.
   virtual const VectorBase<R>& unsimplifiedRedCost()
   {
      assert(m_postsolved);
      return m_redCost;
   }
   /// gets basis status for a single row.
   virtual typename SPxSolverBase<R>::VarStatus getBasisRowStatus(int i) const
   {
      assert(m_postsolved);
      return m_rBasisStat[i];
   }
   /// gets basis status for a single column.
   virtual typename SPxSolverBase<R>::VarStatus getBasisColStatus(int j) const
   {
      assert(m_postsolved);
      return m_cBasisStat[j];
   }
   /// get optimal basis.
   virtual void getBasis(typename SPxSolverBase<R>::VarStatus rows[],
                         typename SPxSolverBase<R>::VarStatus cols[], const int rowsSize = -1, const int colsSize = -1) const
   {
      assert(m_postsolved);
      assert(rowsSize < 0 || rowsSize >= m_rBasisStat.size());
      assert(colsSize < 0 || colsSize >= m_cBasisStat.size());

      for(int i = 0; i < m_rBasisStat.size(); ++i)
         rows[i] = m_rBasisStat[i];

      for(int j = 0; j < m_cBasisStat.size(); ++j)
         cols[j] = m_cBasisStat[j];
   }
   ///@}

private:
   //------------------------------------
   //**@name Types */
   ///@{
   /// comparator for class SVectorBase<R>::Element: compare nonzeros according to value
   struct ElementCompare
   {
   public:
      ElementCompare() {}

      int operator()(const typename SVectorBase<R>::Element& e1,
                     const typename SVectorBase<R>::Element& e2) const
      {
         if(EQ(e1.val, e2.val))
            return 0;

         if(e1.val < e2.val)
            return -1;
         else // (e1.val > e2.val)
            return 1;
      }
   };
   /// comparator for class SVectorBase<R>::Element: compare nonzeros according to index
   struct IdxCompare
   {
   public:
      IdxCompare() {}

      int operator()(const typename SVectorBase<R>::Element& e1,
                     const typename SVectorBase<R>::Element& e2) const
      {
         if(EQ(e1.idx, e2.idx))
            return 0;

         if(e1.idx < e2.idx)
            return -1;
         else // (e1.idx > e2.idx)
            return 1;
      }
   };
   ///@}
};

} // namespace soplex

// For including general templated functions
#include "spxmainsm.hpp"

#endif // _SPXMAINSM_H_
