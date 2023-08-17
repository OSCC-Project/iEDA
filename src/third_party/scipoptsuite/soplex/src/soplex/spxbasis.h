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

/**@file  spxbasis.h
 * @brief Simplex basis.
 */
#ifndef _SPXBASIS_H_
#define _SPXBASIS_H_

/* undefine SOPLEX_DEBUG flag from including files; if SOPLEX_DEBUG should be defined in this file, do so below */
#ifdef SOPLEX_DEBUG
#define SOPLEX_DEBUG_SPXBASIS
#undef SOPLEX_DEBUG
#endif

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <sstream>

#include "soplex/spxdefines.h"
#include "soplex/spxlp.h"
#include "soplex/svector.h"
#include "soplex/ssvector.h"
#include "soplex/dataarray.h"
#include "soplex/slinsolver.h"
#include "soplex/nameset.h"
#include "soplex/spxout.h"
#include "soplex/timerfactory.h"

//#define MEASUREUPDATETIME

namespace soplex
{
template <class R>
class SPxSolverBase;

/**@class SPxBasisBase
   @brief   Simplex basis.
   @ingroup Algo

   Consider the linear program as provided from class SPxLP:
   \f[
   \begin{array}{rl}
   \hbox{max}  & c^T x                 \\
   \hbox{s.t.} & l_r \le Ax \le u_r    \\
   & l_c \le  x \le u_c
   \end{array}
   \f]
   where \f$c, l_c, u_c, x \in {\bf R}^n\f$, \f$l_r, u_r \in {\bf R}^m\f$ and
   \f$A \in {\bf R}^{m \times n}\f$. Solving this LP with the simplex algorithm
   requires the definition of a \em basis. Such can be defined as a set of
   column vectors or a set of row vectors building a non-singular matrix. We
   will refer to the first case as the \em columnwise \em representation and
   the latter case will be called the \em rowwise \em representation. In both
   cases, a \em basis is a set of vectors forming a non-singular matrix. The
   dimension of the vectors is referred to as the basis' \em dimension,
   whereas the number of vectors belonging to the LP is called the basis'
   \em codimension.

   Class SPxBasisBase is designed to represent a generic simplex basis, suitable
   for both representations. At any time the representation can be changed by
   calling method setRep().

   Class SPxBasisBase provides methods for solving linear systems with the basis
   matrix. However, SPxBasisBase does not provide a linear solver by its own.
   Instead, a SLinSolver object must be #load%ed to a SPxBasisBase which will
   be called for solving linear systems.
*/
template <class R> // theLP gets templated
class SPxBasisBase
{
public:

   /// basis status.
   /** Each SPxBasisBase is assigned a status flag, which can take on of the
       above values.
   */
   enum SPxStatus
   {
      NO_PROBLEM = -2,  ///< No Problem has been loaded to the basis.
      SINGULAR   = -1,  ///< Basis is singular.
      REGULAR    = 0,   ///< Basis is not known to be dual nor primal feasible.
      DUAL       = 1,   ///< Basis is dual feasible.
      PRIMAL     = 2,   ///< Basis is primal feasible.
      OPTIMAL    = 3,   ///< Basis is optimal, i.e. dual and primal feasible.
      UNBOUNDED  = 4,   ///< LP has been proven to be primal unbounded.
      INFEASIBLE = 5    ///< LP has been proven to be primal infeasible.
   };


   /// Basis descriptor.
   class Desc
   {
   public:

      //------------------------------------
      ///@name Status
      ///@{
      /// Status of a variable.
      /** A basis is described by assigning a Status to all of the LP
          variables and covariables. This assignment is maintained by the
          basis #Desc%riptor.

          Variables and covariables (slackvariables) may have a primal or dual Status. The
          first type specifies that a variable is set on a primal bound, while
          the latter type indicates a dual variable to be set on a bound.
          If a row variable has a primal status, say #P_ON_UPPER, this means
          that the upper bound of the inequality is set to be tight. Hence,
          in this case the upper bound must not be infinity.

          Equivalently, if the status of a variable is dual, say #D_ON_UPPER,
          it means that the dual variable corresponding to the upper bound
          inequality of this variable is set to 0.

          For a column basis, primal #Status%es correspond to nonbasic
          variables, while dual ones are basic. This is reversed for a row
          basis. We will now reveal in more detail the significance of
          variable #Status%es.

          <b>Primal Variables</b>

          Consider a range inequality \f$l_r \le a^T x \le u_r\f$ or bounds on
          a variable \f$l_c \le x_c \le u_c\f$. The following table reveals
          what is implied if the corresponding variable or covariable is
          assigned to a primal #Status:

          \f[
          \begin{array}{lcl}
          l_c \le x_c \le u_c   & \mbox{Status}(x_i)  & l_r \le a^T x \le u_r \\
          \hline
          x_c = u_c < \infty    & \mbox{P\_ON\_UPPER} & a^T x = u_r < \infty  \\
          x_c = l_c > -\infty   & \mbox{P\_ON\_LOWER} & a^T x = l_r > -\infty \\
          -\infty < l_c = x_c = u_c < \infty
          & \mbox{P\_FIXED}     &
          -\infty < l_r = a^T x = u_r < \infty  \\
          -\infty = l_i < x_i=0 < u_i = \infty
          & \mbox{P\_FREE}      &
          -\infty = l_r < a^T x = 0 < u_r = \infty  \\
          \end{array}
          \f]

          Note that to determine whether a variable with #Status stat is set to
          its upper bound, one can compute the test (-stat | -#P_ON_UPPER).
          This will yield true even if the variable is fixed, i.e., sitting on
          both bounds at the same time.

          <b>Dual Variables</b>

          In principle for implementing the Simplex algorithm it would suffice
          to use only one dual #Status. However, for performance reasons it
          is advisable to introduce various dual status types, reflecting
          the structure of the bounds. Given an upper bound \f$u\f$ and a lower
          bound \f$l\f$ of a constraint or variable, the following table
          indicates the setting of the dual Status of this variable.

          \f[
          \begin{array}{cl}
          l \le ... \le u               & \mbox{Status}        \\
          \hline
          -\infty < l \ne u < \infty    & \mbox{D\_ON\_BOTH}   \\
          -\infty < l \ne u = \infty    & \mbox{D\_ON\_UPPER}  \\
          -\infty = l \ne u < \infty    & \mbox{D\_ON\_LOWER}  \\
          -\infty < l  =  u < \infty    & \mbox{D\_FREE}       \\
          -\infty = l \ne u = \infty    & \mbox{D\_UNDEFINED}  \\
          \end{array}
          \f]

          Note that unbounded primal variables are reflected by an #D_UNDEFINED
          dual variable, since no reduced costs exist for them. To facilitate
          the assignment of dual #Status%es, class SPxBasisBase provides methods
          #dualStatus(), #dualColStatus() and #dualRowStatus)().
      */
      enum Status
      {
         P_ON_LOWER  = -4,  ///< primal variable is set to its lower bound
         P_ON_UPPER  = -2,  ///< primal variable is set to its upper bound
         P_FREE      = -1,  ///< primal variable is left free, but unset
         P_FIXED     = P_ON_UPPER + P_ON_LOWER,  ///< primal variable is fixed to both bounds
         D_FREE      = 1,   ///< dual variable is left free, but unset
         D_ON_UPPER  = 2,   ///< dual variable is set to its upper bound
         D_ON_LOWER  = 4,   ///< dual variable is set to its lower bound
         D_ON_BOTH   = D_ON_LOWER + D_ON_UPPER,  ///< dual variable has two bounds
         D_UNDEFINED = 8    ///< primal or dual variable is undefined
      };
      ///@}

      friend SPxBasisBase<R>;
      template <class T> friend std::ostream& operator<< (std::ostream& os,
            const Status& stat); //@todo is the <> required here?

   private:

      //------------------------------------
      ///@name Data
      ///@{
      DataArray < Status > rowstat;   ///< status of rows.
      DataArray < Status > colstat;   ///< status of columns.
      DataArray < Status >* stat;     ///< basis' status.
      DataArray < Status >* costat;   ///< cobasis' status.
      ///@}

   public:

      //------------------------------------
      ///@name Access / modification
      ///@{
      /// returns number of columns.
      int nCols() const
      {
         return colstat.size();
      }
      /// returns number of rows.
      int nRows() const
      {
         return rowstat.size();
      }
      /// returns dimension.
      int dim() const
      {
         return stat->size();
      }
      /// returns codimension.
      int coDim() const
      {
         return costat->size();
      }
      ///
      Status& rowStatus(int i)
      {
         return rowstat[i];
      }
      /// returns status of row \p i.
      Status rowStatus(int i) const
      {
         return rowstat[i];
      }
      /// returns the array of row  \ref soplex::SPxBasisBase<R>::Desc::Status "Status"es.
      const Status* rowStatus(void) const
      {
         return rowstat.get_const_ptr();
      }
      ///
      Status& colStatus(int i)
      {
         return colstat[i];
      }
      /// returns status of column \p i.
      Status colStatus(int i) const
      {
         return colstat[i];
      }
      /// returns the array of column \ref soplex::SPxBasisBase<R>::Desc::Status "Status"es.
      const Status* colStatus(void) const
      {
         return colstat.get_const_ptr();
      }
      ///
      Status& status(int i)
      {
         return (*stat)[i];
      }
      /// returns status of variable \p i.
      Status status(int i) const
      {
         return (*stat)[i];
      }
      /// returns the array of variable \ref soplex::SPxBasisBase<R>::Desc::Status "Status"es.
      const Status* status(void) const
      {
         return stat->get_const_ptr();
      }
      ///
      Status& coStatus(int i)
      {
         return (*costat)[i];
      }
      /// returns status of covariable \p i.
      Status coStatus(int i) const
      {
         return (*costat)[i];
      }
      /// returns the array of covariable \ref soplex::SPxBasisBase<R>::Desc::Status "Status"es.
      const Status* coStatus(void) const
      {
         return costat->get_const_ptr();
      }
      /// resets dimensions.
      void reSize(int rowDim, int colDim);
      ///@}

      //------------------------------------
      ///@name Debugging
      ///@{
      /// Prints out status.
      void dump() const;

      /// consistency check.
      bool isConsistent() const;
      ///@}

      //------------------------------------
      ///@name Construction / destruction
      ///@{
      /// default constructor
      Desc()
         : stat(0)
         , costat(0)
      {}
      explicit Desc(const SPxSolverBase<R>& base);

      /// copy constructor
      Desc(const Desc& old);
      /// assignment operator
      Desc& operator=(const Desc& rhs);
      ///@}
   };

protected:

   //------------------------------------
   //**@name Protected data
   /**
      For storing the basis matrix we keep two arrays: Array #theBaseId
      contains the SPxId%s of the basis vectors, and #matrix the pointers to
      the vectors themselfes. Method #loadMatrixVecs() serves for loading
      #matrix according to the SPxId%s stored in #theBaseId. This method must
      be called whenever the VectorBase<R> pointers may have
      changed due to manipulations of the LP.
   */
   ///@{
   /// the LP
   SPxSolverBase<R>* theLP;
   /// SPxId%s of basic vectors.
   DataArray < SPxId > theBaseId;
   /// pointers to the vectors of the basis matrix.
   DataArray < const SVectorBase<R>* > matrix;
   /// \c true iff the pointers in \ref soplex::SPxBasisBase<R>::matrix "matrix" are set up correctly.
   bool matrixIsSetup;

   /* @brief LU factorization of basis matrix
      The factorization of the matrix is stored in #factor if #factorized != 0.
      Otherwise #factor is undefined.
   */
   SLinSolver<R>* factor;
   /// \c true iff \ref soplex::SPxBasisBase<R>::factor "factor" = \ref soplex::SPxBasisBase<R>::matrix "matrix" \f$^{-1}\f$.
   bool factorized;

   /// number of updates before refactorization.
   /** When a vector of the basis matrix is exchanged by a call to method
       #change(), the LU factorization of the matrix is updated
       accordingly. However, after atmost #maxUpdates updates of the
       factorization, it is recomputed in order to regain numerical
       stability and reduce fill in.
   */
   int   maxUpdates;

   /// allowed increase of nonzeros before refactorization.
   /** When the number of nonzeros in LU factorization exceeds
       #nonzeroFactor times the number of nonzeros in B, the
       basis matrix is refactorized.
   */
   R   nonzeroFactor;

   /// allowed increase in relative fill before refactorization
   /** When the real relative fill is bigger than fillFactor times lastFill
    *  the Basis will be refactorized.
    */
   R   fillFactor;

   /// allowed total increase in memory consumption before refactorization
   R   memFactor;

   /* Rank-1-updates to the basis may be performed via method #change(). In
      this case, the factorization is updated, and the following members are
      reset.
   */
   int    iterCount;     ///< number of calls to change() since last manipulation
   int    lastIterCount; ///< number of calls to change() before halting the simplex
   int    iterDegenCheck;///< number of calls to change() since last degeneracy check
   int    updateCount;   ///< number of calls to change() since last factorize()
   int    totalUpdateCount; ///< number of updates
   int    nzCount;       ///< number of nonzeros in basis matrix
   int    lastMem;       ///< memory needed after last fresh factorization
   R   lastFill;      ///< fill ratio that occured during last factorization
   int    lastNzCount;   ///< number of nonzeros in basis matrix after last fresh factorization

   Timer* theTime;  ///< time spent in updates
   Timer::TYPE timerType;   ///< type of timer (user or wallclock)

   SPxId  lastin;        ///< lastEntered(): variable entered the base last
   SPxId  lastout;       ///< lastLeft(): variable left the base last
   int    lastidx;       ///< lastIndex(): basis index where last update was done
   R   minStab;       ///< minimum stability
   ///@}

private:

   //------------------------------------
   //**@name Private data */
   ///@{
   SPxStatus thestatus;      ///< current status of the basis.
   Desc      thedesc;        ///< the basis' Descriptor
   bool      freeSlinSolver; ///< true iff factor should be freed inside of this object
   SPxOut*   spxout;         ///< message handler

   ///@}

public:

   //------------------------------------------------
   /**@name Status and Descriptor related Methods */
   ///@{
   /// returns current SPxStatus.
   SPxStatus status() const
   {
      return thestatus;
   }

   /// sets basis SPxStatus to \p stat.
   void setStatus(SPxStatus stat)
   {

      if(thestatus != stat)
      {
#ifdef SOPLEX_DEBUG
         MSG_DEBUG(std::cout << "DBSTAT01 SPxBasisBase<R>::setStatus(): status: "
                   << int(thestatus) << " (" << thestatus << ") -> "
                   << int(stat) << " (" << stat << ")" << std::endl;)
#endif

         thestatus = stat;

         if(stat == NO_PROBLEM)
            invalidate();
      }
   }

   // TODO control factorization frequency dynamically
   /// change maximum number of iterations until a refactorization is performed
   void setMaxUpdates(int maxUp)
   {
      assert(maxUp >= 0);
      maxUpdates = maxUp;
   }

   /// returns maximum number of updates before a refactorization is performed
   int getMaxUpdates() const
   {
      return maxUpdates;
   }

   ///
   const Desc& desc() const
   {
      return thedesc;
   }
   /// returns current basis Descriptor.
   Desc& desc()
   {
      return thedesc;
   }

   /// dual Status for the \p i'th column variable of the loaded LP.
   typename Desc::Status dualColStatus(int i) const;

   /// dual Status for the column variable with ID \p id of the loaded LP.
   typename Desc::Status dualStatus(const SPxColId& id) const;

   /// dual Status for the \p i'th row variable of the loaded LP.
   typename Desc::Status dualRowStatus(int i) const;

   /// dual Status for the row variable with ID \p id of the loaded LP.
   typename Desc::Status dualStatus(const SPxRowId& id) const;

   /// dual Status for the variable with ID \p id of the loaded LP.
   /** It is automatically detected, whether the \p id is one of a
       row or a column variable, and the correct row or column status
       is returned.
   */
   typename Desc::Status dualStatus(const SPxId& id) const
   {
      return id.isSPxRowId()
             ? dualStatus(SPxRowId(id))
             : dualStatus(SPxColId(id));
   }
   ///@}


   //-----------------------------------
   /**@name Inquiry Methods */
   ///@{
   ///
   inline SPxId& baseId(int i)
   {
      return theBaseId[i];
   }
   /// returns the Id of the \p i'th basis vector.
   inline SPxId baseId(int i) const
   {
      return theBaseId[i];
   }

   /// returns the \p i'th basic vector.
   const SVectorBase<R>& baseVec(int i) const
   {
      assert(matrixIsSetup);
      return *matrix[i];
   }

   /// returns SPxId of last VectorBase<R> included to the basis.
   inline SPxId lastEntered() const
   {
      return lastin;
   }

   /// returns SPxId of last vector that left the basis.
   inline SPxId lastLeft() const
   {
      return lastout;
   }

   /// returns index in basis where last update was done.
   inline int lastIndex() const
   {
      return lastidx;
   }

   /// returns number of basis changes since last refactorization.
   inline int lastUpdate() const
   {
      return updateCount;
   }

   /// returns number of basis changes since last \ref soplex::SPxBasisBase<R>::load() "load()".
   inline int iteration() const
   {
      return iterCount;
   }

   /// returns the number of iterations prior to the last break in execution
   inline int prevIteration() const
   {
      return lastIterCount;
   }

   /// returns the number of iterations since the last degeneracy check
   inline int lastDegenCheck() const
   {
      return iterDegenCheck;
   }

   /// returns loaded solver.
   inline SPxSolverBase<R>* solver() const
   {
      return theLP;
   }
   ///@}

   //-----------------------------------
   /**@name Linear Algebra */
   ///@{
   /// Basis-vector product.
   /** Depending on the representation, for an SPxBasisBase B,
       B.multBaseWith(x) computes
       - \f$x \leftarrow Bx\f$    in the columnwise case, and
       - \f$x \leftarrow x^TB\f$  in the rowwise case.

       Both can be seen uniformly as multiplying the basis matrix \p B with
       a vector \p x aligned the same way as the \em vectors of \p B.
   */
   VectorBase<R>& multBaseWith(VectorBase<R>& x) const;

   /// Basis-vector product
   void multBaseWith(SSVectorBase<R>& x, SSVectorBase<R>& result) const;

   /// Vector-basis product.
   /** Depending on the representation, for a #SPxBasisBase B,
       B.multWithBase(x) computes
       - \f$x \leftarrow x^TB\f$  in the columnwise case and
       - \f$x \leftarrow Bx\f$    in the rowwise case.

       Both can be seen uniformly as multiplying the basis matrix \p B with
       a vector \p x aligned the same way as the \em covectors of \p B.
   */
   VectorBase<R>& multWithBase(VectorBase<R>& x) const;

   /// VectorBase<R>-basis product
   void multWithBase(SSVectorBase<R>& x, SSVectorBase<R>& result) const;

   /* compute an estimated condition number for the current basis matrix
    * by computing estimates of the norms of B and B^-1 using the power method.
    * maxiters and tolerance control the accuracy of the estimate.
    */
   R condition(int maxiters = 10, R tolerance = 1e-6);

   /* wrapper to compute an estimate of the condition number of the current basis matrix */
   R getEstimatedCondition()
   {
      return condition(20, 1e-6);
   }

   /* wrapper to compute the exact condition number of the current basis matrix */
   R getExactCondition()
   {
      return condition(1000, 1e-9);
   }

   /** compute one of several matrix metrics based on the diagonal of the LU factorization
     * type = 0: max/min ratio
     * type = 1: trace of U (sum of diagonal elements)
    *  type = 2: determinant (product of diagonal elements)
     */
   R getMatrixMetric(int type = 0);

   /// returns the stability of the basis matrix.
   R stability() const
   {
      return factor->stability();
   }
   ///
   void solve(VectorBase<R>& x, const VectorBase<R>& rhs)
   {
      if(rhs.dim() == 0)
      {
         x.clear();
         return;
      }

      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solveRight(x, rhs);
   }
   ///
   void solve(SSVectorBase<R>& x, const SVectorBase<R>& rhs)
   {
      if(rhs.size() == 0)
      {
         x.clear();
         return;
      }

      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solveRight(x, rhs);
   }
   /// solves linear system with basis matrix.
   /** Depending on the representation, for a SPxBasisBase B,
       B.solve(x) computes
       - \f$x \leftarrow B^{-1}rhs\f$       in the columnwise case and
       - \f$x \leftarrow rhs^TB^{-1}\f$     in the rowwise case.

       Both can be seen uniformly as solving a linear system with the basis
       matrix \p B and a right handside vector \p x aligned the same way as
       the \em vectors of \p B.
   */
   void solve4update(SSVectorBase<R>& x, const SVectorBase<R>& rhs)
   {
      if(rhs.size() == 0)
      {
         x.clear();
         return;
      }

      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solveRight4update(x, rhs);
   }
   /// solves two systems in one call.
   void solve4update(SSVectorBase<R>& x, VectorBase<R>& y, const SVectorBase<R>& rhsx,
                     SSVectorBase<R>& rhsy)
   {
      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solve2right4update(x, y, rhsx, rhsy);
   }
   /// solves two systems in one call using only sparse data structures
   void solve4update(SSVectorBase<R>& x, SSVectorBase<R>& y, const SVectorBase<R>& rhsx,
                     SSVectorBase<R>& rhsy)
   {
      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solve2right4update(x, y, rhsx, rhsy);
   }
   /// solves three systems in one call.
   void solve4update(SSVectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& y2,
                     const SVectorBase<R>& rhsx, SSVectorBase<R>& rhsy, SSVectorBase<R>& rhsy2)
   {
      if(!factorized)
         SPxBasisBase<R>::factorize();

      assert(rhsy.isSetup());
      assert(rhsy2.isSetup());
      factor->solve3right4update(x, y, y2, rhsx, rhsy, rhsy2);
   }
   /// solves three systems in one call using only sparse data structures
   void solve4update(SSVectorBase<R>& x, SSVectorBase<R>& y, SSVectorBase<R>& y2,
                     const SVectorBase<R>& rhsx, SSVectorBase<R>& rhsy, SSVectorBase<R>& rhsy2)
   {
      if(!factorized)
         SPxBasisBase<R>::factorize();

      assert(rhsy.isSetup());
      assert(rhsy2.isSetup());
      factor->solve3right4update(x, y, y2, rhsx, rhsy, rhsy2);
   }
   /// Cosolves linear system with basis matrix.
   /** Depending on the representation, for a SPxBasisBase B,
       B.coSolve(x) computes
       - \f$x \leftarrow rhs^TB^{-1}\f$     in the columnwise case and
       - \f$x \leftarrow B^{-1}rhs\f$       in the rowwise case.

       Both can be seen uniformly as solving a linear system with the basis
       matrix \p B and a right handside vector \p x aligned the same way as
       the \em covectors of \p B.
   */
   void coSolve(VectorBase<R>& x, const VectorBase<R>& rhs)
   {
      if(rhs.dim() == 0)
      {
         x.clear();
         return;
      }

      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solveLeft(x, rhs);
   }
   /// Sparse version of coSolve
   void coSolve(SSVectorBase<R>& x, const SVectorBase<R>& rhs)
   {
      if(rhs.size() == 0)
      {
         x.clear();
         return;
      }

      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solveLeft(x, rhs);
   }
   /// solves two systems in one call.
   void coSolve(SSVectorBase<R>& x, VectorBase<R>& y, const SVectorBase<R>& rhsx,
                SSVectorBase<R>& rhsy)
   {
      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solveLeft(x, y, rhsx, rhsy);
   }
   /// Sparse version of solving two systems in one call.
   void coSolve(SSVectorBase<R>& x, SSVectorBase<R>& y, const SVectorBase<R>& rhsx,
                SSVectorBase<R>& rhsy)
   {
      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solveLeft(x, y, rhsx, rhsy);
   }
   /// solves three systems in one call. May be improved by using just one pass through the basis.
   void coSolve(SSVectorBase<R>& x, VectorBase<R>& y, VectorBase<R>& z, const SVectorBase<R>& rhsx,
                SSVectorBase<R>& rhsy, SSVectorBase<R>& rhsz)
   {
      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solveLeft(x, y, z, rhsx, rhsy, rhsz);
   }
   /// Sparse version of solving three systems in one call.
   void coSolve(SSVectorBase<R>& x, SSVectorBase<R>& y, SSVectorBase<R>& z, const SVectorBase<R>& rhsx,
                SSVectorBase<R>& rhsy, SSVectorBase<R>& rhsz)
   {
      if(!factorized)
         SPxBasisBase<R>::factorize();

      factor->solveLeft(x, y, z, rhsx, rhsy, rhsz);
   }
   ///@}


   //------------------------------------
   /**@name Modification notification.
      These methods must be called after the loaded LP has been modified.
   */
   ///@{
   /// inform SPxBasisBase, that \p n new rows had been added.
   void addedRows(int n);
   /// inform SPxBasisBase that row \p i had been removed.
   void removedRow(int i);
   /// inform SPxBasisBase that rows in \p perm with negative entry were removed.
   void removedRows(const int perm[]);
   /// inform SPxBasisBase that \p n new columns had been added.
   void addedCols(int n);
   /// inform SPxBasisBase that column \p i had been removed.
   void removedCol(int i);
   /// inform SPxBasisBase that columns in \p perm with negative entry were removed.
   void removedCols(const int perm[]);
   /// inform SPxBasisBase that a row had been changed.
   void changedRow(int);
   /// inform SPxBasisBase that a column had been changed.
   void changedCol(int);
   /// inform SPxBasisBase that a matrix entry had been changed.
   void changedElement(int, int);
   ///@}


   //--------------------------------
   /**@name Miscellaneous */
   ///@{
   /// performs basis update.
   /** Changes the \p i 'th vector of the basis with the vector associated to
       \p id. This includes:
       - updating the factorization, or recomputing it from scratch by
       calling   \ref soplex::SPxSolverBase<R>::factorize()   "factorize()",
       - resetting \ref soplex::SPxSolverBase<R>::lastEntered() "lastEntered()",
       - resetting \ref soplex::SPxSolverBase<R>::lastIndex()   "lastIndex()",
       - resetting \ref soplex::SPxSolverBase<R>::lastLeft()    "lastLeft()",
       - resetting \ref soplex::SPxSolverBase<R>::lastUpdate()  "lastUpdate()",
       - resetting \ref soplex::SPxSolverBase<R>::iterations()  "iterations()".

       The basis descriptor is \em not \em modified, since #factor()
       cannot know about how to set up the status of the involved variables
       correctly.

       A vector \p enterVec may be passed for a fast ETA update of the LU
       factorization associated to the basis. It must be initialized with
       the solution vector \f$x\f$ of the right linear system \f$Bx = b\f$
       with the entering vector as right-hand side vector \f$b\f$, where \f$B\f$
       denotes the basis matrix. This can be computed using method #solve().
       When using FAST updates, a vector \p eta may be passed for
       improved performance. It must be initialized by a call to
       factor->solveRightUpdate() as described in SLinSolver. The
       implementation hidden behind FAST updates depends on the
       SLinSolver implementation class.
   */
   virtual void change(int i, SPxId& id,
                       const SVectorBase<R>* enterVec, const SSVectorBase<R>* eta = 0);

   /** Load basis from \p in in MPS format. If \p rowNames and \p colNames
    *  are \c NULL, default names are used for the constraints and variables.
    */
   virtual bool readBasis(std::istream& in,
                          const NameSet* rowNames, const NameSet* colNames);

   /** Write basis to \p os in MPS format. If \p rowNames and \p colNames are
    *  \c NULL, default names are used for the constraints and variables.
    */
   virtual void writeBasis(std::ostream& os,
                           const NameSet* rownames, const NameSet* colnames, const bool cpxFormat = false) const;

   virtual void printMatrix() const;

   /** Prints current basis matrix to a file using the MatrixMarket format:
    *  row col value
    *  The filename is basis/basis[number].mtx where number is a parameter.
    */
   void printMatrixMTX(int number);

   /// checks if a Descriptor is valid for the current LP w.r.t. its bounds
   virtual bool isDescValid(const Desc& ds);

   /// sets up basis.
   /** Loads a Descriptor to the basis and sets up the basis matrix and
       all vectors accordingly. The Descriptor must have the same number of
       rows and columns as the currently loaded LP.
   */
   virtual void loadDesc(const Desc&);

   /// sets up linear solver to use.
   /** If destroy is true, solver will be freed inside this object, e.g. in the destructor.
    */
   virtual void loadBasisSolver(SLinSolver<R>* solver, const bool destroy = false);

   /// loads the LP \p lp to the basis.
   /** This involves resetting all counters to 0 and setting up a regular
       default basis consisting of slacks, artificial variables or bounds.
   */
   virtual void load(SPxSolverBase<R>* lp, bool initSlackBasis = true);

   /// unloads the LP from the basis.
   virtual void unLoad()
   {
      theLP = 0;
      setStatus(NO_PROBLEM);
   }

   /// invalidates actual basis.
   /** This method makes the basis matrix and vectors invalid. The basis will
       be reinitialized if needed.
   */
   void invalidate();

   /// Restores initial basis.
   /** This method changes the basis to that present just after loading the LP
       (see addedRows() and addedCols()). This may be necessary if a row or a
       column is changed, since then the current basis may become singular.
   */
   void restoreInitialBasis();

   /// output basis entries.
   void dump();

   /// consistency check.
   bool isConsistent() const;

   /// time spent in updates
   Real getTotalUpdateTime() const
   {
      return theTime->time();
   }
   /// number of updates performed
   int getTotalUpdateCount() const
   {
      return totalUpdateCount;
   }

   /// returns statistical information in form of a string.
   std::string statistics() const
   {
      std::stringstream s;
      s  << factor->statistics()
#ifdef MEASUREUPDATETIME
         << "Updates            : " << std::setw(10) << getTotalUpdateCount() << std::endl
         << "  Time spent       : " << std::setw(10) << getTotalUpdateTime() << std::endl
#endif
         ;

      return s.str();
   }

   void setOutstream(SPxOut& newOutstream)
   {
      spxout = &newOutstream;
   }
   ///@}

   //--------------------------------------
   /**@name Constructors / Destructors */
   ///@{
   /// default constructor.
   SPxBasisBase<R>(Timer::TYPE ttype = Timer::USER_TIME);
   /// copy constructor
   SPxBasisBase<R>(const SPxBasisBase<R>& old);
   /// assignment operator
   SPxBasisBase<R>& operator=(const SPxBasisBase<R>& rhs);
   /// destructor.
   virtual ~SPxBasisBase<R>();
   ///@}


protected:

   //--------------------------------------
   /**@name Protected helpers */
   ///@{
   /// loads \ref soplex::SPxBasisBase<R>::matrix "matrix" according to the SPxId%s stored in \ref soplex::SPxBasisBase<R>::theBaseId "theBaseId".
   /** This method must  be called whenever there is a chance, that the vector
       pointers may have changed due to manipulations of the LP.
   */
   void loadMatrixVecs();

   /// resizes internal arrays.
   /** When a new LP is loaded, the basis matrix and vectors become invalid
       and possibly also of the wrong dimension. Hence, after loading an
       LP, #reDim() is called to reset all arrays etc. accoriding to the
       dimensions of the loaded LP.
   */
   void reDim();

   /// factorizes the basis matrix.
   virtual void factorize();

   /// sets descriptor representation according to loaded LP.
   void setRep();
   ///@}

};


//
// Auxiliary functions.
//

/// Pretty-printing of basis status.
template <class R>
std::ostream& operator<<(std::ostream& os,
                         const typename SPxBasisBase<R>::SPxStatus& status);


/* For backwards compatibility */
typedef SPxBasisBase<Real> SPxBasis;

} // namespace soplex

// General templated definitions
#include "spxbasis.hpp"
#include "spxdesc.hpp"

/* reset the SOPLEX_DEBUG flag to its original value */
#undef SOPLEX_DEBUG
#ifdef SOPLEX_DEBUG_SPXBASIS
#define SOPLEX_DEBUG
#undef SOPLEX_DEBUG_SPXBASIS
#endif

#endif // _SPXBASIS_H_
