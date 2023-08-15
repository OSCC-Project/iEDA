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

/**@file  spxscaler.h
 * @brief LP scaling base class.
 */
#ifndef _SPXSCALER_H_
#define _SPXSCALER_H_

#include <assert.h>

#include "soplex/spxdefines.h"
#include "soplex/dataarray.h"
#include "soplex/vector.h"
#include "soplex/svector.h"
#include "soplex/svset.h"
#include "soplex/dsvector.h"
#include "soplex/dvector.h"
#include <vector>

namespace soplex
{

template < class R >
class SPxLPBase;
/**@brief   LP scaler abstract base class.
   @ingroup Algo

   Instances of classes derived from SPxScaler may be loaded to SoPlex in
   order to scale LPs before solving them. SoPlex will load() itself to
   the SPxScaler and then call #scale(). Generally any SPxLP can be
   loaded to a SPxScaler for #scale()%ing it. The scaling can
   be undone by calling unscale().

   Mathematically, the scaling of a constraint matrix A can be written
   as \f$ A' = R A C \f$, with \f$ R \f$ and \f$ C \f$, being diagonal matrices
   corresponding to the row and column scale factors, respectively. Besides the
   constraints matrix, also the upper and lower bounds of both columns and rows
   need to be scaled.

   Note that by default scaling is performed both before and after presolving and
   the former scaling factors are retained during branch-and-bound (persistent scaling).
   However, while within SoPlex the scaled problem is used, data accessed through
   the soplex.cpp interface is provided w.r.t. the original problem (i.e., in unscaled form).
   For instance, consider a scaled constraints matrix A' that is extended by artificial slack
   variables to the matrix (A',I).
   A basis \f$ B' = [(A',I)P]_{[1:m][1:m] }\f$ (with P being a permutation matrix)
   for the scaled problem corresponds to the basis
   \f$ B = R^{-1} [(A',I)P]_{[1:m][1:m]} [P^{T} \tilde{C}^{-1} P]_{[1:m][1:m] } \f$. In
   this equation, \f$ \tilde{C} \f$ is of the form

   \f[
    \begin{array}{cc}
         C & 0 \\
         O & R^{-1}
   \end{array}
    \f]

   Note that in SoPlex only scaling factors \f$ 2^k, k \in \mathbb{Z} \f$ are used.


*/

template <class R>
class SPxScaler
{
protected:

   //-------------------------------------
   /**@name Data */
   ///@{
   const char*        m_name;      ///< Name of the scaler
   DataArray < int >* m_activeColscaleExp; ///< pointer to currently active column scaling factors
   DataArray < int >* m_activeRowscaleExp; ///< pointer to currently active row scaling factors
   bool               m_colFirst;  ///< do column scaling first
   bool               m_doBoth;    ///< do columns and rows
   SPxOut*            spxout;      ///< message handler
   ///@}

   //-------------------------------------
   /**@name Protected helpers */
   ///@{

   /// clear and setup scaling arrays in the LP
   virtual void setup(SPxLPBase<R>& lp);
   ///@}

public:

   /// compute a single scaling vector , e.g. of a newly added row
   virtual int computeScaleExp(const SVectorBase<R>& vec, const DataArray<int>& oldScaleExp) const;

   // The following is now redundant because of the above function.
   // virtual int computeScaleExp(const SVectorBase<Rational>& vec, const DataArray<int>& oldScaleExp) const;

   /// applies m_colscale and m_rowscale to the \p lp.
   virtual void applyScaling(SPxLPBase<R>& lp);


   template <class T>
   friend std::ostream& operator<<(std::ostream& s, const SPxScaler<T>& sc);

   //-------------------------------------
   /**@name Construction / destruction */
   ///@{
   /// constructor
   explicit SPxScaler(const char* name, bool colFirst = false, bool doBoth = true,
                      SPxOut* spxout = NULL);
   /// copy constructor
   SPxScaler(const SPxScaler&);
   /// assignment operator
   SPxScaler& operator=(const SPxScaler&);
   /// destructor.
   virtual ~SPxScaler();
   /// clone function for polymorphism
   virtual SPxScaler* clone() const = 0;
   ///@}

   //-------------------------------------
   /**@name Access / modification */
   ///@{
   /// get name of scaler
   virtual const char* getName() const;
   /// set scaling order
   virtual void setOrder(bool colFirst);
   /// set wether column and row scaling should be performed
   virtual void setBoth(bool both);
   /// set message handler
   virtual void setOutstream(SPxOut& newOutstream)
   {
      spxout = &newOutstream;
   }
   /// set R parameter
   virtual void setRealParam(R param, const char* name = "realparam");
   /// set int parameter
   virtual void setIntParam(int param, const char* name = "intparam");
   ///@}

   //-------------------------------------
   /**@name Scaling */
   ///@{
   /// scale SPxLP.
   virtual void scale(SPxLPBase<R>& lp, bool persistent = true) = 0;
   /// unscale SPxLP
   virtual void unscale(SPxLPBase<R>& lp);
   /// returns scaling factor for column \p i
   virtual int getColScaleExp(int i) const;
   /// returns scaling factor for row \p i
   virtual int getRowScaleExp(int i) const;
   /// gets unscaled column \p i
   virtual void getColUnscaled(const SPxLPBase<R>& lp, int i, DSVectorBase<R>& vec) const;
   /// returns maximum absolute value of unscaled column \p i
   virtual R getColMaxAbsUnscaled(const SPxLPBase<R>& lp, int i) const;
   /// returns minumum absolute value of unscaled column \p i
   virtual R getColMinAbsUnscaled(const SPxLPBase<R>& lp, int i) const;
   /// returns unscaled upper bound \p i
   virtual R upperUnscaled(const SPxLPBase<R>& lp, int i) const;
   /// returns unscaled upper bound vector of \p lp
   virtual void getUpperUnscaled(const SPxLPBase<R>& lp, VectorBase<R>& vec) const;
   /// returns unscaled lower bound \p i
   virtual R lowerUnscaled(const SPxLPBase<R>& lp, int i) const;
   /// gets unscaled lower bound vector
   virtual void getLowerUnscaled(const SPxLPBase<R>& lp, VectorBase<R>& vec) const;
   /// returns unscaled objective function coefficient of \p i
   virtual R maxObjUnscaled(const SPxLPBase<R>& lp, int i) const;
   /// gets unscaled objective function
   virtual void getMaxObjUnscaled(const SPxLPBase<R>& lp, VectorBase<R>& vec) const;
   /// returns unscaled row \p i
   virtual void getRowUnscaled(const SPxLPBase<R>& lp, int i, DSVectorBase<R>& vec) const;
   /// returns maximum absolute value of unscaled row \p i
   virtual R getRowMaxAbsUnscaled(const SPxLPBase<R>& lp, int i) const;
   /// returns minimum absolute value of unscaled row \p i
   virtual R getRowMinAbsUnscaled(const SPxLPBase<R>& lp, int i) const;
   /// returns unscaled right hand side \p i
   virtual R rhsUnscaled(const SPxLPBase<R>& lp, int i) const;
   /// gets unscaled right hand side vector
   virtual void getRhsUnscaled(const SPxLPBase<R>& lp, VectorBase<R>& vec) const;
   /// returns unscaled left hand side \p i of \p lp
   virtual R lhsUnscaled(const SPxLPBase<R>& lp, int i) const;
   /// returns unscaled left hand side vector of \p lp
   virtual void getLhsUnscaled(const SPxLPBase<R>& lp, VectorBase<R>& vec) const;
   /// returns unscaled coefficient of \p lp
   virtual R getCoefUnscaled(const SPxLPBase<R>& lp, int row, int col) const;
   /// unscale dense primal solution vector given in \p x.
   virtual void unscalePrimal(const SPxLPBase<R>& lp, VectorBase<R>& x) const;
   /// unscale dense slack vector given in \p s.
   virtual void unscaleSlacks(const SPxLPBase<R>& lp, VectorBase<R>& s) const;
   /// unscale dense dual solution vector given in \p pi.
   virtual void unscaleDual(const SPxLPBase<R>& lp, VectorBase<R>& pi) const;
   /// unscale dense reduced cost vector given in \p r.
   virtual void unscaleRedCost(const SPxLPBase<R>& lp, VectorBase<R>& r) const;
   /// unscale primal ray given in \p ray.
   virtual void unscalePrimalray(const SPxLPBase<R>& lp, VectorBase<R>& ray) const;
   /// unscale dual ray given in \p ray.
   virtual void unscaleDualray(const SPxLPBase<R>& lp, VectorBase<R>& ray) const;
   /// apply scaling to objective function vector \p origObj.
   virtual void scaleObj(const SPxLPBase<R>& lp, VectorBase<R>& origObj) const;
   /// returns scaled objective function coefficient \p origObj.
   virtual R scaleObj(const SPxLPBase<R>& lp, int i, R origObj) const;
   /// returns scaled LP element in \p row and \p col.
   virtual R scaleElement(const SPxLPBase<R>& lp, int row, int col, R val) const;
   /// returns scaled lower bound of column \p col.
   virtual R scaleLower(const SPxLPBase<R>& lp, int col, R lower) const;
   /// returns scaled upper bound of column \p col.
   virtual R scaleUpper(const SPxLPBase<R>& lp, int col, R upper) const;
   /// returns scaled left hand side of row \p row.
   virtual R scaleLhs(const SPxLPBase<R>& lp, int row, R lhs) const;
   /// returns scaled right hand side of row \p row.
   virtual R scaleRhs(const SPxLPBase<R>& lp, int row, R rhs) const;
   /// absolute smallest column scaling factor
   virtual R minAbsColscale() const;
   /// absolute biggest column scaling factor
   virtual R maxAbsColscale() const;
   /// absolute smallest row scaling factor
   virtual R minAbsRowscale() const;
   /// absolute biggest row scaling factor
   virtual R maxAbsRowscale() const;
   /// maximum ratio between absolute biggest and smallest element in any column.
   virtual R maxColRatio(const SPxLPBase<R>& lp) const;
   /// maximum ratio between absolute biggest and smallest element in any row.
   virtual R maxRowRatio(const SPxLPBase<R>& lp) const;
   /// round vector entries to power of 2
   void computeExpVec(const std::vector<R>& vec, DataArray<int>& vecExp);
   ///@}

   //-------------------------------------
   /**@name Debugging */
   ///@{
   /// consistency check
   virtual bool isConsistent() const;
   ///@}
};
} // namespace soplex

// General templated definitions
#include "spxscaler.hpp"

#endif // _SPXSCALER_H_
