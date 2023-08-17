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

/**@file  vectorbase.h
 * @brief Dense vector.
 */
#ifndef _VECTORBASE_H_
#define _VECTORBASE_H_

#include <assert.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "vector"
#include "algorithm"

#include "soplex/spxdefines.h"
#include "soplex/stablesum.h"
#include "soplex/rational.h"

namespace soplex
{
template < class R > class SVectorBase;
template < class R > class SSVectorBase;

/**@brief   Dense vector.
 * @ingroup Algebra
 *
 *  Class VectorBase provides dense linear algebra vectors. Internally, VectorBase wraps std::vector.
 *
 *  After construction, the values of a VectorBase can be accessed with the subscript operator[]().  Safety is provided by
 *  qchecking of array bound when accessing elements with the subscript operator[]() (only when compiled without \c
 *  -DNDEBUG).
 *
 *  A VectorBase is distinguished from a simple array of %Reals or %Rationals by providing a set of mathematical
 *  operations.
 *
 *  The following mathematical operations are provided by class VectorBase (VectorBase \p a, \p b; R \p x):
 *
 *  <TABLE>
 *  <TR><TD>Operation</TD><TD>Description   </TD><TD></TD>&nbsp;</TR>
 *  <TR><TD>\c -=    </TD><TD>subtraction   </TD><TD>\c a \c -= \c b </TD></TR>
 *  <TR><TD>\c +=    </TD><TD>addition      </TD><TD>\c a \c += \c b </TD></TR>
 *  <TR><TD>\c *     </TD><TD>scalar product</TD>
 *      <TD>\c x = \c a \c * \c b </TD></TR>
 *  <TR><TD>\c *=    </TD><TD>scaling       </TD><TD>\c a \c *= \c x </TD></TR>
 *  <TR><TD>maxAbs() </TD><TD>infinity norm </TD>
 *      <TD>\c a.maxAbs() == \f$\|a\|_{\infty}\f$ </TD></TR>
 *  <TR><TD>minAbs() </TD><TD> </TD>
 *      <TD>\c a.minAbs() == \f$\min |a_i|\f$ </TD></TR>
 *
 *  <TR><TD>length() </TD><TD>euclidian norm</TD>
 *      <TD>\c a.length() == \f$\sqrt{a^2}\f$ </TD></TR>
 *  <TR><TD>length2()</TD><TD>square norm   </TD>
 *      <TD>\c a.length2() == \f$a^2\f$ </TD></TR>
 *  <TR><TD>multAdd(\c x,\c b)</TD><TD>add scaled vector</TD>
 *      <TD> \c a +=  \c x * \c b </TD></TR>
 *  </TABLE>
 *
 *  When using any of these operations, the vectors involved must be of the same dimension.  Also an SVectorBase \c b is
 *  allowed if it does not contain nonzeros with index greater than the dimension of \c a.q
 */
template < class R >
class VectorBase
{

   // VectorBase is a friend of VectorBase of different template type. This is so
   // that we can do conversions.
   template <typename S>
   friend class VectorBase;


protected:

   // ------------------------------------------------------------------------------------------------------------------
   /**@name Data */
   ///@{

   /// Values of vector.
   std::vector<R> val;

   ///@}

public:

   // ------------------------------------------------------------------------------------------------------------------
   /**@name Construction and assignment */
   ///@{

   /// Constructor.
   /** There is no default constructor since the storage for a VectorBase must be provided externally.  Storage must be
    *  passed as a memory block val at construction. It must be large enough to fit at least dimen values.
    */

   // Default constructor
   VectorBase<R>()
   {
      // Default constructor
      ;
   }

   // Construct from pointer, copies the values into the VectorBase
   VectorBase<R>(int dimen, R* p_val)
   {
      val.assign(p_val, p_val + dimen);
   }

   // do not convert int to empty vectorbase
   explicit VectorBase<R>(int p_dimen)
   {
      val.resize(p_dimen);
   }

   // Constructing an element (usually involving casting Real to Rational and
   // vice versa.)
   template <typename S>
   VectorBase<R>(const VectorBase<S>& vec)
   {
      this->operator=(vec);
   }

   // The move constructor
   VectorBase<R>(const VectorBase<R>&& vec)noexcept: val(std::move(vec.val))
   {
   }

   // Copy constructor
   VectorBase<R>(const VectorBase<R>& vec): val(vec.val)
   {
   }


   /// Assignment operator.
   // Supports assignment from a Rational vector to Real and vice versa
   template < class S >
   VectorBase<R>& operator=(const VectorBase<S>& vec)
   {
      if((VectorBase<S>*)this != &vec)
      {
         val.clear();
         val.reserve(vec.dim());

         for(auto& v : vec.val)
         {
            val.push_back(R(v));
         }
      }

      return *this;
   }

   /// Assignment operator.
   VectorBase<R>& operator=(const VectorBase<R>& vec)
   {
      if(this != &vec)
      {
         val.reserve(vec.dim());

         val = vec.val;
      }

      return *this;
   }

   /// Move assignment operator
   VectorBase<R>& operator=(const VectorBase<R>&& vec)
   {
      val = std::move(vec.val);
      return *this;
   }

   /// scale and assign
   VectorBase<R>& scaleAssign(int scaleExp, const VectorBase<R>& vec)
   {
      if(this != &vec)
      {
         assert(dim() == vec.dim());

         auto dimen = dim();

         for(decltype(dimen) i = 0 ; i < dimen; i++)
            val[i] = spxLdexp(vec[i], scaleExp);

      }

      return *this;
   }

   /// scale and assign
   VectorBase<R>& scaleAssign(const int* scaleExp, const VectorBase<R>& vec, bool negateExp = false)
   {
      if(this != &vec)
      {
         assert(dim() == vec.dim());

         if(negateExp)
         {
            auto dimen = dim();

            for(decltype(dimen) i = 0; i < dimen; i++)
               val[i] = spxLdexp(vec[i], -scaleExp[i]);
         }
         else
         {
            auto dimen = dim();

            for(decltype(dimen) i = 0; i < dimen; i++)
               val[i] = spxLdexp(vec[i], scaleExp[i]);
         }

      }

      return *this;
   }


   /// Assignment operator.
   /** Assigning an SVectorBase to a VectorBase using operator=() will set all values to 0 except the nonzeros of \p vec.
    *  This is diffent in method assign().
    */
   template < class S >
   VectorBase<R>& operator=(const SVectorBase<S>& vec);

   /// Assignment operator.
   /** Assigning an SSVectorBase to a VectorBase using operator=() will set all values to 0 except the nonzeros of \p
    *  vec.  This is diffent in method assign().
    */
   /**@todo do we need this also in non-template version, because SSVectorBase can be automatically cast to VectorBase? */
   template < class S >
   VectorBase<R>& operator=(const SSVectorBase<S>& vec);

   /// Assign values of \p vec.
   /** Assigns all nonzeros of \p vec to the vector.  All other values remain unchanged. */
   template < class S >
   VectorBase<R>& assign(const SVectorBase<S>& vec);

   /// Assign values of \p vec.
   /** Assigns all nonzeros of \p vec to the vector.  All other values remain unchanged. */
   template < class S >
   VectorBase<R>& assign(const SSVectorBase<S>& vec);

   ///@}

   // ------------------------------------------------------------------------------------------------------------------
   /**@name Access */
   ///@{

   /// Dimension of vector.
   int dim() const
   {
      return int(val.size());
   }

   /// Return \p n 'th value by reference.
   R& operator[](int n)
   {
      assert(n >= 0 && n < dim());
      return val[n];
   }

   /// Return \p n 'th value.
   const R& operator[](int n) const
   {
      assert(n >= 0 && n < dim());
      return val[n];
   }

   /// Equality operator.
   friend bool operator==(const VectorBase<R>& vec1, const VectorBase<R>& vec2)
   {
      return (vec1.val == vec2.val);
   }

   /// Return underlying std::vector.
   const std::vector<R>& vec()
   {
      return val;
   }

   ///@}

   // ------------------------------------------------------------------------------------------------------------------
   /**@name Arithmetic operations */
   ///@{

   /// Set vector to contain all-zeros (keeping the same length)
   void clear()
   {
      for(auto& v : val)
         v = 0;
   }

   /// Addition.
   template < class S >
   VectorBase<R>& operator+=(const VectorBase<S>& vec)
   {
      assert(dim() == vec.dim());

      auto dimen = dim();

      for(decltype(dimen) i = 0; i < dimen; i++)
         val[i] += vec[i];

      return *this;
   }

   /// Addition.
   template < class S >
   VectorBase<R>& operator+=(const SVectorBase<S>& vec);

   /// Addition.
   template < class S >
   VectorBase<R>& operator+=(const SSVectorBase<S>& vec);

   /// Subtraction.
   template < class S >
   VectorBase<R>& operator-=(const VectorBase<S>& vec)
   {
      assert(dim() == vec.dim());

      auto dimen = dim();

      for(decltype(dimen) i = 0; i < dimen; i++)
         val[i] -= vec[i];

      return *this;
   }

   /// Subtraction.
   template < class S >
   VectorBase<R>& operator-=(const SVectorBase<S>& vec);

   /// Subtraction.
   template < class S >
   VectorBase<R>& operator-=(const SSVectorBase<S>& vec);

   /// Scaling.
   template < class S >
   VectorBase<R>& operator*=(const S& x)
   {

      auto dimen = dim();

      for(decltype(dimen) i = 0; i < dimen; i++)
         val[i] *= x;

      return *this;
   }

   /// Division.
   template < class S >
   VectorBase<R>& operator/=(const S& x)
   {
      assert(x != 0);

      auto dimen = dim();

      for(decltype(dimen) i = 0; i < dimen; i++)
         val[i] /= x;

      return *this;
   }

   /// Inner product.
   R operator*(const VectorBase<R>& vec) const
   {
      StableSum<R> x;

      auto dimen = dim();

      for(decltype(dimen) i = 0; i < dimen; i++)
         x += val[i] * vec.val[i];

      return x;
   }

   /// Inner product.
   R operator*(const SVectorBase<R>& vec) const;

   /// Inner product.
   R operator*(const SSVectorBase<R>& vec) const;

   /// Maximum absolute value, i.e., infinity norm.
   R maxAbs() const
   {
      assert(dim() > 0);

      // A helper function for the std::max_element. Because we compare the absolute value.
      auto absCmpr = [](R a, R b)
      {
         return (spxAbs(a) < spxAbs(b));
      };

      auto maxReference = std::max_element(val.begin(), val.end(), absCmpr);

      R maxi = spxAbs(*maxReference);

      assert(maxi >= 0.0);

      return maxi;
   }

   /// Minimum absolute value.
   R minAbs() const
   {
      assert(dim() > 0);

      // A helper function for the std::min_element. Because we compare the absolute value.
      auto absCmpr = [](R a, R b)
      {
         return (spxAbs(a) < spxAbs(b));
      };

      auto minReference = std::min_element(val.begin(), val.end(), absCmpr);

      R mini = spxAbs(*minReference);

      assert(mini >= 0.0);

      return mini;
   }

   /// Floating point approximation of euclidian norm (without any approximation guarantee).
   R length() const
   {
      return spxSqrt(length2());
   }

   /// Squared norm.
   R length2() const
   {
      return (*this) * (*this);
   }

   /// Addition of scaled vector.
   template < class S, class T >
   VectorBase<R>& multAdd(const S& x, const VectorBase<T>& vec)
   {
      assert(vec.dim() == dim());

      auto dimen = dim();

      for(decltype(dimen) i = 0; i < dimen; i++)
         val[i] += x * vec.val[i];

      return *this;
   }

   /// Addition of scaled vector.
   template < class S, class T >
   VectorBase<R>& multAdd(const S& x, const SVectorBase<T>& vec);

   /// Subtraction of scaled vector.
   template < class S, class T >
   VectorBase<R>& multSub(const S& x, const SVectorBase<T>& vec);

   /// Addition of scaled vector.
   template < class S, class T >
   VectorBase<R>& multAdd(const S& x, const SSVectorBase<T>& vec);

   ///@}

   // ------------------------------------------------------------------------------------------------------------------
   /**@name Utilities */
   ///@{

   /// Conversion to C-style pointer.
   /** This function serves for using a VectorBase in an C-style function. It returns a pointer to the first value of
    *  the array.
    *
    *  @todo check whether this non-const c-style access should indeed be public
    */
   R* get_ptr()
   {
      return val.data();
   }

   /// Conversion to C-style pointer.
   /** This function serves for using a VectorBase in an C-style function. It returns a pointer to the first value of
    *  the array.
    */
   const R* get_const_ptr() const
   {
      return val.data();
   }

   // Provides access to the iterators of std::vector<R> val
   typename std::vector<R>::const_iterator begin() const
   {
      return val.begin();
   }

   typename std::vector<R>::iterator begin()
   {
      return val.begin();
   }

   // Provides access to the iterators of std::vector<R> val
   typename std::vector<R>::const_iterator end() const
   {
      return val.end();
   }

   typename std::vector<R>::iterator end()
   {
      return val.end();
   }

   // Functions from VectorBase

   // This used to be VectorBase's way of having std::vector's capacity. This
   // represents the maximum number of elements the std::vector can have without,
   // needing any more resizing. Bigger than size, mostly.
   int memSize() const
   {
      return int(val.capacity());
   }

   /// Resets \ref soplex::VectorBase "VectorBase"'s dimension to \p newdim.
   void reDim(int newdim, const bool setZero = true)
   {
      if(setZero && newdim > dim())
      {
         // Inserts 0 to the rest of the vectors.
         //
         // TODO: Is this important after the change of raw pointers to
         // std::vector. This is just a waste of operations, I think.
         val.insert(val.end(), newdim - VectorBase<R>::dim(), 0);
      }
      else
      {
         val.resize(newdim);
      }

   }


   /// Resets \ref soplex::VectorBase "VectorBase"'s memory size to \p newsize.
   void reSize(int newsize)
   {
      assert(newsize > VectorBase<R>::dim());

      // Problem: This is not a conventional resize for std::vector. This only
      // updates the capacity, i.e., by pushing elements to the vector after this,
      // there will not be any (internal) resizes.
      val.reserve(newsize);
   }

   // For operations such as vec1 - vec2
   const VectorBase<R> operator-(const VectorBase<R>& vec) const
   {
      assert(vec.dim() == dim());
      VectorBase<R> res;
      res.val.reserve(dim());

      auto dimen = dim();

      for(decltype(dimen) i = 0; i < dimen; i++)
      {
         res.val.push_back(val[i] - vec[i]);
      }

      return res;
   }

   // Addition
   const VectorBase<R> operator+(const VectorBase<R>& v) const
   {
      assert(v.dim() == dim());
      VectorBase<R> res;
      res.val.reserve(dim());

      auto dimen = dim();

      for(decltype(dimen) i = 0; i < dimen; i++)
      {
         res.val.push_back(val[i] + v[i]);
      }

      return res;
   }

   // The negation operator. e.g. -vec1;
   friend VectorBase<R> operator-(const VectorBase<R>& vec)
   {
      VectorBase<R> res;

      res.val.reserve(vec.dim());

      for(auto& v : vec.val)
      {
         res.val.push_back(-(v));
      }

      return res;
   }


   ///@}
   /// Consistency check.
   bool isConsistent() const
   {
      return true;
   }

};

/// Inner product.
template<>
inline
Rational VectorBase<Rational>::operator*(const VectorBase<Rational>& vec) const
{
   assert(vec.dim() == dim());

   if(dim() <= 0 || vec.dim() <= 0)
      return Rational();

   Rational x = val[0];
   x *= vec.val[0];

   auto dimen = dim();

   for(decltype(dimen) i = 1; i < dimen; i++)
      x += val[i] * vec.val[i];

   return x;
}

} // namespace soplex
#endif // _VECTORBASE_H_
