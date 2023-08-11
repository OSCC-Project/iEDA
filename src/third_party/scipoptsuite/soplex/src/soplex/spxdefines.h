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

/**@file  spxdefines.h
 * @brief Debugging, floating point type and parameter definitions.
 *
 * In optimized code with \c NDEBUG defined, only
 * \ref soplex::SPxOut::INFO1 "INFO1",
 * \ref soplex::SPxOut::INFO2 "INFO2", and
 * \ref soplex::SPxOut::INFO3 "INFO3" are set.
 * If \c NDEBUG is not defined, the code within \#TRACE is used.
 * If \c SOPLEX_DEBUG is defined, the code within
 * \ref soplex::SPxOut::DEBUG "DEBUG" is also used.
 *
 * If \c WITH_LONG_DOUBLE is defined, all Real numbers are of type
 * long double instead of just double.
 */
#ifndef _SPXDEFINES_H_
#define _SPXDEFINES_H_
#include <cmath>

#ifdef _MSC_VER
#include <float.h>
#endif

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <iostream>

#include <cstdlib>
#include <memory>

/*
 * include build configuration flags
 */
#ifndef SOPLEX_NO_CONFIG_HEADER
#include "soplex/config.h"
#endif

#ifdef SOPLEX_WITH_BOOST
#include "boost/multiprecision/number.hpp"
#ifdef SOPLEX_WITH_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

#ifdef SOPLEX_WITH_MPFR
// For multiple precision
#include <boost/multiprecision/mpfr.hpp>
#ifndef NDEBUG
#include "boost/multiprecision/debug_adaptor.hpp" // For debuging mpf numbers
#endif // NDEBUG
#endif // SOPLEX_WITH_MPFR
#ifdef SOPLEX_WITH_CPPMPF
#include <boost/multiprecision/cpp_dec_float.hpp>
#endif  // SOPLEX_WITH_CPPMPF

#ifdef SOPLEX_WITH_GMP
#include <boost/multiprecision/gmp.hpp>
#else
#include <boost/multiprecision/cpp_int.hpp>
using mpq_t = double;
#endif

#endif

namespace soplex
{
// Overloaded EQ function
bool EQ(int a, int b);

#define SOPLEX_VERSION         603
#define SOPLEX_SUBVERSION        0
#define SOPLEX_APIVERSION       13
#define SOPLEX_COPYRIGHT       "Copyright (c) 1996-2022 Konrad-Zuse-Zentrum fuer Informationstechnik Berlin (ZIB)"

/*-----------------------------------------------------------------------------
 * Assertion Macros etc.
 *-----------------------------------------------------------------------------
 */

/**
   \brief Macro to turn some assertions into warnings.

   If both \c NDEBUG and \c WITH_WARNINGS are defined then the failed
   assertion is converted to a warning. In all other cases this macro is
   equivalent to assert().

   @param  prefix  Short string for grepping in source code.
   @param  expr    Expression that must be satisfied.
*/
#if defined (NDEBUG) && defined (WITH_WARNINGS)
#define ASSERT_WARN( prefix, expr )                        \
   if ( !( expr ) )                                        \
      {                                                    \
         std::cerr                                         \
         << prefix                                         \
         << " failed assertion on line " << __LINE__       \
         << " in file " << __FILE__ << ": "                \
         << #expr                                          \
         << std::endl;                                     \
      }
#else // just a normal assert
#define ASSERT_WARN( prefix, expr ) ( assert( expr ) )
#endif



/*-----------------------------------------------------------------------------
 * Debugging Macros etc.
 *-----------------------------------------------------------------------------
 */

/**
   Prints/Executes \p stream with verbosity level \p verbosity, resetting
   the old verbosity level afterwards.
   Usually the parameter \p stream prints something out.
   This is an internal define used by MSG_ERROR, MSG_WARNING, etc.
*/
#ifdef DISABLE_VERBOSITY
#define DO_WITH_TMP_VERBOSITY( verbosity, spxout, do_something ) {}
#define DO_WITH_ERR_VERBOSITY( do_something ) {}
#else
#define DO_WITH_TMP_VERBOSITY( verbosity, spxout, do_something ) \
   {                                                             \
     if( &spxout != NULL )                                       \
     {                                                           \
        if( verbosity <= spxout.getVerbosity() )                 \
        {                                                        \
           const SPxOut::Verbosity  old_verbosity = spxout.getVerbosity(); \
           spxout.setVerbosity( verbosity );                     \
           do_something;                                         \
           spxout.setVerbosity( old_verbosity );                 \
        }                                                        \
     }                                                           \
   }
#define DO_WITH_ERR_VERBOSITY( do_something ) { do_something; }
#endif

/// Prints out message \p x if the verbosity level is at least SPxOut::ERROR.
#define MSG_ERROR(x)            { DO_WITH_ERR_VERBOSITY( x ) }
/// Prints out message \p x if the verbosity level is at least SPxOut::WARNING.
#define MSG_WARNING(spxout, x)  { DO_WITH_TMP_VERBOSITY( SPxOut::WARNING, spxout, x ) }
/// Prints out message \p x if the verbosity level is at least SPxOut::INFO1.
#define MSG_INFO1(spxout, x)    { DO_WITH_TMP_VERBOSITY( SPxOut::INFO1, spxout, x ) }
/// Prints out message \p x if the verbosity level is at least SPxOut::INFO2.
#define MSG_INFO2(spxout, x)    { DO_WITH_TMP_VERBOSITY( SPxOut::INFO2, spxout, x ) }
/// Prints out message \p x if the verbosity level is at least SPxOut::INFO3.
#define MSG_INFO3(spxout, x)    { DO_WITH_TMP_VERBOSITY( SPxOut::INFO3, spxout, x ) }

extern bool msginconsistent(const char* name, const char* file, int line);

#define MSGinconsistent(name) msginconsistent(name, __FILE__, __LINE__)

#if defined(SOPLEX_DEBUG)
// print output in any case, regardless of Param::verbose():
#define MSG_DEBUG(x) { x; }
#else
#define MSG_DEBUG(x) /**/
#endif //!SOPLEX_DEBUG


/*-----------------------------------------------------------------------------
 * multi-thread support
 *-----------------------------------------------------------------------------
 */
// enable the user to compile without thread_local by setting USRCXXFLAGS=-DTHREADLOCAL=""
#if !defined(THREADLOCAL)
#if defined(_MSC_VER) && _MSC_VER < 1900
#define THREADLOCAL
#else
#define THREADLOCAL thread_local
#endif
#endif

/*-----------------------------------------------------------------------------
 * Long double support, Parameters and Epsilons
 *-----------------------------------------------------------------------------
 */


#ifdef WITH_LONG_DOUBLE


typedef long double Real;

#ifndef REAL
#define REAL(x)  x##L
#define REAL_FORMAT "Lf"
#endif
/// default allowed bound violation
#ifndef DEFAULT_BND_VIOL
#define DEFAULT_BND_VIOL   1e-12L
#endif
/// default allowed additive zero: 1.0 + EPS_ZERO == 1.0
#ifndef DEFAULT_EPS_ZERO
#define DEFAULT_EPS_ZERO   1e-28L
#endif
/// epsilon for factorization
#ifndef DEFAULT_EPS_FACTOR
#define DEFAULT_EPS_FACTOR 1e-30L
#endif
/// epsilon for factorization update
#ifndef DEFAULT_EPS_UPDATE
#define DEFAULT_EPS_UPDATE 1e-26L
#endif
#ifndef DEFAULT_EPS_PIVOT
#define DEFAULT_EPS_PIVOT 1e-20L
#endif
///
#define DEFAULT_INFINITY   1e100L


#else

#ifdef WITH_FLOAT

typedef float Real;

#ifndef REAL
#define REAL(x)  x
#define REAL_FORMAT "f"
#endif
/// default allowed bound violation
#ifndef DEFAULT_BND_VIOL
#define DEFAULT_BND_VIOL   1e-1f
#endif
/// default allowed additive zero: 1.0 + EPS_ZERO == 1.0
#ifndef DEFAULT_EPS_ZERO
#define DEFAULT_EPS_ZERO   1e-7f
#endif
#ifndef DEFAULT_EPS_FACTOR
#define DEFAULT_EPS_FACTOR 1e-7f
#endif
#ifndef DEFAULT_EPS_UPDATE
#define DEFAULT_EPS_UPDATE 1e-6f
#endif
#ifndef DEFAULT_EPS_PIVOT
#define DEFAULT_EPS_PIVOT 1e-6f
#endif
#define DEFAULT_INFINITY   1e35f

#else

typedef double Real;

#ifndef REAL
#define REAL(x)  x
#define REAL_FORMAT "lf"
#endif
/// default allowed bound violation
#ifndef DEFAULT_BND_VIOL
#define DEFAULT_BND_VIOL   1e-6
#endif
/// default allowed additive zero: 1.0 + EPS_ZERO == 1.0
#ifndef DEFAULT_EPS_ZERO
#define DEFAULT_EPS_ZERO   1e-16
#endif
#ifndef DEFAULT_EPS_FACTOR
#define DEFAULT_EPS_FACTOR 1e-20
#endif
#ifndef DEFAULT_EPS_UPDATE
#define DEFAULT_EPS_UPDATE 1e-16
#endif
#ifndef DEFAULT_EPS_PIVOT
#define DEFAULT_EPS_PIVOT 1e-10
#endif
#define DEFAULT_INFINITY   1e100

#endif // !WITH_FLOAT
#endif // !WITH_LONG_DOUBLE

#define MAXIMUM(x,y)        ((x)>(y) ? (x) : (y))
#define MINIMUM(x,y)        ((x)<(y) ? (x) : (y))

#define SPX_MAXSTRLEN       1024 /**< maximum string length in SoPlex */

THREADLOCAL extern const Real infinity;

class Param
{
private:

   //------------------------------------
   /**@name Data */
   ///@{
   /// default allowed additive zero: 1.0 + EPS_ZERO == 1.0
   THREADLOCAL static Real s_epsilon;
   /// epsilon for factorization
   THREADLOCAL static Real s_epsilon_factorization;
   /// epsilon for factorization update
   THREADLOCAL static Real s_epsilon_update;
   /// epsilon for pivot zero tolerance in factorization
   THREADLOCAL static Real s_epsilon_pivot;
   ///@}

public:

   //------------------------------------
   /**@name Access / modification */
   ///@{
   ///
   static Real epsilon();
   ///
   static void setEpsilon(Real eps);
   ///
   static Real epsilonFactorization();
   ///
   static void setEpsilonFactorization(Real eps);
   ///
   static Real epsilonUpdate();
   ///
   static void setEpsilonUpdate(Real eps);
   ///
   static Real epsilonPivot();
   ///
   static void setEpsilonPivot(Real eps);
   ///@}
};

// A generic version of spxAbs. It would be nice if we could replace spxAbs
// with std::abs. Currently there are different versions of spxAbs under
// compile time #if. It's better to make this an overloaded function. Even
// better, replace it by std::abs since types from boost/multiprecision would
// need no extra modification.
template <class R>
R spxAbs(R a)
{
   return abs(a);
}

// cmath means proper long double function gets called, e.g. for fabs -> fabsl.
// Documentation unclear for nextafterl, so the ifdef remains for that case.
#ifdef WITH_LONG_DOUBLE
// returns the next representable value after x in the direction of y
inline Real spxNextafter(Real x, Real y)
{
   return nextafterl(x, y);
}
#else
// returns the next representable value after x in the direction of y
inline Real spxNextafter(Real x, Real y)
{
#ifndef _MSC_VER
   return nextafter(x, y);
#else
   return _nextafter(x, y);
#endif
}
#endif

/// returns |a|
template <>
inline Real spxAbs(Real a)
{
   return fabs(a);
}

/// returns square root
inline Real spxSqrt(Real a)
{
   return std::sqrt(a);
}

/// returns max(|a|,|b|)
inline Real maxAbs(Real a, Real b)
{
   const Real absa = spxAbs(a);
   const Real absb = spxAbs(b);

   return absa > absb ? absa : absb;
}

/// returns (a-b) / max(|a|,|b|,1.0)
inline Real relDiff(Real a, Real b)
{
   return (a - b) / (maxAbs(a, b) > 1.0 ? maxAbs(a, b) : 1.0);
}

/// safe version of snprintf
inline int spxSnprintf(
   char*                 t,                  /**< target string */
   size_t                len,                /**< length of the string to copy */
   const char*           s,                  /**< source string */
   ...                                       /**< further parameters */
)
{
   va_list ap;
   int n;

   assert(t != NULL);
   assert(len > 0);

   va_start(ap, s); /*lint !e826*/

#if defined(_WIN32) || defined(_WIN64)
   n = _vsnprintf(t, len, s, ap);
#else
   n = vsnprintf(t, len, s, ap); /*lint !e571*/
#endif
   va_end(ap);

   if(n < 0 || (size_t) n >= len)
   {
#ifndef NDEBUG

      if(n < 0)
      {
         MSG_ERROR(std::cerr << "vsnprintf returned " << n << " while reading: " << s << std::endl;)
      }

#endif
      t[len - 1] = '\0';
      n = (int) len - 1;
   }

   return n;
}

#ifdef SOPLEX_WITH_BOOST

using namespace boost::multiprecision;

#ifdef SOPLEX_WITH_GMP
template<boost::multiprecision::expression_template_option eto>
inline number<gmp_rational, eto> ldexp(number<gmp_rational, eto>, int exp)
{
   assert(false);
   return number<gmp_rational>();
}

template<boost::multiprecision::expression_template_option eto>
inline number<gmp_rational, eto> frexp(number<gmp_rational, eto>, int* exp)
{
   assert(false);
   return number<gmp_rational>();
}
#else
inline cpp_rational ldexp(cpp_rational r, int exp)
{
   assert(false);
   return cpp_rational();
}

inline cpp_rational frexp(cpp_rational, int* exp)
{
   assert(false);
   return cpp_rational();
}
#endif

// wrapped frexp function
template <typename T, boost::multiprecision::expression_template_option eto>
boost::multiprecision::number<T, eto> spxFrexp(boost::multiprecision::number<T, eto> y, int* exp)
{
   return frexp(y, exp);
}

// Overloaded spxLdexp
template <typename T, boost::multiprecision::expression_template_option eto>
boost::multiprecision::number<T> spxLdexp(boost::multiprecision::number<T, eto> x, int exp)
{
   return ldexp(x, exp);
}

// Overloaded function to return the square-root
template <typename T, expression_template_option ep>
number<T, ep> spxSqrt(number<T, ep> a)
{
   return sqrt(a);
}

// the nextafter function
template <typename T, expression_template_option eto>
number<T, eto> spxNextafter(number<T, eto> x,
                            number<T, eto> y)
{
   // Turns out that nextafter is not supported in the mpfr library? The mpfr
   // library does a different function named nextabove. Probably a
   // replacement? I've made an issue about this.
   // return nextafter(x,y);

   // @todo Temporarily, I'm returning 0
   assert(false);
   return 0;
}

// Returns the square root
template <typename T>
number<T> spxSqrt(number<T> a)
{
   return sqrt(a);
}

/// returns max(|a|,|b|)
template <typename T, expression_template_option et>
inline number<T, et> maxAbs(
   number<T, et> a, number<T, et> b)
{
   const auto absa = spxAbs(a);
   const auto absb = spxAbs(b);

   return absa > absb ? absa : absb;
}

template <typename T, expression_template_option et>
inline number<T, et> relDiff(number<T, et> a,
                             number<T, et> b)
{
   return (a - b) / (maxAbs(a, b) > 1.0 ? maxAbs(a, b) : 1.0);
}
#endif
using namespace soplex;

} // namespace soplex

// For the templated functions
#include "spxdefines.hpp"

#endif // _SPXDEFINES_H_
