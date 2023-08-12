/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*               This file is part of the program and library                */
/*    PaPILO --- Parallel Presolve for Integer and Linear Optimization       */
/*                                                                           */
/* Copyright (C) 2020-2022 Konrad-Zuse-Zentrum                               */
/*                     fuer Informationstechnik Berlin                       */
/*                                                                           */
/* This program is free software: you can redistribute it and/or modify      */
/* it under the terms of the GNU Lesser General Public License as published  */
/* by the Free Software Foundation, either version 3 of the License, or      */
/* (at your option) any later version.                                       */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program.  If not, see <https://www.gnu.org/licenses/>.    */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _PAPILO_MISC_NUM_HPP_
#define _PAPILO_MISC_NUM_HPP_

#include "papilo/misc/ParameterSet.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace papilo
{

template <typename T>
struct num_traits
{
   constexpr static bool is_rational =
       std::numeric_limits<T>::is_specialized == true &&
       std::numeric_limits<T>::is_integer == false &&
       std::numeric_limits<T>::is_exact == true &&
       std::numeric_limits<T>::min_exponent == 0 &&
       std::numeric_limits<T>::max_exponent == 0 &&
       std::numeric_limits<T>::min_exponent10 == 0 &&
       std::numeric_limits<T>::max_exponent10 == 0;

   constexpr static bool is_floating_point =
       std::numeric_limits<T>::is_specialized == true &&
       std::numeric_limits<T>::is_integer == false &&
       std::numeric_limits<T>::is_exact == false &&
       std::numeric_limits<T>::min_exponent != 0 &&
       std::numeric_limits<T>::max_exponent != 0 &&
       std::numeric_limits<T>::min_exponent10 != 0 &&
       std::numeric_limits<T>::max_exponent10 != 0;

   constexpr static bool is_integer =
       std::numeric_limits<T>::is_specialized == true &&
       std::numeric_limits<T>::is_integer == true &&
       std::numeric_limits<T>::is_exact == true &&
       std::numeric_limits<T>::min_exponent == 0 &&
       std::numeric_limits<T>::max_exponent == 0 &&
       std::numeric_limits<T>::min_exponent10 == 0 &&
       std::numeric_limits<T>::max_exponent10 == 0;
};

template <typename T>
class provides_numerator_and_denominator_overloads
{
   struct no
   {
   };

   template <typename T2>
   static decltype( numerator( std::declval<T2>() ) )
   test_numerator( int );

   template <typename T2>
   static decltype( numerator( std::declval<T2>() ) )
   test_denominator( int );

   template <typename T2>
   static no
   test_numerator( ... );

   template <typename T2>
   static no
   test_denominator( ... );

 public:
   constexpr static bool value =
       num_traits<decltype( test_numerator<T>( 0 ) )>::is_integer &&
       num_traits<decltype( test_denominator<T>( 0 ) )>::is_integer;
};

template <typename Rational,
          typename std::enable_if<
              num_traits<Rational>::is_rational &&
                  provides_numerator_and_denominator_overloads<Rational>::value,
              int>::type = 0>
Rational
floor( const Rational& x, ... )
{
   if( x >= 0 )
      return numerator( x ) / denominator( x );

   if( numerator( x ) < 0 )
      return -1 + ( numerator( x ) + 1 ) / denominator( x );

   return -1 + ( numerator( x ) - 1 ) / denominator( x );
}

template <typename Rational,
          typename std::enable_if<
              num_traits<Rational>::is_rational &&
                  provides_numerator_and_denominator_overloads<Rational>::value,
              int>::type = 0>
Rational
ceil( const Rational& x, ... )
{
   if( x <= 0 )
      return numerator( x ) / denominator( x );

   if( numerator( x ) < 0 )
      return 1 + ( numerator( x ) + 1 ) / denominator( x );

   return 1 + ( numerator( x ) - 1 ) / denominator( x );
}

using std::abs;
using std::ceil;
using std::copysign;
using std::exp;
using std::floor;
using std::frexp;
using std::ldexp;
using std::log;
using std::log2;
using std::pow;
using std::sqrt;

template <typename REAL, bool exact = std::numeric_limits<REAL>::is_exact>
struct DefaultTolerances;

template <typename REAL>
struct DefaultTolerances<REAL, false>
{
   static constexpr REAL
   epsilon()
   {
      return feasTol() *
             pow( REAL{ 10 }, -( std::numeric_limits<REAL>::digits10 / 4 ) );
   }

   static constexpr REAL
   feasTol()
   {
      return pow( REAL{ 10 },
                  -( std::numeric_limits<REAL>::digits10 / 2 - 1 ) );
   }
};

template <typename REAL>
struct DefaultTolerances<REAL, true>
{
   static constexpr REAL
   epsilon()
   {
      return REAL{ 0 };
   }

   static constexpr REAL
   feasTol()
   {
      return REAL{ 0 };
   }
};

template <typename REAL>
class Num
{
 public:
   Num()
       : epsilon( REAL{ 1e-9 } ), feastol( REAL{ 1e-6 } ),
         hugeval( REAL{ 1e8 } )
   {
   }

   template <typename R>
   static constexpr REAL
   round( const R& x )
   {
      return floor( REAL( x + REAL( 0.5 ) ) );
   }

   template <typename R1, typename R2>
   bool
   isEq( const R1& a, const R2& b ) const
   {
      return abs( a - b ) <= epsilon;
   }

   template <typename R1, typename R2>
   bool
   isFeasEq( const R1& a, const R2& b ) const
   {
      return abs( a - b ) <= feastol;
   }

   template <typename R1, typename R2>
   bool
   isGE( const R1& a, const R2& b ) const
   {
      return a - b >= -epsilon;
   }

   template <typename R1, typename R2>
   bool
   isFeasGE( const R1& a, const R2& b ) const
   {
      return a - b >= -feastol;
   }

   template <typename R1, typename R2>
   bool
   isLE( const R1& a, const R2& b ) const
   {
      return a - b <= epsilon;
   }

   template <typename R1, typename R2>
   bool
   isFeasLE( const R1& a, const R2& b ) const
   {
      return a - b <= feastol;
   }

   template <typename R1, typename R2>
   bool
   isGT( const R1& a, const R2& b ) const
   {
      return a - b > epsilon;
   }

   template <typename R1, typename R2>
   bool
   isFeasGT( const R1& a, const R2& b ) const
   {
      return a - b > feastol;
   }

   template <typename R1, typename R2>
   bool
   isLT( const R1& a, const R2& b ) const
   {
      return a - b < -epsilon;
   }

   template <typename R1, typename R2>
   bool
   isFeasLT( const R1& a, const R2& b ) const
   {
      return a - b < -feastol;
   }

   template <typename R1, typename R2>
   static REAL
   max( const R1& a, const R2& b )
   {
      return a > b ? REAL( a ) : REAL( b );
   }

   template <typename R1, typename R2>
   static REAL
   min( const R1& a, const R2& b )
   {
      return a < b ? REAL( a ) : REAL( b );
   }

   template <typename R1, typename R2>
   static REAL
   relDiff( const R1& a, const R2& b )
   {
      return ( a - b ) / max( max( abs( a ), abs( b ) ), 1 );
   }

   template <typename R1, typename R2>
   bool
   isRelEq( const R1& a, const R2& b ) const
   {
      return abs( relDiff( a, b ) ) <= epsilon;
   }

   template <typename R1, typename R2>
   bool
   isFeasRelEq( const R1& a, const R2& b ) const
   {
      return abs( relDiff( a, b ) ) <= feastol;
   }

   template <typename R1, typename R2>
   bool
   isRelGE( const R1& a, const R2& b ) const
   {
      return relDiff( a, b ) >= -epsilon;
   }

   template <typename R1, typename R2>
   bool
   isFeasRelGE( const R1& a, const R2& b ) const
   {
      return relDiff( a, b ) >= -feastol;
   }

   template <typename R1, typename R2>
   bool
   isRelLE( const R1& a, const R2& b ) const
   {
      return relDiff( a, b ) <= epsilon;
   }

   template <typename R1, typename R2>
   bool
   isFeasRelLE( const R1& a, const R2& b ) const
   {
      return relDiff( a, b ) <= feastol;
   }

   template <typename R1, typename R2>
   bool
   isRelGT( const R1& a, const R2& b ) const
   {
      return relDiff( a, b ) > epsilon;
   }

   template <typename R1, typename R2>
   bool
   isFeasRelGT( const R1& a, const R2& b ) const
   {
      return relDiff( a, b ) > feastol;
   }

   template <typename R1, typename R2>
   bool
   isRelLT( const R1& a, const R2& b ) const
   {
      return relDiff( a, b ) < -epsilon;
   }

   template <typename R1, typename R2>
   bool
   isFeasRelLT( const R1& a, const R2& b ) const
   {
      return relDiff( a, b ) < -feastol;
   }

   template <typename R1, typename R2>
   static constexpr bool
   isSafeEq( const R1& a, const R2& b )
   {
      return !num_traits<REAL>::is_floating_point ||
             ( abs( relDiff( a, b ) ) <=
               ( 1024 * std::numeric_limits<REAL>::epsilon() ) );
   }

   template <typename R1, typename R2>
   static constexpr bool
   isSafeGE( const R1& a, const R2& b )
   {
      return !num_traits<REAL>::is_floating_point ||
             ( relDiff( a, b ) >=
               -( 1024 * std::numeric_limits<REAL>::epsilon() ) );
   }

   template <typename R1, typename R2>
   static constexpr bool
   isSafeLE( const R1& a, const R2& b )
   {
      return num_traits<REAL>::is_floating_point
                 ? ( relDiff( a, b ) <=
                     ( 1024 * std::numeric_limits<REAL>::epsilon() ) )
                 : true;
   }

   template <typename R1, typename R2>
   static constexpr bool
   isSafeGT( const R1& a, const R2& b )
   {
      return !num_traits<REAL>::is_floating_point ||
             ( relDiff( a, b ) >
               ( 1024 * std::numeric_limits<REAL>::epsilon() ) );
   }

   template <typename R1, typename R2>
   bool static constexpr isSafeLT( const R1& a, const R2& b )
   {
      return !num_traits<REAL>::is_floating_point ||
             ( relDiff( a, b ) <
               -( 1024 * std::numeric_limits<REAL>::epsilon() ) );
   }

   template <typename R>
   REAL
   feasCeil( const R& a ) const
   {
      return ceil( REAL( a - feastol ) );
   }

   template <typename R>
   REAL
   epsCeil( const R& a ) const
   {
      return ceil( REAL( a - epsilon ) );
   }

   template <typename R>
   REAL
   feasFloor( const R& a ) const
   {
      return floor( REAL( a + feastol ) );
   }

   template <typename R>
   REAL
   epsFloor( const R& a ) const
   {
      return floor( REAL( a + epsilon ) );
   }

   template <typename R>
   bool
   isIntegral( const R& a ) const
   {
      return isEq( a, round( a ) );
   }

   template <typename R>
   bool
   isFeasIntegral( const R& a ) const
   {
      return isFeasEq( a, round( a ) );
   }

   template <typename R>
   bool
   isZero( const R& a ) const
   {
      return abs( a ) <= epsilon;
   }

   template <typename R>
   bool
   isFeasZero( const R& a ) const
   {
      return abs( a ) <= feastol;
   }

   static std::size_t
   hashCode( const REAL& xval )
   {
      int theexp;
      double x = static_cast<double>( xval );

      // normalize the value
      double val = frexp( x, &theexp );

      // include the sign bit and the value in the upper 16 bits, i.e. use the
      // upper 15 bits of the mantissa. Cast to signed 16 bit int then to
      // unsigned 16 bit int to have the wrap-around for negative numbers at
      // 2^16.
      uint16_t upperhalf =
          static_cast<uint16_t>( static_cast<int16_t>( ldexp( val, 14 ) ) );

      // in the lower 16 bits include the exponent
      uint16_t lowerhalf = static_cast<uint16_t>( theexp );

      // compose the halfs into one 32 bit integral value and add to state
      return ( static_cast<std::size_t>( upperhalf ) << 16 ) |
             static_cast<std::size_t>( lowerhalf );
   }

   const REAL&
   getEpsilon() const
   {
      return epsilon;
   }

   const REAL&
   getFeasTol() const
   {
      return feastol;
   }

   const REAL&
   getHugeVal() const
   {
      return hugeval;
   }

   template <typename R>
   bool
   isHugeVal( const R& a ) const
   {
      return abs( a ) >= hugeval;
   }

   void
   setEpsilon( REAL value )
   {
      assert( value >= 0 );
      this->epsilon = value;
   }

   void
   setFeasTol( REAL value )
   {
      assert( value >= 0 );
      this->feastol = value;
   }

   void
   setHugeVal( REAL value )
   {
      assert( value >= 0 );
      this->hugeval = value;
   }

   template <typename Archive>
   void
   serialize( Archive& ar, const unsigned int version )
   {
      ar& epsilon;
      ar& feastol;
      ar& hugeval;
   }

 private:
   REAL epsilon;
   REAL feastol;
   REAL hugeval;
};

} // namespace papilo

#endif
