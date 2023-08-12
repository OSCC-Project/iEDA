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

#ifndef _PAPILO_MISC_MULTIPRECISION_HPP_
#define _PAPILO_MISC_MULTIPRECISION_HPP_

#include "papilo/Config.hpp"

// work around build failure with boost on Fedora 37
#include <memory>
#include <boost/serialization/split_free.hpp>

#ifdef PAPILO_HAVE_FLOAT128
#include <boost/multiprecision/float128.hpp>
namespace papilo
{
using Quad = boost::multiprecision::float128;
} // namespace papilo
#elif defined( PAPILO_HAVE_GMP )
#include <boost/multiprecision/gmp.hpp>

namespace papilo
{
using Quad =
    boost::multiprecision::number<boost::multiprecision::gmp_float<35>>;
} // namespace papilo
BOOST_SERIALIZATION_SPLIT_FREE( papilo::Quad )

#else
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/serialization/nvp.hpp>
namespace papilo
{
using Quad = boost::multiprecision::cpp_bin_float_quad;
} // namespace papilo
#endif

#ifdef PAPILO_HAVE_GMP
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/serialization/nvp.hpp>

// unfortunately the multiprecision gmp types do not provide an overload for
// serialization
namespace papilo
{
using Rational = boost::multiprecision::mpq_rational;
using Float100 = boost::multiprecision::mpf_float_100;
using Float500 = boost::multiprecision::mpf_float_500;
using Float1000 = boost::multiprecision::mpf_float_1000;
} // namespace papilo

BOOST_SERIALIZATION_SPLIT_FREE( papilo::Rational )
BOOST_SERIALIZATION_SPLIT_FREE( papilo::Float100 )
BOOST_SERIALIZATION_SPLIT_FREE( papilo::Float500 )
BOOST_SERIALIZATION_SPLIT_FREE( papilo::Float1000 )

namespace boost
{
namespace serialization
{

template <class Archive>
void
save( Archive& ar, const papilo::Rational& num, const unsigned int version )
{
   boost::multiprecision::cpp_rational t( num );
   ar& t;
}

template <class Archive>
void
load( Archive& ar, papilo::Rational& num, const unsigned int version )
{
   boost::multiprecision::cpp_rational t;
   ar& t;
   num = papilo::Rational( t );
}

template <class Archive, unsigned M>
void
save( Archive& ar,
      const boost::multiprecision::number<boost::multiprecision::gmp_float<M>>&
          num,
      const unsigned int version )
{
   boost::multiprecision::number<boost::multiprecision::cpp_bin_float<M>> t(
       num );
   ar& t;
}

template <class Archive, unsigned M>
void
load( Archive& ar,
      boost::multiprecision::number<boost::multiprecision::gmp_float<M>>& num,
      const unsigned int version )
{
   boost::multiprecision::number<boost::multiprecision::cpp_bin_float<M>> t;
   ar& t;
   num =
       boost::multiprecision::number<boost::multiprecision::gmp_float<M>>( t );
}

} // namespace serialization
} // namespace boost

#else
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/serialization/nvp.hpp>

namespace papilo
{
using Rational = boost::multiprecision::cpp_rational;
using Float100 =
    boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100>>;
using Float500 =
    boost::multiprecision::number<boost::multiprecision::cpp_bin_float<500>>;
using Float1000 =
    boost::multiprecision::number<boost::multiprecision::cpp_bin_float<1000>>;
} // namespace papilo
#endif

#endif
