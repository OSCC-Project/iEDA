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

#ifndef _PAPILO_CORE_VARIABLE_DOMAINS_HPP_
#define _PAPILO_CORE_VARIABLE_DOMAINS_HPP_

#include "papilo/misc/Flags.hpp"
#include "papilo/misc/MultiPrecision.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/misc/compress_vector.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif

namespace papilo
{

enum class ColFlag : uint8_t
{
   kNone = 0,
   kLbInf = 1 << 0,
   kLbHuge = 1 << 1,
   kUbInf = 1 << 2,
   kUbHuge = 1 << 3,
   kIntegral = 1 << 4,
   kFixed = 1 << 5,
   kSubstituted = 1 << 6,
   kImplInt = 1 << 7,
   kUnbounded = static_cast<uint8_t>( ColFlag::kLbInf ) |
                static_cast<uint8_t>( ColFlag::kUbInf ),
   kInactive = static_cast<uint8_t>( ColFlag::kFixed ) |
               static_cast<uint8_t>( ColFlag::kSubstituted ),
   kLbUseless = static_cast<uint8_t>( ColFlag::kLbInf ) |
                static_cast<uint8_t>( ColFlag::kLbHuge ),
   kUbUseless = static_cast<uint8_t>( ColFlag::kUbInf ) |
                static_cast<uint8_t>( ColFlag::kUbHuge ),
};

using ColFlags = Flags<ColFlag>;

/// Type to store the domains for the variables in a problem.
/// This includes the lower and upper bounds, and whether the
/// variable is constraint to integral values.
template <typename REAL>
struct VariableDomains
{
   Vec<REAL> lower_bounds;
   Vec<REAL> upper_bounds;
   Vec<ColFlags> flags;

   void
   compress( const Vec<int>& colmapping, bool full = false );

   bool
   isBinary( int col ) const
   {
      return flags[col].test( ColFlag::kIntegral ) &&
             !flags[col].test( ColFlag::kLbUseless, ColFlag::kUbUseless,
                               ColFlag::kInactive ) &&
             lower_bounds[col] == 0 && upper_bounds[col] == 1;
   }

   template <typename Archive>
   void
   serialize( Archive& ar, const unsigned int version )
   {
      ar& lower_bounds;
      ar& upper_bounds;
      ar& flags;
   }
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template struct VariableDomains<double>;
extern template struct VariableDomains<Quad>;
extern template struct VariableDomains<Rational>;
#endif

template <typename REAL>
void
VariableDomains<REAL>::compress( const Vec<int>& colmapping, bool full )
{
#ifdef PAPILO_TBB
   tbb::parallel_invoke(
       [this, &colmapping, full]() {
          compress_vector( colmapping, lower_bounds );
          if( full )
             lower_bounds.shrink_to_fit();
       },
       [this, &colmapping, full]() {
          compress_vector( colmapping, upper_bounds );
          if( full )
             upper_bounds.shrink_to_fit();
       },
       [this, &colmapping, full]() {
          compress_vector( colmapping, flags );
          if( full )
             flags.shrink_to_fit();
       } );
#else
   compress_vector( colmapping, lower_bounds );
   compress_vector( colmapping, upper_bounds );
   compress_vector( colmapping, flags );
   if( full )
   {
      flags.shrink_to_fit();
      upper_bounds.shrink_to_fit();
      lower_bounds.shrink_to_fit();
   }

#endif
}
} // namespace papilo

#endif
