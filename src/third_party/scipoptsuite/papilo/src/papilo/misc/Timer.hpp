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

#ifndef _PAPILO_MISC_TIMER_HPP_
#define _PAPILO_MISC_TIMER_HPP_

#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#else
#include <chrono>
#endif

namespace papilo
{

#ifdef PAPILO_TBB
class Timer
{
 public:
   Timer( double& time_ ) : time( time_ ) { start = tbb::tick_count::now(); }

   double
   getTime() const
   {
      return ( tbb::tick_count::now() - start ).seconds();
   }

   ~Timer() { time += ( tbb::tick_count::now() - start ).seconds(); }

 private:
   tbb::tick_count start;
   double& time;
};
#else
class Timer
{
 public:
   Timer( double& time_ ) : time( time_ ) { start = std::chrono::steady_clock::now(); }

   double
   getTime() const
   {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start )
                 .count() /1000.0;
   }

   ~Timer() {
      time += std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - start )
                  .count() /1000.0;
   }

 private:
   std::chrono::steady_clock::time_point start;
   double& time;
};
#endif

} // namespace papilo

#endif
