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

#include "papilo/core/postsolve/Postsolve.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/postsolve/PostsolveStatus.hpp"
#include <boost/archive/binary_iarchive.hpp>

using namespace papilo;

TEST_CASE( "finding-the-right-value-in-postsolve-for-a-column-fixed-neg-inf",
           "[core]" )
{

   const Num<double> num{};
   Message msg{};
   PostsolveStorage<double> postsolveStorage{};

   std::ifstream inArchiveFile( "./resources/dual_fix_neg_inf.postsolve",
                                std::ios_base::binary );
   boost::archive::binary_iarchive inputArchive( inArchiveFile );
   inputArchive >> postsolveStorage;
   inArchiveFile.close();
   Solution<double> reduced_solution{};
   Solution<double> original_solution{};
   Postsolve<double> postsolve{msg, num};

   REQUIRE( postsolve.undo( reduced_solution, original_solution, postsolveStorage) ==
            PostsolveStatus::kOk );
   papilo::Vec<double> values = original_solution.primal;
   papilo::Vec<double> expected_values{ -11, -5, -5 };
   REQUIRE( values == expected_values );
}

TEST_CASE( "finding-the-right-value-in-postsolve-for-a-column-fixed-pos-inf",
           "[core]" )
{

   const Num<double> num{};
   Message msg{};
   PostsolveStorage<double> postsolveStorage{};

   std::ifstream inArchiveFile( "./resources/dual_fix_pos_inf.postsolve",
                                std::ios_base::binary );
   boost::archive::binary_iarchive inputArchive( inArchiveFile );
   inputArchive >> postsolveStorage;
   inArchiveFile.close();
   Solution<double> reduced_solution{};
   Solution<double> original_solution{};
   Postsolve<double> postsolve{msg, num};

   REQUIRE( postsolve.undo( reduced_solution, original_solution, postsolveStorage ) ==
            PostsolveStatus::kOk );
   papilo::Vec<double> values = original_solution.primal;
   papilo::Vec<double> expected_values{ 13, 9, -5, -2.5 };
   REQUIRE( values == expected_values );

}
