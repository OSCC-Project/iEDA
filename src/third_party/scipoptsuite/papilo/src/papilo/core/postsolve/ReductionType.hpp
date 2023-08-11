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

#ifndef _PAPILO_CORE_REDUCTION_TYPE_HPP_
#define _PAPILO_CORE_REDUCTION_TYPE_HPP_


/// possible types of post solving
enum class ReductionType : int
{
   kFixedCol = 0,
   kFixedInfCol = 5,
   kParallelCol = 2,
   kSubstitutedColWithDual = 3,
   kSubstitutedCol = 1,
   kVarBoundChange = 4,

   kRedundantRow = 7,
   kRowBoundChange = 8,
   kReasonForRowBoundChangeForcedByRow = 9,
   kRowBoundChangeForcedByRow = 10,

   kSaveRow = 11,

   kReducedBoundsCost = 12,
   kColumnDualValue = 13,
   kRowDualValue = 14,
   kCoefficientChange = 15,
};

#endif
