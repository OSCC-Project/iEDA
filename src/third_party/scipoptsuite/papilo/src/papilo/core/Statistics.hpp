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

#ifndef _PAPILO_CORE_STATISTICS_HPP_
#define _PAPILO_CORE_STATISTICS_HPP_

namespace papilo
{

struct Statistics
{
   double presolvetime;
   int ntsxapplied;
   int ntsxconflicts;
   int nboundchgs;
   int nsidechgs;
   int ncoefchgs;
   int nrounds;
   int ndeletedcols;
   int ndeletedrows;

   Statistics( double _presolvetime, int _ntsxapplied, int _ntsxconflicts,
               int _nboundchgs, int _nsidechgs, int _ncoefchgs, int _nrounds,
               int _ndeletedcols, int _ndeletedrows )
       : presolvetime( _presolvetime ), ntsxapplied( _ntsxapplied ),
         ntsxconflicts( _ntsxconflicts ), nboundchgs( _nboundchgs ),
         nsidechgs( _nsidechgs ), ncoefchgs( _ncoefchgs ), nrounds( _nrounds ),
         ndeletedcols( _ndeletedcols ), ndeletedrows( _ndeletedrows )
   {
   }

   Statistics()
       : presolvetime( 0.0 ), ntsxapplied( 0 ), ntsxconflicts( 0 ),
         nboundchgs( 0 ), nsidechgs( 0 ), ncoefchgs( 0 ), nrounds( 0 ),
         ndeletedcols( 0 ), ndeletedrows( 0 )
   {
   }
};

inline Statistics
operator-( const Statistics& a, const Statistics& b )
{
   return {
       0.0, a.ntsxapplied - b.ntsxapplied, a.ntsxconflicts - b.ntsxconflicts,
       a.nboundchgs - b.nboundchgs, a.nsidechgs - b.nsidechgs,
       a.ncoefchgs - b.ncoefchgs, a.nrounds - b.nrounds,
       a.ndeletedcols - b.ndeletedcols, a.ndeletedrows - b.ndeletedrows };
}

} // namespace papilo

#endif