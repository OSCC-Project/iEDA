///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2018, The Regents of the University of California
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

/****************************************************************************/
/* This software was written by C.-W. Albert Tsao  and is subject to:   */
/*                                                                          */
/*  Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao         */
/*  Computer Science Department, University of California, Los Angeles. */
/****************************************************************************/
#ifndef _F_COORD_H
#define _F_COORD_H

#include "BiStates.h"
#include "BaseDefine.h"

extern "C++"
{

  // NN  is a data type used for the coordinates.
  // NNDD is a type with a larger precision.
  template <class NN, class NNDD, bool DOCHECK = true>
  class F_Coord
  {

    private:
        NN  m_coord [ 2] ;
    public:
        // constructor 
        F_Coord (NN  x, NN  y ) {
          m_coord[ 0] = x, m_coord[ 1 ] = y;
        } ;

        F_Coord (const F_Coord& p2) {
            m_coord[0] = p2.m_coord[0];
            m_coord[1] = p2.m_coord[1];
        } ;

        // member function  
        NN  operator[] ( TwoStates i) const {
            return m_coord[i];
        } ;
    } ;
    template <class NN, class NNDD, bool DOCHECK = true>
    class F_Point {
    public:
      F_Point () {
        x=y=0;
        max=min=0;
        t=0 ;
      } ;
      NN     x,y;   /* coordinate */ 
      NN     max, min;  /* (max, min) delays after merging */
      NN     t;    /* usded for sorting */
    } ;

}
#endif
