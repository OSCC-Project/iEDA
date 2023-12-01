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

#ifndef _F_TRR_H
#define _F_TRR_H


// #include "float.h"
#include "BiStates.h"
#include "MyInterval.h"
#include "BaseDefine.h"
#include "Coordinate.h"
// #include <limits.h>

extern "C++"
{

    // NN is a data type used for coordinates
    // NNDD is a date type that have larger precision
    template <typename NN, typename NNDD, bool DOCHECK = true>
    class F_Trr
    {
public:
  
  typedef  F_Interval<NN,NNDD,DOCHECK> Trr_Interval ;
  void Initialize ( ) {
    // Trr_Interval  k ;
    // xlow = ylow = k[Low] ;
    // xhi  = yhi  = k[High] ;
    xlow = ylow = 1      ;
    xhi  = yhi  = 0       ;
  }

  void MakeDiamond ( const F_Point<NN, NNDD, DOCHECK>  &p, NN r ) {
    NN tval;

      tval = p.x - p.y;
      xlow = tval - r;
      xhi  = tval + r;

      tval = p.x + p.y;
      ylow = tval - r;
      yhi = tval + r;

  }
  
  F_Trr () {
    Initialize () ;
  }
  F_Trr ( NN x1, NN x2, NN y1, NN y2 ) : xlow(x1),xhi(x2),ylow(y1), yhi(y2) {
  }
  F_Trr ( const F_Point<NN, NNDD, DOCHECK>  &p, NN r ) {
    MakeDiamond ( p, r ) ;
  }
  void ErrCorrection ( ) {
     NN a = xlow ;
     NN b = xhi  ;
     if ( equal(a,b) ) {
       xlow = xhi = (a+b)/2 ;
     }
     a = ylow ;
     b = yhi  ;
     if ( equal(a,b) ) {
       ylow = yhi = (a+b)/2 ;
     }
  }
  void Enclose ( const F_Trr &b ) {
    if ( IsEmpty() ) {
      xlow = b.xlow ;
      ylow = b.ylow ;
      xhi  = b.xhi  ;
      yhi  = b.yhi  ;
    } else {
      xlow = tMIN (xlow, b.xlow) ;
      ylow = tMIN (ylow, b.ylow) ;
      xhi  = tMAX (xhi , b.xhi ) ;
      yhi  = tMAX (yhi , b.yhi ) ;
    }
  }
  bool IsEmpty ( ) const {
    F_Interval<NN,NNDD,DOCHECK>  xin (xlow, xhi ) ;
    F_Interval<NN,NNDD,DOCHECK>  yin (ylow, yhi ) ;
    return xin.IsEmpty() || yin.IsEmpty() ;
  }
  void SelfDOCHECK ( ) {
    ErrCorrection () ;
    assert ( !IsEmpty() ) ;
  }
  NN Width ( TwoStates i ) const {
    if ( i== 0) { 
      return xhi-xlow;
    } else {
      return yhi-ylow;
    }
  }
  NN Diameter () const {
    return tMAX ( Width(0), Width(1) ) ;
  }

  F_Trr MakeIntersect   ( const F_Trr &trr1, const F_Trr &trr2 ) const {
    NN xlow = tMAX ( trr1.xlow, trr2.xlow ) ;
    NN xhi  = tMIN ( trr1.xhi , trr2.xhi  ) ;
    NN ylow = tMAX ( trr1.ylow, trr2.ylow ) ;
    NN yhi  = tMIN ( trr1.yhi , trr2.yhi  ) ;
    F_Trr trr (xlow,xhi,ylow,yhi ) ;
    trr.SelfDOCHECK () ;
    return  trr ;
  }
  F_Trr MakeCore    ( ) const {
    NN r = this->Radius () ;
    NN x1= (xlow+ r ); 
    NN x2= (xhi - r ); 
    NN y1= (ylow+ r ); 
    NN y2= (yhi - r ); 
    return F_Trr (x1,x2,y1,y2 ) ;
  }

  NN xlow,xhi,ylow,yhi;
} ;

}
#endif
