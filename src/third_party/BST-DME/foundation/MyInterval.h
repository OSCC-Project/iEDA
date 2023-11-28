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

#ifndef _F_INTERVAL_H
#define _F_INTERVAL_H


#include "BiStates.h"
#include "BaseDefine.h"

extern "C++"
{
  template <class NN, class  NNDD, bool DOCHECK = true >
  class F_Interval {
  public: 

    /* constructor */
    F_Interval (void) { m_bound[Low] = 1; m_bound[High] = 0; }

    F_Interval ( NN a) { m_bound[Low] = a; m_bound[High] = a; }

    F_Interval ( NN a,  NN b) {
      m_bound[Low] = tMIN (a,b);
      m_bound[High] = tMAX (a,b);
    }

    F_Interval (const F_Interval& x) {
      m_bound[Low] = x.m_bound[Low];
      m_bound[High] = x.m_bound[High];
    }

    /* member function */
    bool
    IsEmpty (void) const {
        return m_bound[Low] > m_bound[High];
    }
    
    bool
    IsPoint (void) const {
       return m_bound[Low] == m_bound[High];
    }
    void Enclose (  NN n) {
      if ( IsEmpty() ) {
        m_bound[0]  = n ;
        m_bound[1]  = n ;
      } else {
        m_bound[Low]  = tMIN ( m_bound[Low], n ) ;
        m_bound[High] = tMAX ( m_bound[High], n ) ;
      }
    }

    void Enclose ( const F_Interval< NN, NNDD,DOCHECK>& p ) {
      if ( !p.IsEmpty() ) {
        Enclose ( p[Low] ) ;
        Enclose ( p[High] ) ;
      }
    }


    
    bool
    IsEnclose ( NN n) const {
        return n >= m_bound[Low] && n <= m_bound[High];
    }
    
    bool
    IsEnclose (const F_Interval< NN, NNDD,DOCHECK>& p) const {
        return p[Low] >= m_bound[Low] && p[High] <= m_bound[High];
    }
    
     NN Width (void) const {
        if (IsEmpty()) return 0;
        return m_bound[High] - m_bound[Low];
    }

    
    private:
     NN m_bound [2];
  } ;
} ;


#endif
