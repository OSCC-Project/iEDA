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


/*
#**************************************************************************
#**   Please read README to see how to run this program
***   Created by Chung-Wen Albert Tsao  on May 2, 2000*
#**
#**
#**************************************************************************
*/


#ifndef _BSTDME_H
#define _BSTDME_H

extern "C++"
{

#include  <string>

using namespace std;
class BstTree  ;

/**************************************************/
/**************************************************/
class BST_GenTestcase {
public:
  /*  generate a testcase of n sinks */
  static bool GenerateTestcase( unsigned n ) ;

} ;
/**************************************************/
/**************************************************/
class BST_DME 
{

public:
  enum DelayModelType 
  {
      LINEARMODEL  = 0,   
      ELMOREMODEL , 
      NumDelayModel          
  };

public:
  // constructor 
  BST_DME ( 
    const string &inputSinksFileName,
    const string &inputTopologyFileName,
    const string &inputObstructionFileName,
    double skewBound,
    DelayModelType delayModel ) ;

  // destructor 
  ~BST_DME() ; 

  /*************************************************/
  /* initiate the construction of BST */
  /*************************************************/
  void ConstructTree ( void ) ;

  /*************************************************/
  /* total length of the resulting BST after calling ConstructTree() ; */
  /*************************************************/
  double TotalLength ( void ) const ;

private:
// data:
   BstTree *m_bstdme ;
   
} ;

}
#endif

