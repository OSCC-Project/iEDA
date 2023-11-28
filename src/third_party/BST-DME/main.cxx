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

#include <stdio.h>
#include <float.h>
#include "bstdme.h"
#include "facility.h"
#include <string.h>
#include <stdlib.h>
using namespace std;


/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void tPrintTotalLength (
  const BST_DME &tree
) {
  double wire = tree.TotalLength () ;

  printf("=================================\n" );
  iShowTime () ;
  printf("\n" );
  printf("Total wirelength: %f\n", wire);
  printf("=================================\n" );
}

/****************************************************************************/
/*   parse arguments                                                        */
/****************************************************************************/
unsigned parse_argument( unsigned argc, char *argv[], 
  string &inputSinksFileName ,     
  string &inputTopologyFileName ,     
  string &inputObstructionFileName ,
  double  &skewBound ,     
  BST_DME::DelayModelType   &delayModel      
) {
unsigned nterms = 0 ;
  
  for (unsigned i=1;i<argc;i++) {
    if (strcmp("-i",argv[i]) == 0)  {
      i++;
      inputSinksFileName = argv[i] ;
    } else if (strcmp("-G",argv[i]) == 0)  {
      i++;
      inputTopologyFileName = argv[i] ;
    } else if (strcmp("-O",argv[i]) == 0)  { // not complete yet
      i++;
      inputObstructionFileName = argv[i] ;
    } else if (strcmp("-N",argv[i]) == 0)  { // to generate random testcase
      i++;
      sscanf(argv[i],"%d",&nterms);
    } else if (strcmp("-D",argv[i]) == 0)  { // for delay model
      i++;
      unsigned model ;
      sscanf(argv[i],"%d",& model );
      delayModel = (BST_DME::DelayModelType ) model ;
    } else if (strcmp("-B",argv[i]) == 0)  { // for skew bound
      i++;
      sscanf(argv[i],"%lf",& skewBound);
      if ( skewBound < 0) { 
         skewBound = DBL_MAX; 
      } 
    } else {
      printf("Argument %d incorrect\n",i);
      exit(0);
    }
  }

  return nterms ;
}

const string bstusage = 
    "usage: bst -i inputFileName -B number (pico-seconds) \n"    ; 

/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
static
void bstsVersion () {
  const string lin = 
   "**********************************************************************\n" ;
  const string msg = 
      lin   
    + "* Version BSTsource1.1 ,  05-11-2002                                          *\n"   
    + "* Copyright (C) 1994-2002  by Andrew B. Kahng, C.-W. Albert Tsao              *\n"   
    +  lin  
    ; 
  cout << msg << endl ;
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
int 
main(int argc, char *argv[]) {

  bstsVersion () ;

string inputSinksFileName = "" ; // mandatory
string inputTopologyFileName = "" ;
string inputObstructionFileName = "" ;
double  skewBound = 0 ;
BST_DME::DelayModelType delayModel = BST_DME::ELMOREMODEL;

     unsigned nterms = parse_argument (argc, argv, 
                      inputSinksFileName,
                      inputTopologyFileName, 
                      inputObstructionFileName,
                      skewBound,
                      delayModel
                      );


  if ( nterms ) { // generate testcase only 
    BST_GenTestcase::GenerateTestcase( nterms ) ;
  } else if ( inputSinksFileName.empty() ) { // mandatory 
     cout << bstusage << endl ;
     exit (0 ) ;
  } else {
    BST_DME tree (inputSinksFileName,
                inputTopologyFileName,
                inputObstructionFileName,
                skewBound,
                delayModel ) ;
    tree.ConstructTree() ;
    tPrintTotalLength ( tree ) ;
  }


}
