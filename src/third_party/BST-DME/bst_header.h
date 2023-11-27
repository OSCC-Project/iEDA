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

#ifndef _HEADER_H
#define _HEADER_H


#include "foundation/Coordinate.h"
#include "foundation/MyInterval.h"
#include "foundation/Trr.h"
#include "bstdme.h"
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include  <sys/times.h>
#include <sys/types.h>
// #include <sys/param.h>
#include  <sys/resource.h>
#include  <time.h>
#include  <assert.h>
#include  <float.h>
#include  <string.h>
#include <stdlib.h>
#include  <vector>
#include  "foundation/Trr.h"

typedef F_Interval < double, double, true > R_Interval ;
typedef F_Coord < double, double, true > R_Coord ;
typedef F_Point < double, double, true > PointType ;

typedef F_Trr < double, double, true > TrrType ;


#define H_   0
#define V_   1
#define NORMAL  0
#define FAST    1

#define MAX_TURN_PTS       12
#define MAX_mr_PTS        16                      
#define MAXPTS            10                     


#define PUCAP_SCALE        (1E+12)


#define N_Neighboring_Clusters      6

/* types of interseciton of two line segments */
#define P1             1
#define P2             2
#define P3             3
#define P4             4
#define XING           5
#define OVERLAPPING    6
#define SAME_EDGE      7

#define LINEAR         0
#define ELMORE         1

#define BME_MODE       0
#define IME_MODE       1
#define HYBRID_MODE    2

#define LEFT   0            /* vertical or horizontal line segment */
#define TOP    1
#define RIGHT  2
#define BOTTOM 3
#define PARA_MANHATTAN_ARC 4

#define INSIDE        -1
#define ON             0
#define OUTSIDE       +1


/* types of a line segment */ 
#define VERTICAL          0
#define HORIZONTAL        1
#define MANHATTAN_ARC     2           /* include single points */
#define FLAT              3
#define TILT              4

#define NIL    -1
#define BUCKET_CAPACITY   240  /* max. number of elements in a bucket */

#define MAX_BUFFER_LEVEL   8

#define MANHATTAN         0
#define EUCLIDEAN         1

#define MAX_N_SKEW_SAMPLES 50

#define N_BUFFER_SIZE 50

#define MAX_N_SINKS       3101
#define MAX_N_NODES       (2*MAX_N_SINKS)


class PairType {
public:
int x, y;
double cost;
} ;



/* used for CostFunction == 1 only */
typedef struct clustertype {
double x;   /* ever min merging cost since this cluster exists */
double y;   /* min merging cost in current Nearest Neighbor Graph */
/* merging cost increase of Cluster = y-x >= 0 */
} ClusterType;

class TmpClusterType {
public:
  int n, id, marked;
  double capac, t;
  TrrType  *ms;

  TmpClusterType ( ) : ms(0) {
  } ;
  ~TmpClusterType () {
    if ( ms ) {
      delete ms;
      ms = 0 ;
    }
  }
} ;




typedef struct buckettype {
  int num;    /* number of elements in the bucket */   
  int element[BUCKET_CAPACITY];
} BucketType;

class AreaType {
public:
  int n_mr;   /* number of vertices plus skew turning points */
  PointType mr[MAX_mr_PTS]; /* skew turning pts + vertices */
  int npts;    /* number of vertices */
  int vertex[MAXPTS];   /* vertices */
  int L_area, R_area;  /* which area of child nodes is used */
  double L_EdgeLen, R_EdgeLen; /* edge length to its children */
  double L_StubLen, R_StubLen; /* edge length of stubs of buffered children */
  double subtree_cost;  /* the (wire) cost of tree rooted at this node */
  double dist; /* the distance between its two children */
               /*  node.dist = linedist(node.line[2][2]) */
  int R_buffer;
  double ClusterDelay;
  double capac, unbuf_capac;    /* capacitance of the subtree */
  PointType line[2][2];
  AreaType *area_L, *area_R; 
  TrrType L_MS, R_MS;
} ;

class NodeType {
public:
  NodeType ():area(0), ms(0) {
  } ;
  /*
  ~NodeType () {
    if ( ms ) {
      delete ms;
      ms = 0 ;
    }
    if ( area ) {
      delete [] area ; // nate 
      area = 0 ;
    }
  } ;
  */
  

  /* member function */
  void Merge2Nodes( NodeType *node_L,NodeType *node_R) ;

  PointType m_stnPt ;
  /* construct multiple merging areas for a node */
  int   n_area, ca;
  AreaType  *area;
  /**************************************************/
  TrrType  *ms;  /* merging segment in ZST, and TRR in BST */  
  int parent, L, R, id, root_id;
  char pattern;   /* routing pattern */
  char buffered;   /* indicated if buffered */
} ;


class AreaSetType {
public:
  int npoly;
  AreaType  *freg; 
} ;

/**************************************************/
/**************************************************/
class BstTree {
public:

public:
  // constructor 
  BstTree ( 
    const string &inputSinksFileName,
    const string &inputTopologyFileName,
    const string &inputObstructionFileName,
    double skewBound,
    BST_DME::DelayModelType delayModel ) ;

  // destructor 
  ~BstTree() ; 

  /*************************************************/
  /* initiate the construction of BST */
  /*************************************************/
  void ConstructTree ( void ) ;

  /*************************************************/
  /* total length of the resulting BST after calling ConstructTree() ; */
  /*************************************************/
  double TotalLength ( void ) const ;


  /*************************************************/
  /* for internal use only */
  /* for internal use only */
  /* for internal use only */
  /* for internal use only */
  /* for internal use only */
  /*************************************************/

  /* set the file name for input sinks, and per-unit rc */
  /* mandotory */

  /* set the topology file name if run BST-DME for fixed topology */
  /* optinoal, default is null */
  void SetTopologyFileName (const string &fn ) {
    m_inputTopologyFileName = fn ;
  } ;


  /* set the global skew bound for BST-DME (default 0ps)*/
  void SetSkewBound ( double b ) {
    m_skewBound = b ;
  }

   /* Set the file name for obstruction data (optional) */
  void SetObstructionFileName (const string &fn ) {
    m_inputObstructionFileName = fn ;
  } ;

  /* Optional, default is Elmore delay model */
  void SetDelayModel ( BST_DME::DelayModelType i ) {
    m_delayModel = i ;
  }

  string SinksFileName () const {
    return m_inputSinksFileName ;
  } ;

  /* if ToplogyFileName() not empty, then run BST-DME on fixed topology */
  bool FixedTopology () const {
    return !TopologyFileName().empty() ;
  };

  string TopologyFileName () const {
    return m_inputTopologyFileName ;
  };

  string ObstructionFileName () const {
    return m_inputObstructionFileName ;
  } ;

  double SkewBound ( void ) const {
    return m_skewBound ;
  } ;

  double Orig_Skew_B ( void ) const {
    return m_skewBound ;
  }
  bool   LinearDelayModel () const {
    return m_delayModel == BST_DME::LINEARMODEL ;  
  } ;
  
  bool   ElmoreDelayModel () const {
    return m_delayModel == BST_DME::ELMOREMODEL ;  
  } ;

  int RootNodeIndex () const {
    return 2*m_nterms - 1 ;
  }

  int SuperRootNodeIndex () const {
    return m_nterms ;
  }

  NodeType *SuperRootNode ( ) {
    return m_treeNode[ m_nterms ] ;
  } ;

  NodeType *RootNode ( ) {
    return m_treeNode[ 2*m_nterms - 1] ;
  } ;

  NodeType *TreeNode ( unsigned i ) {
    return m_treeNode[i] ;
  } ;

  NodeType * GetCandiRoot ( unsigned i ) {
    return m_candiRoot [i] ;                
  } ;
  NodeType * GetTempNode ( unsigned i ) {
    return m_tempNode [i] ;                
  }

  void AddTreeNode ( NodeType * node ) {
    m_treeNode.push_back ( node ) ;
  }
  void AddCandiRoot (NodeType * node ) {
    m_candiRoot.push_back ( node ) ;
  }
  void AddTempNode (NodeType * node ) {
    m_tempNode.push_back ( node ) ;
  }

  const TrrType & Ctrr () const {
    return m_ctrr;
  }
 
  void BuildTrr ( unsigned n ) ;

  NodeType* AddOneNode ( unsigned i , double x, double y,
   double cap, double delay) ;

  unsigned Nterms () const {
    return m_nterms ;
  }

  void SetNterms ( unsigned i ) {
    m_nterms = i ;
  }

  unsigned Npoints () const {
    return 2*Nterms() ;
  }

  double Pucap ( unsigned i=0 ) {
    return m_pucap[i] ;
  }

  double Pures ( unsigned i=0 ) {
    return m_pures[i] ;
  }

  void SetPerUnitResistance ( double pures[2] ) {
    m_pures[0] = pures [0];
    m_pures[1] = pures [1];
  }

  void SetPerUnitCapacitance ( double pucap[2] ) {
    m_pucap[0] = pucap [0];
    m_pucap[1] = pucap [1];
  }

  void SetLayerParasitics ( double pures[2], double pucap[2] ) {
    SetPerUnitResistance ( pures ) ;
    SetPerUnitCapacitance ( pucap ) ;
  }

private:
// data:
   string m_inputSinksFileName ;
   string m_inputTopologyFileName ; 
   string m_inputObstructionFileName ;

   double  m_skewBound ;
   BST_DME::DelayModelType   m_delayModel ;
   
   vector <NodeType *> m_treeNode ;
   vector <NodeType *> m_candiRoot ;
   vector <NodeType *> m_tempNode ;

   unsigned m_nterms ; // num of termianls 
   TrrType m_ctrr ;
   double m_pures[2], m_pucap[2]; /*per unit resistance and capacitance */

   
} ;

#endif

