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


#ifndef _GLOBAL_VAR_H
#define _GLOBAL_VAR_H

extern "C++"
{

extern BstTree   *gBoundedSkewTree ;
extern vector <NodeType> Node, CandiRoot, TempNode;
extern AreaType *TempArea;
extern int N_TempArea;
extern int N_Sampling ; 
extern  TrrType *L_sampling,  *R_sampling;
extern char *Marked;
extern int *UnMarkedNodes;
extern char Fixed_Topology; 
extern int CHECK ; 
extern int Max_Temp_Area; 
extern int MaxRegionRemoved ; 
extern int N_Area_Per_Node  ; 
extern int k_Parameter ;
extern int  N_Neighbor  ;  
extern int Cost_Function ; 
extern BucketType **Bucket;
extern int BST_Mode ;
extern int Batch_Mode ; 
extern int Dynamic_Selection ;   /* no dynamic topology change  */
extern int N_Index, MAX_N_Index; 
extern TrrType Ctrr;
extern double Start_Tcost, Start_Skew_B, Orig_Skew_B, Skew_B, 
       Skew_B_CLS[MAX_N_SINKS];
extern double Last_Time, Gamma;
extern double PURES[2], PUCAP[2]; /*per unit resistance and capacitance */
extern double PURES_V_SCALE; 
extern double PUCAP_V_SCALE; 
extern double K[2];
       /* K[i] = 0.5*PURES[i]*PUCAP[i], quardratic terms in Elmore delay */
extern PairType *Best_Pair;
extern int  Read_Delay_Skew_File ;
extern PointType  Fms_Pt[2][2];   /* feasible merging section on JR */
extern int n_Fms_Pt[2];
extern PointType  JR_corner[2];
extern  int N_Bal_Pt[2];
extern  PointType Bal_Pt[2][2]; 
extern int  n_JR[2];
extern PointType  JR[2][MAX_TURN_PTS];
extern int  n_JS[2];
extern PointType  JS[2][MAX_TURN_PTS];
extern TrrType  L_MS, R_MS;
extern vector<double> EdgeLength, StubLength;
extern int     *N_neighbors;
extern int   **The_NEIghors;
extern double **Neighbor_Cost;
extern ClusterType *Cluster;
/* =============================  */
extern double MAX_x, MAX_y, MIN_x, MIN_y;
extern double Split_Factor ;
extern int *NearestCenter;
extern int *TmpClusterId;
extern TmpClusterType *TmpCluster, Tmp_x_Cluster, Tmp_y_Cluster;
/* =============================*/

extern int N_Buffer_Level, N_Clusters[MAX_BUFFER_LEVEL], Total_CL;
extern int TreeRoots[MAX_N_NODES];
extern int Cluster_id[MAX_N_NODES], *Buffered;
extern double R_buffer;
extern double C_buffer;
extern double Delay_buffer;
extern double R_buffer_size[N_BUFFER_SIZE];

extern int All_Top;

extern int Expand_Size;
extern double Weight;
extern int Cmode;


extern PointType  *Points;
extern int    *TmpMarked;
extern double *Capac;

extern int Hierachy_Cluster_id[MAX_N_NODES];

extern int N_Obstacle;   /* Number of obstacles */

extern double MaxClusterDelay, *ClusterDelay;

}

#endif

