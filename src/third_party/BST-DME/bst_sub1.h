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


#ifndef _BST_SUB1_H
#define _BST_SUB1_H

void ShowTime() ;

void print_Point(FILE *f, const PointType& pt) ;

void read_input_file( const string &fn );
void print_merging_tree( const char fn[], int v);
void print_bst(const char fn[], int v, double TCost, double Tdist);
void read_input_topology ( const string &fn ) ;
void print_topology(const char fn[], int v, double TCost, double Tdist);
void RunTime() ;
int equal(double x,double y) ;

void print_clustering_info();


void check_JS_MS() ;
double ms_distance(TrrType *ms1,TrrType *ms2) ;

void ms2line(TrrType *ms, PointType *p1, PointType *p2) ;

double calc_boundary_length(AreaType *area) ;

int Manhattan_arc(PointType p1,PointType p2) ;

void make_Point_TRR(PointType p, TrrType *trr) ;

int areaJS_line_type(AreaType *area) ;
void print_area(AreaType *area) ;
void print_area_info(AreaType *area) ;
void assign_NodeType(NodeType *node1, NodeType *node2) ;

void calc_merge_distance(double r,double c, double cap1,double delay1,
      double cap2, double delay2, double d,double *d1,double *d2)  ;


#endif

