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


#include  <sys/times.h>
#include <sys/param.h>
#include <stdio.h>


#include "Time.h"

#define CPU_PER_SECOND  HZ
#define CPU_PER_MINUTE  (HZ * 60)


/************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew Kahng, Chung-Wen Albert Tsao     */
/************************************************************************/
void RunTime() {
static int lastSysTime = 0;
static int lastUserTime = 0;

struct tms      buffer;
int             uhrs, shrs;
int             umins, smins;
float           usecs, ssecs;

  times(&buffer);
  umins = buffer.tms_utime - lastUserTime;
  smins = buffer.tms_stime - lastSysTime;
  lastUserTime = buffer.tms_utime;
  lastSysTime = buffer.tms_stime;
  usecs = umins % CPU_PER_MINUTE;
  usecs /= CPU_PER_SECOND;
  umins /= CPU_PER_MINUTE;
  uhrs  = umins/60;
  umins = umins %60;
  ssecs = smins % CPU_PER_MINUTE;
  ssecs /= CPU_PER_SECOND;
  smins /= CPU_PER_MINUTE;
  shrs  = smins/60;
  smins = smins %60;
  printf("User: [%02d:%02d:%02.0f]\n", uhrs, umins, usecs);
  printf("Sys:  [%02d:%02d:%02.0f]\n", shrs, smins, ssecs);
  printf("\n\n\n");
/*
  printf("CPU_PER_SECOND = %d\n", CPU_PER_SECOND);
  printf("CPU_PER_MINUTE = %d\n", CPU_PER_MINUTE);
*/
  return ;
}


/****************************************************************************/
/*  Display CPU time and current time.                                      */
/****************************************************************************/
void ShowTime() {
struct tms      buffer;
int days,hours,minutes, seconds, hh, mm, ss;


  times(&buffer);
  seconds = (buffer.tms_utime)/CPU_PER_SECOND;
  ss=seconds%60;
  minutes=seconds/60;
  mm=minutes%60;
  hours=minutes/60;
  hh=hours%24;
  days=hours/24;

  printf("Time:");
  if (days>0) { printf("%d:", days); }
  printf("%02d:%02d:%02d(%5d sec.)",hh,mm,ss, seconds);

  fflush(stdout);
}




