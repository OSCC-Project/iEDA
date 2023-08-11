/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: random.c                                                      */
/*   Name....: Random number functions                                       */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2007-2022 by Thorsten Koch <koch@zib.de>
 * 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

/* #define TRACE 1 */

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/random.h"

/* 
   A C-program for MT19937, with initialization improved 2002/2/10.
   Coded by Takuji Nishimura and Makoto Matsumoto.
   This is a faster version by taking Shawn Cokus's optimization,
   Matthe Bellew's simplification, Isaku Wada's real version.

   Before using, initialize the state by using init_genrand(seed) .

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.keio.ac.jp/matumoto/emt.html
   email: matumoto@math.keio.ac.jp
*/

/* Period parameters */  
#define N            624
#define M            397
#define MATRIX_A     0x9908b0dfU   /* constant vector a */
#define UMASK        0x80000000U /* most significant w-r bits */
#define LMASK        0x7fffffffU /* least significant r bits */
#define MIXBITS(u,v) (((u) & UMASK) | ((v) & LMASK))
#define TWIST(u,v)   ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))

static unsigned int        state[N]; /* the array for the state vector  */
static int                 left  = 1;
static const unsigned int* next;

/* initializes state[N] with a seed */
void rand_init(unsigned long s)
{
   unsigned int j;
    
   state[0] = s & 0xffffffffU;

   for(j = 1; j < N; j++)
   {
      state[j] = (1812433253U * (state[j-1] ^ (state[j-1] >> 30)) + j); 

      /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
      /* In the previous versions, MSBs of the seed affect   */
      /* only MSBs of the array state[].                     */
      /* 2002/01/09 modified by Makoto Matsumoto             */
      state[j] &= 0xffffffffU;  /* for >32 bit machines */
   }
   left = 1;
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned int rand_get_int32(void)
{
   unsigned int y;

   if (--left == 0)
   {
      unsigned int* p = state;
      int           j;

      left = N;
      next = state;
    
      for(j = N - M + 1; --j; p++) //lint !e440 !e441 !e443 
         *p = p[M] ^ TWIST(p[0], p[1]);

      for(j = M; --j; p++) //lint !e440 !e441 !e443 
         *p = p[M-N] ^ TWIST(p[0], p[1]);

      *p = p[M-N] ^ TWIST(p[0], state[0]);
   }
   y = *next++;

   /* Tempering */
   y ^= (y >> 11);
   y ^= (y <<  7) & 0x9d2c5680U;
   y ^= (y << 15) & 0xefc60000U;
   y ^= (y >> 18);

   return y;
}

/* ----------------------------------------------------------------------------- */
/* Should give a random number between min and max
 */
int rand_get_range(int mini, int maxi)
{
   double r = rand_get_int32() / 4294967295.0;

   assert(mini < maxi);
   
   return (int)(r * (maxi - mini) + 0.5) + mini;
}

