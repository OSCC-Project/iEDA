/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: numbdbl.c                                                     */
/*   Name....: Number Functions using double                                 */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2001-2022 by Thorsten Koch <koch@zib.de>
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
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <assert.h>

/* #define TRACE 1 */

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/random.h"
#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/mme.h"

#define NUMB_STORE_SIZE  1000
#define NUMB_SID         0x4e756d62

typedef struct numb_storage NumbStore;

/* Note: In case of gmt use refc and infinity indicator.
 */
struct number
{
   SID
   union 
   {
      double numb;
      Numb*  next;
   } value;
};

struct numb_storage
{
   Numb*       begin;
   NumbStore*  next;
};

static NumbStore* store_anchor = NULL;
static Numb*      store_free   = NULL;
static int        store_count  = 0;

/* constants
 */
static Numb* numb_const_zero     = NULL;
static Numb* numb_const_one      = NULL;
static Numb* numb_const_minusone = NULL;

static void          genrand_init(unsigned long s);
static unsigned long genrand_int32(void);

static void extend_storage(void)
{
   NumbStore* store = calloc(1, sizeof(*store));
   Numb*      numb;
   int        i;
   
   assert(store != NULL);
   
   store->begin = malloc(NUMB_STORE_SIZE * sizeof(*store->begin));
   store->next  = store_anchor;
   store_anchor = store;

   for(i = 0; i < NUMB_STORE_SIZE - 1; i++)
   {
      numb             = &store->begin[i];
      numb->value.next = &store->begin[i + 1];
      SID_set(numb, NUMB_SID);
      assert(numb_is_valid(numb));
   }
   numb             = &store->begin[i];
   numb->value.next = store_free;  
   SID_set(numb, NUMB_SID);
   assert(numb_is_valid(numb));
   
   store_free       = &store->begin[0];

   assert(store->begin != NULL);
   assert(store_anchor != NULL);
   assert(store_free   != NULL);
}

void numb_init(bool with_management)
{
   store_anchor = NULL;
   store_free   = NULL;

   numb_const_zero     = numb_new();
   numb_const_one      = numb_new_integer(1);
   numb_const_minusone = numb_new_integer(-1);
}

void numb_exit()
{
   NumbStore* store;
   NumbStore* next;

   numb_free(numb_const_zero);
   numb_free(numb_const_one);
   numb_free(numb_const_minusone);

   numb_const_zero     = NULL;
   numb_const_one      = NULL;
   numb_const_minusone = NULL;

   if (store_count != 0)
      printf("Numb store count %d\n", store_count);
   
   for(store = store_anchor; store != NULL; store = next)
   {
      next = store->next;

      free(store->begin);
      free(store);
   }   
   store_anchor = NULL;
   store_free   = NULL;
   store_count  = 0;
}

/* value is zero */
Numb* numb_new(void)
{
   Numb* numb;

   Trace("numb_new");
   
   if (store_free == NULL)
      extend_storage();

   assert(store_free != NULL);

   numb             = store_free;
   store_free       = numb->value.next;
   store_count++;

   numb->value.numb = 0.0;

   return numb;
}

Numb* numb_new_ascii(char const* val)
{
   Numb* numb = numb_new();
   
   assert(numb != NULL);

   numb->value.numb = atof(val);
   
   return numb;
}

Numb* numb_new_integer(int val)
{
   Numb* numb = numb_new();
   
   assert(numb != NULL);

   numb->value.numb = val;
   
   return numb;
}

Numb* numb_new_longlong(long long val)
{
   Numb* numb = numb_new();
   
   assert(numb != NULL);

   numb->value.numb = val;
   
   return numb;
}

void numb_free(Numb* numb)
{
   Trace("numb_free");

   assert(numb_is_valid(numb));

   numb->value.next = store_free;
   store_free       = numb;

   store_count--;   
}

bool numb_is_valid(Numb const* numb)
{
   return numb != NULL && SID_ok(numb, NUMB_SID);
}

Numb* numb_copy(Numb const* source)
{
   Numb* numb = numb_new();

   assert(numb_is_valid(source));
   assert(numb_is_valid(numb));

   numb->value.numb = source->value.numb;

   return numb;
}

/* true wenn gleich, sonst false
 */
/* ??? This not the same as with gmp :-) */
bool numb_equal(Numb const* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   return numb_a->value.numb == numb_b->value.numb;
}

/* Return a positive value if op1 > op2, zero if op1 = op2, and a negative value if op1 < op2
 */
int numb_cmp(Numb const* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   if (numb_a->value.numb > numb_b->value.numb)
      return 1;
   if (numb_a->value.numb < numb_b->value.numb)
      return -1;
   return 0;
}

void numb_set(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb_a->value.numb = numb_b->value.numb;
}

void numb_add(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb_a->value.numb += numb_b->value.numb;
}

Numb* numb_new_add(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb->value.numb = numb_a->value.numb + numb_b->value.numb;

   return numb;
}

void numb_sub(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb_a->value.numb -= numb_b->value.numb;
}

Numb* numb_new_sub(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb->value.numb = numb_a->value.numb - numb_b->value.numb;

   return numb;
}

void numb_mul(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb_a->value.numb *= numb_b->value.numb;
}

Numb* numb_new_mul(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb->value.numb = numb_a->value.numb * numb_b->value.numb;

   return numb;
}

void numb_div(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb_a->value.numb /= numb_b->value.numb;
}

Numb* numb_new_div(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb->value.numb = numb_a->value.numb / numb_b->value.numb;

   return numb;
}

void numb_intdiv(Numb* numb_a, Numb const* numb_b)
{
   double q;

   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb_a->value.numb = trunc(numb_a->value.numb / numb_b->value.numb);
}

Numb* numb_new_intdiv(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb->value.numb = trunc(numb_a->value.numb / numb_b->value.numb);

   return numb;
}

void numb_mod(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb_a->value.numb = fmod(numb_a->value.numb, numb_b->value.numb);
}

Numb* numb_new_mod(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   numb->value.numb = fmod(numb_a->value.numb, numb_b->value.numb);

   return numb;
}

Numb* numb_new_pow(Numb const* base, int expo)
{
   Numb* numb = numb_new();
   
   assert(numb != NULL);
   assert(numb_is_valid(base));

   numb->value.numb = pow(base->value.numb, expo);
   
   return numb;
}

Numb* numb_new_fac(int n)
{
   Numb* numb = numb_new();
   int   i;

   assert(numb != NULL);
   assert(n    >= 0);

   numb->value.numb = 1;
   
   for(i = 2; i <= n; i++)
      numb->value.numb *= i;
   
   return numb;
}

void numb_neg(Numb* numb)
{
   assert(numb_is_valid(numb));

   numb->value.numb *= -1.0;
}

void numb_abs(Numb* numb)
{
   assert(numb_is_valid(numb));

   numb->value.numb = fabs(numb->value.numb);
}

void numb_sgn(Numb* numb)
{
   assert(numb_is_valid(numb));

   if (numb->value.numb < 0.0)
      numb->value.numb = -1.0;
   else if (numb->value.numb > 0.0)
      numb->value.numb =  1.0;
   else
      numb->value.numb =  0.0;
}

int numb_get_sgn(Numb const* numb)
{
   assert(numb_is_valid(numb));

   if (numb->value.numb < 0.0)
      return -1;

   if (numb->value.numb > 0.0)
      return  1;

   return 0;
}

void numb_round(Numb* numb)
{
   assert(numb_is_valid(numb));

   if (numb->value.numb > 0.0)
      numb->value.numb = trunc(numb->value.numb + 0.5);
   else
      numb->value.numb = trunc(numb->value.numb - 0.5);      
}

void numb_ceil(Numb* numb)
{
   assert(numb_is_valid(numb));

   numb->value.numb = ceil(numb->value.numb);
}

void numb_floor(Numb* numb)
{
   assert(numb_is_valid(numb));

   numb->value.numb = floor(numb->value.numb);
}

Numb* numb_new_log(Numb const* numb_a)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));

   numb->value.numb = log10(numb_a->value.numb);

   /* !finite == !isfinite == isnan || isinf */
   if (numb->value.numb != numb->value.numb) 
   {
      char temp[256];

      sprintf(temp, "*** Error 700: log(%f)", numb_a->value.numb);
      perror(temp);
      return NULL;
   }
   return numb;
}

Numb* numb_new_sqrt(Numb const* numb_a)
{
   Numb* numb = numb_new();
   
   assert(numb != NULL);
   assert(numb_is_valid(numb_a));

   numb->value.numb = sqrt(numb_a->value.numb);

   /* !finite == !isfinite == isnan || isinf */
   if (numb->value.numb != numb->value.numb) 
   {
      char temp[256];

      sprintf(temp, "*** Error 701: sqrt(%f)", numb_a->value.numb);
      perror(temp);
      return NULL;
   }
   return numb;
}

Numb* numb_new_exp(Numb const* numb_a)
{
   char temp[32];
   Numb* numb = numb_new();
   
   assert(numb != NULL);
   assert(numb_is_valid(numb_a));

   numb->value.numb = exp(numb_a->value.numb);

   return numb;
}

Numb* numb_new_ln(Numb const* numb_a)
{
   Numb* numb = numb_new();
   
   assert(numb != NULL);
   assert(numb_is_valid(numb_a));

   numb->value.numb = log(numb_a->value.numb);

   /* !finite == !isfinite == isnan || isinf */
   if (numb->value.numb != numb->value.numb) 
   {
      char temp[256];
      
      sprintf(temp, "*** Error 702: ln(%f)", numb->value.numb);
      perror(temp);
      return NULL;
   }
   return numb;
}

Numb* numb_new_rand(Numb const* mini, Numb const* maxi)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(mini));
   assert(numb_is_valid(maxi));
   assert(numb_cmp(mini, maxi) <= 0);
   
   numb->value.numb  = rand_get_int32();
   numb->value.numb /= 4294967295.0; /* MAXINT */
   numb->value.numb *= maxi->value.numb - mini->value.numb;
   numb->value.numb += mini->value.numb;
      
   return numb;
}

double numb_todbl(Numb const* numb)
{
   assert(numb_is_valid(numb));
   
   return numb->value.numb;
}

void numb_print(FILE* fp, Numb const* numb)
{
   assert(numb_is_valid(numb));

   fprintf(fp, "%.16g", numb->value.numb);
}

unsigned int numb_hash(Numb const* numb)
{
   union
   {
      struct
      {
         unsigned int a;
         unsigned int b;
      } i;
      double d;
   } d2i;
   
   unsigned int hcode;
   
   d2i.d = numb->value.numb;
   hcode = d2i.i.a ^ d2i.i.b;

   return hcode;
}

char* numb_tostr(Numb const* numb)
{
   char* str;
   
   assert(numb_is_valid(numb));

   str = malloc(32);
      
   assert(str != NULL);
      
   sprintf(str, "%.16g", numb->value.numb);

   return str;
}

Numb const* numb_zero()
{
   return numb_const_zero;
}

Numb const* numb_one()
{
   return numb_const_one;
}

Numb const* numb_minusone()
{
   return numb_const_minusone;
}

bool numb_is_int(Numb const* numb)
{
   return numb->value.numb == (double)((int)numb->value.numb);
}

int numb_toint(Numb const* numb)
{
   assert(numb_is_valid(numb));
   assert(numb_is_int(numb));
   
   return (int)numb->value.numb; 
}

bool numb_is_number(const char *s)
{
   /* 5 !*/
   if (isdigit(*s))
      return true;

   /* maybe -5 or .6 or -.7 ? */
   if (*s != '+' && *s != '-' && *s != '.')
      return false;

   if (*s == '\0')
      return false;

   s++;

   /* -5 or .6 ! */
   if (isdigit(*s))
      return true;

   /* maybe -.7 ? */
   if (*s != '.')
      return false;
   
   if (*s == '\0')
      return false;

   s++;
   
   return isdigit(*s);
}



