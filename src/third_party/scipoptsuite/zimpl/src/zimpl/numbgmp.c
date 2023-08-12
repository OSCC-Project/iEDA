/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: numbgmp.c                                                     */
/*   Name....: Number Functions using gmp                                    */
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
#include <stdbool.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>

#include <gmp.h>

/* #define TRACE 1 */

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/random.h"
#include "zimpl/gmpmisc.h"
#include "zimpl/mme.h"
#include "zimpl/numb.h"

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
      mpq_t  numb;
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
static Numb* numb_const_unknown  = NULL;

/* ??? numb_const_unkown is a hack. GMP has no way to indicate NaN or Infinity.
 * But we need this for example to convey that the startvalue of a variable has
 * not been set. 
 */ 

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

   gmp_init(verbose >= VERB_VERBOSE, with_management);

   numb_const_zero     = numb_new();
   numb_const_one      = numb_new_integer(1);
   numb_const_minusone = numb_new_integer(-1);
   numb_const_unknown  = numb_new_ascii("7e21");
}

void numb_exit()
{
   NumbStore* store;
   NumbStore* next;

   numb_free(numb_const_zero);
   numb_free(numb_const_one);
   numb_free(numb_const_minusone);
   numb_free(numb_const_unknown);

   numb_const_zero     = NULL;
   numb_const_one      = NULL;
   numb_const_minusone = NULL;
   numb_const_unknown  = NULL;

   if (store_count != 0)
      printf("Numb store count %d\n", store_count);
   
   for(store = store_anchor; store != NULL; store = next)
   {
      next = store->next;

      /* ??? mpq_clear() is not called for the used ones.
       * This would be faster then doing it in numb_free.
       */
      /* ??? SIDs are not cleared.
       */
      free(store->begin);
      free(store);
   }   
   store_anchor = NULL;
   store_free   = NULL;
   store_count  = 0;

   gmp_exit();
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

   mpq_init(numb->value.numb);

   return numb;
}

Numb* numb_new_ascii(char const* val)
{
   Numb* numb = numb_new();
   
   assert(numb != NULL);

   gmp_str2mpq(numb->value.numb, val);
   
   return numb;
}

Numb* numb_new_integer(int val)
{
   Numb* numb = numb_new();
   
   assert(numb != NULL);

   mpq_set_si(numb->value.numb, val, 1); 
   
   return numb;
}

Numb* numb_new_longlong(long long val)
{
   assert(val > LLONG_MIN);

   Numb* numb = numb_new();

   assert(numb != NULL);

   bool is_negative = false;
   
   if (val < 0)
   {
      is_negative = true;
      val         = -val;
   }

   mpz_t z;
   mpz_init(z);
   
   /* Function: void mpz_import (
      mpz_t rop,
      size_t count,  // words of size bytes to read
      int order,     // 1 for most significant word first or -1 for least significant first (doesn't matter with one word)
      size_t size,   // sizeof(long long)
      int endian,    // 0 for the native endianness of the host CPU.
      size_t nails,  // 0 bits to skip
      const void *op)
   */
   mpz_import(z, 1, -1, sizeof(val), 0, 0, &val);

   if (is_negative)
      mpz_neg(z, z);

   mpq_set_z(numb->value.numb, z);
   
   mpz_clear(z);
   
   return numb;
}

Numb* numb_new_mpq(const mpq_t val)
{
   Numb* numb = numb_new();
   
   assert(numb != NULL);

   mpq_set(numb->value.numb, val); 
   
   return numb;
}

void numb_free(Numb* numb)
{
   Trace("numb_free");

   assert(numb_is_valid(numb));

   mpq_clear(numb->value.numb);
   
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

   mpq_set(numb->value.numb, source->value.numb);

   return numb;
}

/* true wenn gleich, sonst false
 */
bool numb_equal(Numb const* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   return mpq_equal(numb_a->value.numb, numb_b->value.numb) != 0;
}

/* Return a positive value if op1 > op2, zero if op1 = op2, and a negative value if op1 < op2
 */
int numb_cmp(Numb const* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   return mpq_cmp(numb_a->value.numb, numb_b->value.numb);
}

void numb_set(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_set(numb_a->value.numb, numb_b->value.numb);
}

void numb_add(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_add(numb_a->value.numb, numb_a->value.numb, numb_b->value.numb);
}

Numb* numb_new_add(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_add(numb->value.numb, numb_a->value.numb, numb_b->value.numb);

   return numb;
}

void numb_sub(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_sub(numb_a->value.numb, numb_a->value.numb, numb_b->value.numb);
}

Numb* numb_new_sub(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_sub(numb->value.numb, numb_a->value.numb, numb_b->value.numb);

   return numb;
}

void numb_mul(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_mul(numb_a->value.numb, numb_a->value.numb, numb_b->value.numb);
}

Numb* numb_new_mul(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_mul(numb->value.numb, numb_a->value.numb, numb_b->value.numb);

   return numb;
}

void numb_div(Numb* numb_a, Numb const* numb_b)
{
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_div(numb_a->value.numb, numb_a->value.numb, numb_b->value.numb);
}

Numb* numb_new_div(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_div(numb->value.numb, numb_a->value.numb, numb_b->value.numb);

   return numb;
}

void numb_intdiv(Numb* numb_a, Numb const* numb_b)
{
   mpz_t q;

   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_div(numb_a->value.numb, numb_a->value.numb, numb_b->value.numb);

   mpz_init(q);
   mpz_tdiv_q(q, mpq_numref(numb_a->value.numb), mpq_denref(numb_a->value.numb));
   mpq_set_z(numb_a->value.numb, q);
   mpz_clear(q);
}

Numb* numb_new_intdiv(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();
   mpz_t q;

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpq_div(numb->value.numb, numb_a->value.numb, numb_b->value.numb);

   mpz_init(q);
   mpz_tdiv_q(q, mpq_numref(numb->value.numb), mpq_denref(numb->value.numb));
   mpq_set_z(numb->value.numb, q);
   mpz_clear(q);

   return numb;
}

void numb_mod(Numb* numb_a, Numb const* numb_b)
{
   mpz_t a;
   mpz_t b;
   mpz_t r;

   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpz_init(a);
   mpz_init(b);
   mpz_init(r);

   mpz_mul(a, mpq_numref(numb_a->value.numb), mpq_denref(numb_b->value.numb));
   mpz_mul(b, mpq_numref(numb_b->value.numb), mpq_denref(numb_a->value.numb));
   mpz_mod(r, a, b);
   mpq_set_z(numb_a->value.numb, r);

   mpz_clear(r);
   mpz_clear(b);
   mpz_clear(a);
}

Numb* numb_new_mod(Numb const* numb_a, Numb const* numb_b)
{
   Numb* numb = numb_new();
   mpz_t a;
   mpz_t b;
   mpz_t r;
   

   assert(numb != NULL);
   assert(numb_is_valid(numb_a));
   assert(numb_is_valid(numb_b));

   mpz_init(a);
   mpz_init(b);
   mpz_init(r);

   mpz_mul(a, mpq_numref(numb_a->value.numb), mpq_denref(numb_b->value.numb));
   mpz_mul(b, mpq_numref(numb_b->value.numb), mpq_denref(numb_a->value.numb));
   mpz_mod(r, a, b);
   mpq_set_z(numb->value.numb, r);

   mpz_clear(r);
   mpz_clear(b);
   mpz_clear(a);

   return numb;
}

Numb* numb_new_pow(Numb const* base, int expo)
{
   assert(expo > INT_MIN);

   Numb* numb = numb_new();
   
   assert(numb != NULL);
   assert(numb_is_valid(base));

   bool  is_negative = false;

   /* ??? we should limit the exponent to say 1000
    */
   if (expo < 0)
   {
      is_negative = true;
      expo        = -expo;
   }
   mpq_set_si(numb->value.numb, 1, 1);  /* set to 1 */

   for(int i = 0; i < expo; i++)
      mpq_mul(numb->value.numb, numb->value.numb, base->value.numb);

   if (is_negative)
      mpq_inv(numb->value.numb, numb->value.numb);
   
   return numb;
}

Numb* numb_new_fac(int n)
{
   Numb* numb = numb_new();
   mpz_t z;
   
   assert(numb != NULL);
   assert(n    >= 0);

   mpz_init(z);
   mpz_fac_ui(z, (unsigned int)n);
   mpq_set_z(numb->value.numb, z);
   mpz_clear(z);
   
   return numb;
}

void numb_neg(Numb* numb)
{
   assert(numb_is_valid(numb));

   mpq_neg(numb->value.numb, numb->value.numb);
}

void numb_abs(Numb* numb)
{
   assert(numb_is_valid(numb));

   mpq_abs(numb->value.numb, numb->value.numb);
}

void numb_sgn(Numb* numb)
{
   assert(numb_is_valid(numb));

   /*lint -e(634) Strong type mismatch (type 'Bool') in equality or conditional */
   switch(mpq_sgn(numb->value.numb))
   {
   case -1 :
      mpq_set(numb->value.numb, const_minus_one);
      break;
   case 0 :
      mpq_set(numb->value.numb, const_zero);
      break;
   case 1 :
      mpq_set(numb->value.numb, const_one);
      break;
   default :
      abort();
   }
}

int numb_get_sgn(Numb const* numb)
{
   assert(numb_is_valid(numb));

   return mpq_sgn(numb->value.numb); //lint !e633 !e634 strong type mismatch
}

void numb_round(Numb* numb)
{
   mpz_t q;
   mpq_t h;
   
   assert(numb_is_valid(numb));

   mpz_init(q);
   mpq_init(h);
   mpq_set_d(h, 0.5);

   if (mpq_sgn(numb->value.numb) >= 0) //lint !e633 !e634 !e638 strong type mismatch
      mpq_add(numb->value.numb, numb->value.numb, h);
   else
      mpq_sub(numb->value.numb, numb->value.numb, h);

   mpz_tdiv_q(q, mpq_numref(numb->value.numb), mpq_denref(numb->value.numb));
   mpq_set_z(numb->value.numb, q);

   mpz_clear(q);
   mpq_clear(h);
}

void numb_ceil(Numb* numb)
{
   mpz_t q;
   
   assert(numb_is_valid(numb));

   mpz_init(q);
   mpz_cdiv_q(q, mpq_numref(numb->value.numb), mpq_denref(numb->value.numb));
   mpq_set_z(numb->value.numb, q);
   mpz_clear(q);
}

void numb_floor(Numb* numb)
{
   mpz_t q;
   
   assert(numb_is_valid(numb));

   mpz_init(q);
   mpz_fdiv_q(q, mpq_numref(numb->value.numb), mpq_denref(numb->value.numb));
   mpq_set_z(numb->value.numb, q);
   mpz_clear(q);
}

Numb* numb_new_log(Numb const* numb)
{
   char   temp[256];
   double d;
   
   assert(numb_is_valid(numb));

   d = log10(mpq_get_d(numb->value.numb));

   /* !finite == !isfinite == isnan || isinf */
   CLANG_WARN_OFF(-Wfloat-equal)
      
   if ((d != d) || (d == -HUGE_VAL)) /*lint !e777 */ /* == isnan(d) || isinf(d) */
   {
      sprintf(temp, "*** Error 700: log(%f)", mpq_get_d(numb->value.numb));
      perror(temp);
      return NULL;
   }
   CLANG_WARN_ON
      
   sprintf(temp, "%.16e", d);

   return numb_new_ascii(temp);
}

Numb* numb_new_sqrt(Numb const* numb)
{
   char   temp[256];
   double d;
   
   assert(numb_is_valid(numb));

   d = sqrt(mpq_get_d(numb->value.numb));

   /* !finite == !isfinite == isnan || isinf */
   if (d != d) /*lint !e777 */ /* == isnan(d) || isinf(d) */
   {
      sprintf(temp, "*** Error 701: sqrt(%f)", mpq_get_d(numb->value.numb));
      perror(temp);
      return NULL;
   }
   sprintf(temp, "%.16e", d);

   return numb_new_ascii(temp);
}

Numb* numb_new_exp(Numb const* numb)
{
   char temp[32];
   
   assert(numb_is_valid(numb));

   sprintf(temp, "%.16e", exp(mpq_get_d(numb->value.numb)));

   return numb_new_ascii(temp);
}

Numb* numb_new_ln(Numb const* numb)
{
   char   temp[256];
   double d;
   
   assert(numb_is_valid(numb));

   d = log(mpq_get_d(numb->value.numb));

   CLANG_WARN_OFF(-Wfloat-equal)
      
   /* !finite == !isfinite == isnan || isinf */
   if ((d != d) || (d == -HUGE_VAL)) /*lint !e777 */ /* == isnan(d) || isinf(d) */
   {
      sprintf(temp, "*** Error 702: ln(%f)", mpq_get_d(numb->value.numb));
      perror(temp);
      return NULL;
   }
   CLANG_WARN_ON
      
   sprintf(temp, "%.16e", d);

   return numb_new_ascii(temp);
}

Numb* numb_new_rand(Numb const* mini, Numb const* maxi)
{
   Numb* numb = numb_new();
   mpq_t maxint;
   mpq_t factor;

   assert(numb != NULL);
   assert(numb_is_valid(mini));
   assert(numb_is_valid(maxi));
   assert(numb_cmp(mini, maxi) <= 0);
   
   mpq_init(factor);
   mpq_init(maxint);
   
   mpq_set_ui(numb->value.numb, rand_get_int32(), 1); 
   mpq_set_ui(maxint, 4294967295UL, 1);
   
   mpq_div(numb->value.numb, numb->value.numb, maxint);
   mpq_sub(factor, maxi->value.numb, mini->value.numb);
   mpq_mul(numb->value.numb, numb->value.numb, factor);
   mpq_add(numb->value.numb, numb->value.numb, mini->value.numb);
      
   mpq_clear(factor);
   mpq_clear(maxint);

   return numb;
}

double numb_todbl(Numb const* numb)
{
   assert(numb_is_valid(numb));
   
   return mpq_get_d(numb->value.numb);
}

void numb_get_mpq(Numb const* numb, mpq_t value)
{
   assert(numb_is_valid(numb));
   
   mpq_set(value, numb->value.numb);
}

void numb_print(FILE* fp, Numb const* numb)
{
   assert(numb_is_valid(numb));

   fprintf(fp, "%.16g", mpq_get_d(numb->value.numb));
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
   
   d2i.d = mpq_get_d(numb->value.numb);
   hcode = d2i.i.a ^ d2i.i.b;

   return hcode;
}

char* numb_tostr(Numb const* numb)
{
   char* str;
   
   assert(numb_is_valid(numb));

   str = malloc(32);
      
   assert(str != NULL);
      
   sprintf(str, "%.16g", mpq_get_d(numb->value.numb));

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

Numb const* numb_unknown()
{
   return numb_const_unknown;
}

bool numb_is_int(Numb const* numb)
{
   /* Do we have an integer ?
    */
   if (mpz_get_si(mpq_denref(numb->value.numb)) == 1)
   {
      /* And is it small enough ?
       */
      if (mpz_fits_sint_p(mpq_numref(numb->value.numb)) == 1)
         return true;
   }
   return false;
}

int numb_toint(Numb const* numb)
{
   assert(numb_is_valid(numb));
   assert(numb_is_int(numb));
   
   return (int)mpz_get_si(mpq_numref(numb->value.numb)); 
}

bool numb_is_number(const char *s)
{
   /* 5 !*/
   if (isdigit(*s))
      return true;

   /* maybe -5 or .6 or -.7 ? */
   if (*s != '+' && *s != '-' && *s != '.')
      return false;

   s++;

   /* -5 or .6 ! */
   if (isdigit(*s))
      return true;

   /* maybe -.7 ? */
   if (*s != '.')
      return false;
   
   s++;
   
   return isdigit(*s);
}





