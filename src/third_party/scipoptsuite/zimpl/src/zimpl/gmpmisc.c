/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: ratmisc.c                                                     */
/*   Name....: miscellenious rational arithmetic functions                   */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2003-2022 by Thorsten Koch <koch@zib.de>
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
#include <ctype.h>
#include <stdbool.h>
#include <assert.h>

#include <gmp.h>

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/gmpmisc.h"

mpq_t const_zero;
mpq_t const_one;
mpq_t const_minus_one;

#define POOL_SIZE        10000
#define POOL_ELEM_SIZE   16

typedef union pool_elem PoolElem;
typedef struct pool     Pool;

union pool_elem
{
   char      pad[POOL_ELEM_SIZE]; //lint !e754 union member not referenced
   PoolElem* next;   
};

struct pool
{
   PoolElem elem[POOL_SIZE];
   Pool*    next;
};

static Pool*     pool_root  = NULL;
static PoolElem* pool_next  = NULL;

static bool     gmp_with_management                      = false;
static void* (*gmp_old_alloc)  (size_t)                 = NULL;
static void* (*gmp_old_realloc)(void*, size_t, size_t) = NULL;
static void  (*gmp_old_free)   (void*, size_t)         = NULL;

static void* pool_alloc(void)
{
   Pool*     pool;
   PoolElem* elem;

   if (pool_next == NULL)
   {
      int i;

      pool       = malloc(sizeof(*pool));
      pool->next = pool_root;
      pool_root  = pool;
      
      for(i = 0; i < POOL_SIZE - 1; i++)
         pool->elem[i].next = &pool->elem[i + 1];

      pool->elem[i].next = NULL;

      pool_next = &pool->elem[0];
   }
   assert(pool_next != NULL);

   elem      = pool_next;
   pool_next = elem->next;

   return elem; //lint !e636 strong type difference
}

static void pool_free(void* pv)
{
   PoolElem* elem = pv; //lint !e636 strong type difference

   elem->next = pool_next;
   pool_next  = elem;
}

static void pool_exit(void)
{
   Pool* p;
   Pool* q;
   
   for(p = pool_root; p != NULL; p = q)
   {
      q = p->next;

      free(p);
   }
   pool_root = NULL;
   pool_next = NULL;
}

/* [+|-]?[0-9]*.[0-9]+[[e|E][+|-][0-9]+]? */
/* if it does not fit here, it doesn't fit in a double either */
void gmp_str2mpq(mpq_t value, char const* num)
{
   char  tmp[1024]; 
   int   i;
   int   k = 0;
   int   exponent = 0;
   int   fraction = 0;
   
   assert(num         != NULL);
   assert(strlen(num) <  32);

   /* printf("%s ", num); */
   
   /* Skip initial whitespace
    */
   while(isspace(*num))
      num++;

   /* Skip initial +/-
    */
   if (*num == '+')
      num++;
   else if (*num == '-')
      tmp[k++] = *num++;
   
   for(i = 0; num[i] != '\0'; i++)
   {
      if (isdigit(num[i]))
      {
         tmp[k++]  = num[i];
         exponent -= fraction;
      }
      else if (num[i] == '.')
         fraction = 1;
      else if (tolower(num[i]) == 'e')
      {
         exponent += atoi(&num[i + 1]);
         break;
      }
   }
   while(exponent > 0)
   {
      tmp[k++] = '0';
      exponent--;
   }         
   tmp[k++] = '/';
   tmp[k++] = '1';

   while(exponent < 0)
   {
      tmp[k++] = '0';
      exponent++;
   }         
   tmp[k] = '\0';

   /* printf("%s\n", tmp);*/
   
   (void)mpq_set_str(value, tmp, 10);
   mpq_canonicalize(value);
}

void gmp_print_mpq(FILE* fp, const mpq_t qval)
{
   mpf_t fval;
   
   mpf_init(fval);
   mpf_set_q(fval, qval);
   (void)mpf_out_str(fp, 10, 32, fval); 
   fprintf(fp, " = ");
   (void)mpq_out_str(fp, 10, qval);
   fputc('\n', fp);
   mpf_clear(fval);
}
   
/*ARGSUSED*/
static void* gmp_alloc(size_t size)
{
   if (size <= POOL_ELEM_SIZE)
      return pool_alloc();

   return malloc(size);
}

/*ARGSUSED*/
static void* gmp_realloc(void* ptr, size_t old_size, size_t new_size)
{
   void* p;

   if (old_size <= POOL_ELEM_SIZE && new_size <= POOL_ELEM_SIZE)
      return ptr;

   if (old_size <= POOL_ELEM_SIZE)
   {
      assert(new_size > POOL_ELEM_SIZE);
      assert(new_size > old_size);
      
      p = malloc(new_size);

      memcpy(p, ptr, old_size);
      
      pool_free(ptr);
      
      return p;
   }
   if (new_size <= POOL_ELEM_SIZE) 
   {
      assert(old_size > POOL_ELEM_SIZE);
      assert(old_size > new_size);
      
      p = pool_alloc();

      memcpy(p, ptr, new_size);
      
      free(ptr);
      
      return p;
   }
   return realloc(ptr, new_size);
}

/*ARGSUSED*/
static void gmp_free(void* ptr, size_t size)
{
   if (size <= POOL_ELEM_SIZE)
      pool_free(ptr);
   else
      free(ptr);
}

void gmp_init(bool verbose, bool with_management)
{
   if (with_management)
   {
      gmp_with_management = true;
      mp_get_memory_functions(&gmp_old_alloc, &gmp_old_realloc, &gmp_old_free);      
      mp_set_memory_functions(gmp_alloc, gmp_realloc, gmp_free);
   }
   mpq_init(const_zero);
   mpq_init(const_one);
   mpq_init(const_minus_one);

   mpq_set_ui(const_one, 1, 1);        /* = 1 */
   mpq_set_si(const_minus_one, -1, 1); /* = -1 */

   if (verbose)
      printf("Using GMP Version %s %s\n", 
         gmp_version, with_management ? "[memory management redirected]" : "[memory management unchanged]");
}

void gmp_exit()
{
   mpq_clear(const_zero);
   mpq_clear(const_one);
   mpq_clear(const_minus_one);

   pool_exit();

   if (gmp_with_management)
   {
      mp_set_memory_functions(gmp_old_alloc, gmp_old_realloc, gmp_old_free);
      gmp_with_management = false;
   }
}

