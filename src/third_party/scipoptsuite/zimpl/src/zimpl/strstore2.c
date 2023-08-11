/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: strstore2.c                                                   */
/*   Name....: String Storage Functions                                      */
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
#include <assert.h>

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/strstore.h"
#include "zimpl/mme.h"

typedef struct string_storage StrStore;

struct string_storage
{
   char*      begin;
   size_t     size;
   size_t     used;
   StrStore*  next;
};

#define MIN_STR_STORE_SIZE 65536UL      /* 64k */
#define MAX_STR_STORE_SIZE 1073741824UL /* 1G  */

static StrStore* store_anchor = NULL;
static size_t    store_size   = MIN_STR_STORE_SIZE;

static void extend_storage(void)
{
   StrStore* store;

   assert(store_size > 0);
   
   store = calloc(1, sizeof(*store_anchor));

   assert(store != NULL);

   store->size  = store_size;
   store->used  = 0;
   store->next  = store_anchor;
   store->begin = calloc(store_size, sizeof(*store->begin));

   assert(store->begin != NULL);

   store_anchor = store;
}

char const* str_new(char const* s)
{
   char*  t;
   size_t len;

   assert(store_anchor != NULL);
   assert(s            != NULL);

   len = strlen(s) + 1;

   if (len > MAX_STR_STORE_SIZE)
   {
      fprintf(stderr, "*** Error 803: String too long %zu > %zu\n",
         len + 1, (size_t)MAX_STR_STORE_SIZE); 

      zpl_exit(EXIT_FAILURE);
   }
   if (store_anchor->size - store_anchor->used < len)
   {
      /* Double the store_size at least once each time,
       * but more often in case it is not big enough to hold a very long
       * string.
       */
      if (store_size < MAX_STR_STORE_SIZE)
      {
         do
         {
            store_size *= 2;
         }
         while(len > store_size);
      }
      extend_storage();
   }
   assert(store_anchor->size - store_anchor->used >= len);

   t = &store_anchor->begin[store_anchor->used];

   store_anchor->used += len;
   
   return strcpy(t, s);
}

void str_init(void)
{
   extend_storage();
}

void str_exit(void)
{
   StrStore* p;
   StrStore* q;

   for(p = store_anchor; p != NULL; p = q)
   {
      q = p->next;
      free(p->begin);
      free(p);
   }
   store_anchor = NULL;
}

unsigned int str_hash(char const* s)
{
#if 0
   return (unsigned int)s;
#else
   unsigned int sum = 0;
   int          i;
   
   for(i = 0; s[i] != '\0'; i++)
      sum = sum * 31 + (unsigned int)s[i];

   return sum;
#endif
}

