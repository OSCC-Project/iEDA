/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: blkmem.c                                                      */
/*   Name....: Block Memory Functions                                        */
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
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/blkmem.h"

typedef struct memory_block_element BlkMemElem;
typedef struct memory_block         BlkMem;

struct memory_block_element
{
   BlkMemElem* next;
};

struct memory_block
{
   size_t      elem_count;
   BlkMemElem* elem;
   BlkMem*     next;
};

#define MIN_BLK_ELEM_COUNT  1024
#define MAX_BLK_ELEM_SIZE   64
#define MAX_CHAINS         (MAX_BLK_ELEM_SIZE / 8)

static BlkMem*     blk_anchor[MAX_CHAINS];
static BlkMemElem* first_free[MAX_CHAINS];
static int         blk_fails = 0;
static int         blk_count = 0;

static void extend_memory(int chain_no)
{
   BlkMem* block      = calloc(1, sizeof(*block));
   size_t  elem_size  = ((size_t)chain_no + 1) * 8;
   size_t  offset     = elem_size / sizeof(BlkMemElem);
   size_t  i;
   
   assert(elem_size % (int)sizeof(BlkMemElem) == 0);
   assert(block    != NULL);
   assert(chain_no >= 0);
   assert(chain_no <  MAX_CHAINS);
   
   block->elem_count    = (blk_anchor[chain_no] == NULL)
                        ? MIN_BLK_ELEM_COUNT : (blk_anchor[chain_no]->elem_count * 2);
   block->elem          = malloc(block->elem_count * elem_size);
   block->next          = blk_anchor[chain_no];   
   blk_anchor[chain_no] = block;
   
   assert(block->elem != NULL);
   assert(elem_size == offset * sizeof(*block->elem));
   
   for(i = 0; i < block->elem_count - 1; i++)
      block->elem[i * offset].next = &block->elem[(i + 1) * offset];

   assert(i == block->elem_count - 1);
   
   block->elem[i * offset].next = first_free[chain_no];
   first_free[chain_no]         = &block->elem[0];

   assert(first_free[chain_no] != NULL);
   assert(blk_anchor[chain_no] != NULL);
}

void blk_init(void)
{
   int i;

   for(i = 0; i < MAX_CHAINS; i++)
   {
      blk_anchor[i] = NULL;
      first_free[i] = NULL;
   }
}

void blk_exit(void)
{
   int i;

   if (blk_count != 0)
      printf("Block memory allocation count %d\n", blk_count);

#ifndef NDEBUG
   if (blk_fails != 0)
      printf("Block memory allocation fails: %d\n", blk_fails);
#endif

   for(i = 0; i < MAX_CHAINS; i++)
   {
      BlkMem* anchor;
      BlkMem* next;

      for(anchor = blk_anchor[i]; anchor != NULL; anchor = next)
      {
         next = anchor->next;

         free(anchor->elem);
         free(anchor);
      }
   }
}

void* blk_alloc(int size)
{
   BlkMemElem* elem;
   int         chain_no = (size + 7) / 8 - 1;

   assert(size     >  0);
   assert(size     <  MAX_BLK_ELEM_SIZE);
   assert(chain_no >= 0);

   if (chain_no >= MAX_CHAINS)
   {
      blk_fails++;
      return malloc((size_t)size);
   }
   if (first_free[chain_no] == NULL)
      extend_memory(chain_no);

   assert(first_free[chain_no] != NULL);

   elem                 = first_free[chain_no];
   first_free[chain_no] = elem->next;
   blk_count++;
   
   return elem;
}

void blk_free(void* p, int size)
{
   int         chain_no  = (size + 7) / 8 - 1;
   BlkMemElem* elem      = p;

   assert(p        != NULL);
   assert(size     >  0);
   assert(size     <  MAX_BLK_ELEM_SIZE);
   assert(chain_no >= 0);

   if (chain_no >= MAX_CHAINS)
   {
      free(p);
      return;
   }

#ifndef NDEBUG
   /* Check whether the elem is really from on eof the blocks.
    */
   {
      BlkMem* anchor    = blk_anchor[chain_no];
      int     elem_size = (chain_no + 1) * 8;
      size_t  offset    = (size_t)elem_size / sizeof(BlkMemElem);

      assert(anchor != NULL);

      while(elem < anchor->elem || elem >= anchor->elem + anchor->elem_count * offset)
      {
         anchor = anchor->next;
         assert(anchor != NULL);
      }
   }
#endif
   
   elem->next           = first_free[chain_no];
   first_free[chain_no] = elem;
   blk_count--;
}

