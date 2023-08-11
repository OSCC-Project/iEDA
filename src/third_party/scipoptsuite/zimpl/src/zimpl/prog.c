/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: prog.c                                                        */
/*   Name....: Program Functions                                             */
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

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/symbol.h"
#include "zimpl/entry.h"
#include "zimpl/idxset.h"
#include "zimpl/rdefpar.h"
#include "zimpl/bound.h"
#include "zimpl/define.h"
#include "zimpl/mono.h"
#include "zimpl/term.h"
#include "zimpl/list.h"
#include "zimpl/local.h"
#include "zimpl/code.h"
#include "zimpl/stmt.h"
#include "zimpl/prog.h"

#define PROG_SID         0x50726f67
#define PROG_EXTEND_SIZE 100

static void* lp_data = NULL;

struct program
{
   SID
   int         size;
   int         used;
   int         extend;
   Stmt**      stmt;
};

void* prog_get_lp()
{
   return lp_data;
}

Prog* prog_new()
{
   Prog* prog = calloc(1, sizeof(*prog));

   assert(prog != NULL);

   prog->size     = PROG_EXTEND_SIZE;
   prog->used     = 0;
   prog->extend   = PROG_EXTEND_SIZE;
   prog->stmt     = calloc((size_t)prog->size, sizeof(*prog->stmt));

   SID_set(prog, PROG_SID);
   assert(prog_is_valid(prog));

   return prog;
}

void prog_free(Prog* prog)
{
   int i;
   
   assert(prog_is_valid(prog));
   assert(prog->stmt     != NULL);
   
   SID_del(prog);

   for(i = 0; i < prog->used; i++)
      stmt_free(prog->stmt[i]);

   free(prog->stmt);
   free(prog);
}

bool prog_is_valid(Prog const* prog)
{
   return ((prog != NULL) && SID_ok(prog, PROG_SID));
}

bool prog_is_empty(Prog const* prog)
{
   return prog->used == 0;
}

void prog_add_stmt(Prog* prog, Stmt* stmt)
{
   assert(prog_is_valid(prog));
   assert(stmt_is_valid(stmt));

   assert(prog->used <= prog->size);
   
   if (prog->used == prog->size)
   {
      prog->size   += prog->extend;
      prog->extend += prog->extend;
      prog->stmt    = realloc(
         prog->stmt, (size_t)prog->size * sizeof(*prog->stmt));
      
      assert(prog->stmt != NULL);
   }
   assert(prog->used < prog->size);

   prog->stmt[prog->used] = stmt;
   prog->used++;   
}

void prog_print(FILE* fp, Prog const* prog)
{
   int i;

   assert(prog_is_valid(prog));
   
   fprintf(fp, "Statements: %d\n", prog->used);

   for(i = 0; i < prog->used; i++)
      stmt_print(fp, prog->stmt[i]);
}

void prog_execute(Prog const* prog, void* lp)
{
   int i;

   assert(prog_is_valid(prog));

   code_clear_inst_count();

   lp_data = lp;

   for(i = 0; i < prog->used; i++)
   {
      stmt_parse(prog->stmt[i]);
      stmt_execute(prog->stmt[i]);

      /* These calls should make sure, that all output is really
       * flushed out, even in a Batch environment.
       */
      fflush(stdout);
      fflush(stderr);

#ifdef USE_FSYNC
      /* This is to force the output do disk. It is to my knowledge
       * only needed on AIX batch systems that seem not to flush
       * the output buffer. If then the job is killed for some reason
       * no output is generated.
       */
      (void)fsync(fileno(stdout));
      (void)fsync(fileno(stderr));
#endif
   }
   if (verbose >= VERB_NORMAL)
      printf("Instructions evaluated: %u\n", code_get_inst_count());
}

char* prog_tostr(Prog const* prog, char const* prefix, char const* title, size_t max_output_line_len)
{
   size_t len;
   char*  text;
   int    pos = 0;
   int    i;

   assert(prog_is_valid(prog));
   assert(prefix != NULL);
   assert(max_output_line_len > strlen(prefix));

   /* prefix + title + \n
    * prog->used * (\n + prefix + stmt)
    * \0
    */
   len = strlen(prefix) + strlen(title) + 2;

   for(i = 0; i < prog->used; i++)
   {
      size_t line_len         = strlen(stmt_get_text(prog->stmt[i]));
      size_t max_eff_line_len = max_output_line_len - strlen(prefix) - 1;

      len += line_len + ((line_len + max_eff_line_len - 1) / max_eff_line_len) * (strlen(prefix) + 1);
   }
   text = calloc(len, sizeof(*text));
   pos  = sprintf(&text[pos], "%s%s", prefix, title);

   for(i = 0; i < prog->used; i++)
   {    
      char const* s = stmt_get_text(prog->stmt[i]);
      int         k = 0;

      while(*s != '\0')
      {
         if ((size_t)k % max_output_line_len == 0)
         {
            k = sprintf(&text[pos], "\n%s", prefix);            
            pos += k;
         }
         text[pos] = *s;

         pos++;
         s++;
         k++;
      }
   }
   text[pos++] = '\n';     
   text[pos]   = '\0';

   /* for(i = 0; i < prog->used; i++)
    *  pos += sprintf(&text[pos], "%s%s\n", prefix, stmt_get_text(prog->stmt[i]));
    */
   assert((size_t)pos + 1 == len);
   
   return text;
}
