/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: symbol.c                                                      */
/*   Name....: Symbol Table Functions                                        */
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

#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/entry.h"
#include "zimpl/hash.h"
#include "zimpl/stmt.h"
#include "zimpl/symbol.h"

#define SYMBOL_SID  0x53796d62
#define SYMBOL_EXTEND_SIZE 100

struct symbol
{
   SID
   char const*  name;
   int          size;
   int          used;
   int          extend;
   SymbolType   type;
   Set*         set;
   Hash*        hash;
   Entry**      entry;
   Entry*       deflt;
   Symbol*      next;
};

static Symbol* anchor = NULL;

Symbol* symbol_new(
   char const*  name,
   SymbolType   type,
   Set const*   set,
   int          estimated_size,
   Entry const* deflt)
{
   Symbol* sym;

   assert(name           != NULL);
   assert(strlen(name)   >  0);
   assert(set            != NULL);
   assert(estimated_size >= 0);
   
   sym = calloc(1, sizeof(*sym));

   assert(sym != NULL);
   
   sym->name    = name;
   sym->type    = type;
   sym->size    = 1;
   sym->used    = 0;
   sym->extend  = SYMBOL_EXTEND_SIZE;
   sym->set     = set_copy(set);
   sym->hash    = hash_new(HASH_ENTRY, estimated_size);
   sym->entry   = calloc(1, sizeof(*sym->entry));
   sym->deflt   = (deflt != ENTRY_NULL) ? entry_copy(deflt) : ENTRY_NULL;
   sym->next    = anchor;
   anchor       = sym;

   assert(sym->entry != NULL);

   SID_set(sym, SYMBOL_SID);
   assert(symbol_is_valid(sym));

   return sym;
}

void symbol_exit(void)
{
   Symbol* q;
   Symbol* p;
   int     i;
   
   for(p = anchor; p != NULL; p = q)
   {
      assert(symbol_is_valid(p));

      SID_del(p);

      q = p->next;
      
      for(i = 0; i < p->used; i++)
         entry_free(p->entry[i]);

      free(p->entry);
      set_free(p->set);
      hash_free(p->hash);

      if (p->deflt != NULL)
         entry_free(p->deflt);

      free(p);
   }
   anchor = NULL;
}

bool symbol_is_valid(Symbol const* sym)
{
   if (sym == NULL || !SID_ok(sym, SYMBOL_SID))
      return false;

   mem_check(sym);
   mem_check(sym->entry);
      
   return true;
}

Symbol* symbol_lookup(char const* name)
{
   Symbol* sym;

   assert(name != NULL);

   for(sym = anchor; sym != NULL; sym = sym->next)
      if (!strcmp(sym->name, name))
         break;

   return sym;
}

bool symbol_has_entry(Symbol const* sym, Tuple const* tuple)
{
   assert(symbol_is_valid(sym));
   assert(tuple_is_valid(tuple));

   return hash_has_entry(sym->hash, tuple)
      || (sym->deflt != NULL && set_lookup(sym->set, tuple));
}

/* Liefert NULL wenn nicht gefunden.
 * Falls ein default zurueckgegeben wird, stimmt "tuple" nicht mit
 * entry->tuple ueberein.
 */
Entry const* symbol_lookup_entry(Symbol const* sym, Tuple const* tuple)
{
   Entry const* entry;
   
   assert(symbol_is_valid(sym));
   assert(tuple_is_valid(tuple));

   entry = hash_lookup_entry(sym->hash, tuple);

   if (NULL == entry && sym->deflt != NULL && set_lookup(sym->set, tuple))
      entry = sym->deflt;

   return entry;
}

/* Entry is eaten.
 * No check is done if entry->tuple is a member of sym->set !
 * This has to be done before.
 */
void symbol_add_entry(Symbol* sym, Entry* entry)
{
   Tuple const* tuple;
   
   assert(symbol_is_valid(sym));
   assert(entry_is_valid(entry));
   
   assert(sym->used <= sym->size);
   
   if (sym->used == sym->size)
   {
      sym->size   += sym->extend;
      sym->extend += sym->extend;
      sym->entry   = realloc(
         sym->entry, (size_t)sym->size * sizeof(*sym->entry));
      
      assert(sym->entry != NULL);
   }
   assert(sym->used < sym->size);

   tuple = entry_get_tuple(entry);

   /* There is no index set for the internal symbol.
    */
   assert(!strcmp(sym->name, SYMBOL_NAME_INTERNAL) || set_lookup(sym->set, tuple));

   if (hash_has_entry(sym->hash, tuple))
   {
      if (stmt_trigger_warning(166))
      {
         fprintf(stderr, "--- Warning 166: Duplicate element ");
         tuple_print(stderr, tuple);
         fprintf(stderr, " for symbol %s rejected\n", sym->name);
      }
      entry_free(entry);
   }
   else
   {
      /* Falls noch nicht geschehen, legen wir hier den Typ des
       * Symbols fest.
       */
      if ((sym->type == SYM_ERR) && (sym->used == 0))
         sym->type = entry_get_type(entry);

      assert(sym->type != SYM_ERR);
      
      hash_add_entry(sym->hash, entry);
      
      sym->entry[sym->used] = entry;      
      sym->used++;
   }
}

int symbol_get_dim(Symbol const* sym)
{
   assert(symbol_is_valid(sym));

   return set_get_dim(sym->set);
}

Set const* symbol_get_iset(Symbol const* sym)
{
   assert(symbol_is_valid(sym));

   return sym->set;
}

char const* symbol_get_name(Symbol const* sym)
{
   assert(symbol_is_valid(sym));

   return sym->name;
}

SymbolType symbol_get_type(Symbol const* sym)
{
   assert(symbol_is_valid(sym));

   return sym->type;
}

Numb const* symbol_get_numb(Symbol const* sym, int idx)
{
   assert(symbol_is_valid(sym));
   assert(idx >= 0);
   assert(idx <  sym->used);
   
   return entry_get_numb(sym->entry[idx]);
}

char const* symbol_get_strg(Symbol const* sym, int idx)
{
   assert(symbol_is_valid(sym));
   assert(idx >= 0);
   assert(idx <  sym->used);
   
   return entry_get_strg(sym->entry[idx]);
}

Set const* symbol_get_set(Symbol const* sym, int idx)
{
   assert(symbol_is_valid(sym));
   assert(idx >= 0);
   assert(idx <  sym->used);
   
   return entry_get_set(sym->entry[idx]);
}

Var* symbol_get_var(Symbol const* sym, int idx)
{
   assert(symbol_is_valid(sym));
   assert(idx >= 0);
   assert(idx <  sym->used);
   
   return entry_get_var(sym->entry[idx]);
}

void symbol_print(FILE* fp, Symbol const* sym)
{
   static char const* const type_name[] = { "Error", "Numb", "Strg", "Set", "Var" };
   
   int i;
   
   assert(symbol_is_valid(sym));

   fprintf(fp, "Name  : %s\n", sym->name);
   fprintf(fp, "Type  : %s\n", type_name[sym->type]);

   fprintf(fp, "Index : ");
   set_print(fp, sym->set);
   fprintf(fp, "\nEntries:\n");
   
   for(i = 0; i < sym->used; i++)
   {
      fprintf(fp, "\t%3d: ", i);
      entry_print(fp, sym->entry[i]);
      fprintf(fp, "\n");
   }
   fprintf(fp, "\n");
}

void symbol_print_all(FILE* fp)
{
   Symbol* sym;
   
   assert(fp != NULL);

   for(sym = anchor; sym != NULL; sym = sym->next)
      symbol_print(fp, sym);
}


