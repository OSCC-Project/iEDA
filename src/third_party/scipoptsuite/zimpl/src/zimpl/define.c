/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: define.c                                                      */
/*   Name....: Define Table Functions                                        */
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

#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/define.h"

#define DEFINE_SID  0x44656669

struct define
{
   SID
   char const*  name;
   DefineType   type;
   Tuple*       param;
   CodeNode*    code;
   Define*      next;
};

#ifndef NDEBUG
static Define anchor  = { 0, "", DEF_ERR, NULL, NULL, NULL };
#else
static Define anchor  = {    "", DEF_ERR, NULL, NULL, NULL };
#endif

static bool define_is_valid(Define const* def)
{
   if (def == NULL || !SID_ok(def, DEFINE_SID))
      return false;

   mem_check(def);

   return true;
}

Define* define_new(
   char const*  name,
   DefineType   type)
{
   Define* def;

   assert(name           != NULL);
   assert(strlen(name)   >  0);
   assert(type           != DEF_ERR);
   
   def = calloc(1, sizeof(*def));

   assert(def != NULL);
   
   def->name    = name;
   def->type    = type;
   def->param   = NULL;
   def->code    = NULL;
   def->next    = anchor.next;
   anchor.next  = def;

   SID_set(def, DEFINE_SID);
   assert(define_is_valid(def));

   return def;
}

void define_set_param(
   Define*     def,
   Tuple*      param)
{
   assert(define_is_valid(def));
   assert(tuple_is_valid(param));
   
   def->param   = param;
}

void define_set_code(
   Define*     def,
   CodeNode*   code)
{
   assert(define_is_valid(def));
   assert(code != NULL);
   
   def->code = code;
}

void define_exit(void)
{
   Define* q;
   Define* p;
   
   for(p = anchor.next; p != NULL; p = q)
   {
      assert(define_is_valid(p));

      SID_del(p);

      tuple_free(p->param);
      
      q = p->next;
      
      free(p);
   }
   anchor.next = NULL;
}

Define* define_lookup(char const* name)
{
   Define* def;

   assert(name != NULL);

   for(def = anchor.next; def != NULL; def = def->next)
      if (!strcmp(def->name, name))
         break;

   return def;
}

char const* define_get_name(Define const* def)
{
   assert(define_is_valid(def));

   return def->name;
}

DefineType define_get_type(Define const* def)
{
   assert(define_is_valid(def));

   return def->type;
}

Tuple const* define_get_param(Define const* def)
{
   assert(define_is_valid(def));

   return def->param;
}

CodeNode* define_get_code(Define const* def)
{
   assert(define_is_valid(def));

   return def->code;
}








