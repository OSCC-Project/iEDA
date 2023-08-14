/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: list.c                                                        */
/*   Name....: List Functions                                                */
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

#include "zimpl/blkmem.h"
#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/entry.h"
#include "zimpl/list.h"

#define LIST_SID  0x4c697374

enum list_type
{
   LIST_ERR = 0, LIST_ELEM, LIST_TUPLE, LIST_ENTRY, LIST_IDXELEM, LIST_LIST
};

typedef enum list_type      ListType;
typedef union list_data     ListData; 

union list_data
{
   Entry* entry;
   Tuple* tuple;
   Elem*  elem;
   List*  list;
};

struct list_element
{
   ListData  data;
   ListElem* prev;
   ListElem* next;
};

struct list
{
   SID
   int      refc;
   int      elems;
   ListType type;
   ListElem anchor;
};

static void list_add_data(List* list, ListData const* data)
{
   ListElem* elem = blk_alloc(sizeof(*elem));

   assert(list_is_valid(list));
   assert(elem != NULL);
   assert(data != NULL);
   
   elem->data = *data;

   elem->next              = &list->anchor;
   elem->prev              = list->anchor.prev;
   list->anchor.prev->next = elem;
   list->anchor.prev       = elem;
   list->elems++;
}

static void list_insert_data(List* list, ListData const* data)
{
   ListElem* elem = blk_alloc(sizeof(*elem));

   assert(list_is_valid(list));
   assert(elem != NULL);
   assert(data != NULL);
   
   elem->data = *data;

   elem->next              = list->anchor.next;
   elem->prev              = &list->anchor;
   list->anchor.next->prev = elem;
   list->anchor.next       = elem;

   list->elems++;
}

static List* list_new(ListType type, ListData const* data)
{
   List* list = calloc(1, sizeof(*list));
   
   assert(list != NULL);
   assert(data != NULL);
   
   list->refc        = 1;
   list->elems       = 0;
   list->type        = type;
   list->anchor.prev = &list->anchor;
   list->anchor.next = &list->anchor;
   
   SID_set(list, LIST_SID);
   assert(list_is_valid(list));

   list_add_data(list, data);

   return list;
}

List* list_new_elem(Elem const* elem)
{
   ListData data;
   
   assert(elem_is_valid(elem));

   data.elem = elem_copy(elem);

   return list_new(LIST_ELEM, &data);
}

List* list_new_tuple(Tuple const* tuple)
{
   ListData data;
   
   assert(tuple_is_valid(tuple));

   data.tuple = tuple_copy(tuple);

   return list_new(LIST_TUPLE, &data);
}

List* list_new_entry(Entry const* entry)
{
   ListData data;
   
   assert(entry_is_valid(entry));

   data.entry = entry_copy(entry);

   return list_new(LIST_ENTRY, &data);
}

List* list_new_list(List const* list)
{
   ListData data;
   
   assert(list_is_valid(list));

   data.list = list_copy(list);

   return list_new(LIST_LIST, &data);
}

void list_free(List* list)
{   
   ListElem* p;
   ListElem* q;
   
   assert(list_is_valid(list));

   list->refc--;

   if (list->refc == 0)
   {
      SID_del(list);

      for(p = list->anchor.next; p != &list->anchor; p = q)
      {
         assert(p != NULL);
         
         q = p->next;
         
         switch(list->type)
         {
         case LIST_ELEM :
            elem_free(p->data.elem);
            break;
         case LIST_TUPLE :
            tuple_free(p->data.tuple);
            break;
         case LIST_ENTRY :
            entry_free(p->data.entry);
            break;
         case LIST_IDXELEM :
            break;
         case LIST_LIST :
            list_free(p->data.list);
            break;
         default :
            abort();
         }
         blk_free(p, sizeof(*p));
      }
      free(list);
   }
}

bool list_is_valid(List const* list)
{
   return ((list != NULL) && SID_ok(list, LIST_SID) && (list->refc > 0));
}

bool list_is_elemlist(List const* list)
{
   assert(list_is_valid(list));

   return list->type == LIST_ELEM;
}
   
bool list_is_entrylist(List const* list)
{
   assert(list_is_valid(list));

   return list->type == LIST_ENTRY;
}
   
bool list_is_tuplelist(List const* list)
{
   assert(list_is_valid(list));

   return list->type == LIST_TUPLE;
}

List* list_copy(List const* source)
{
   CLANG_WARN_OFF(-Wcast-qual)
   List* list = (List*)source;
   CLANG_WARN_ON
   
   assert(list_is_valid(list));

   list->refc++;

   return list;
}

void list_add_elem(List* list, Elem const* elem)
{
   ListData data;

   assert(list_is_valid(list));
   assert(elem_is_valid(elem));
   assert(list->type == LIST_ELEM);
   
   data.elem = elem_copy(elem);

   list_add_data(list, &data);
}

void list_insert_elem(List* list, Elem const* elem)
{
   ListData data;

   assert(list_is_valid(list));
   assert(elem_is_valid(elem));
   assert(list->type == LIST_ELEM);
   
   data.elem = elem_copy(elem);

   list_insert_data(list, &data);
}

void list_add_tuple(List* list, Tuple const* tuple)
{
   ListData data;

   assert(list_is_valid(list));
   assert(tuple_is_valid(tuple));
   assert(list->type == LIST_TUPLE);
   
   data.tuple = tuple_copy(tuple);

   list_add_data(list, &data);
}

void list_insert_tuple(List* list, Tuple const* tuple)
{
   ListData data;

   assert(list_is_valid(list));
   assert(tuple_is_valid(tuple));
   assert(list->type == LIST_TUPLE);
   
   data.tuple = tuple_copy(tuple);

   list_insert_data(list, &data);
}

void list_add_entry(List* list, Entry const* entry)
{
   ListData data;

   assert(list_is_valid(list));
   assert(entry_is_valid(entry));
   assert(list->type == LIST_ENTRY);

   data.entry = entry_copy(entry);

   list_add_data(list, &data);
}

void list_insert_entry(List* list, Entry const* entry)
{
   ListData data;

   assert(list_is_valid(list));
   assert(entry_is_valid(entry));
   assert(list->type == LIST_ENTRY);

   data.entry = entry_copy(entry);

   list_insert_data(list, &data);
}

void list_add_list(List* list, List const* ll)
{
   ListData data;

   assert(list_is_valid(list));
   assert(list_is_valid(ll));
   assert(list->type == LIST_LIST);
   
   data.list = list_copy(ll);

   list_add_data(list, &data);
}

int list_get_elems(List const* list)
{
   assert(list_is_valid(list));

   return list->elems;
}

static ListData* list_get_data(List const* list, ListElem** idxp)
{
   assert(list_is_valid(list));
   assert(idxp != NULL);
   
   if (*idxp == NULL)
      *idxp = list->anchor.next;

   assert(*idxp != NULL);

   if (*idxp == &list->anchor)
      return NULL;

   *idxp = (*idxp)->next;

   return &((*idxp)->prev->data);
}

Elem const* list_get_elem(List const* list, ListElem** idxp)
{
   ListData* data;
   
   assert(list_is_valid(list));
   assert(list->type == LIST_ELEM);
   assert(idxp != NULL);

   data = list_get_data(list, idxp);

   return (data == NULL) ? ELEM_NULL : data->elem;
}

Tuple const* list_get_tuple(List const* list, ListElem** idxp)
{
   ListData* data;
   
   assert(list_is_valid(list));
   assert(list->type == LIST_TUPLE);
   assert(idxp != NULL);

   data = list_get_data(list, idxp);

   return (data == NULL) ? TUPLE_NULL : data->tuple;
}

Entry const* list_get_entry(List const* list, ListElem** idxp)
{
   ListData* data;
   
   assert(list_is_valid(list));
   assert(list->type == LIST_ENTRY);
   assert(idxp != NULL);

   data = list_get_data(list, idxp);

   return (data == NULL) ? ENTRY_NULL : data->entry;
}

List const* list_get_list(List const* list, ListElem** idxp)
{
   ListData* data;
   
   assert(list_is_valid(list));
   assert(list->type == LIST_LIST);
   assert(idxp != NULL);

   data = list_get_data(list, idxp);

   return (data == NULL) ? LIST_NULL : data->list;
}

void list_print(FILE* fp, List const* list)
{
   ListElem* le;
   
   for(le = list->anchor.next; le != &list->anchor; le = le->next)
   {
      assert(le != NULL);

      switch(list->type)
      {
      case LIST_ELEM :
         elem_print(fp, le->data.elem, true);
         break;
      case LIST_TUPLE :
         tuple_print(fp, le->data.tuple);
         break;
      case LIST_ENTRY :
         entry_print(fp, le->data.entry);
         break;
      case LIST_LIST :
         list_print(fp, le->data.list);
         break;
      default :
         abort();
      }
      fprintf(fp, "\n");
   }
}



