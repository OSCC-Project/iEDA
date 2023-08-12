/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: set4.h                                                        */
/*   Name....: Set Functions                                                 */
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
#ifndef _SET4_H_
#define _SET4_H_

#ifdef __cplusplus
extern "C" {
#endif

#define SET_TYPE_COUNT 11 // need for pc-lint
   
enum set_type {
   SET_ERROR   =  0,
   SET_EMPTY   =  1, /* dim = ?, Empty Set */
   SET_PSEUDO  =  2, /* dim = 0, Has only empty tuple as element */
   SET_LIST    =  3, /* dim = 1, Explicit enumeration of members */
   SET_RANGE   =  4, /* dim = 1, Range: Start to End by Increment */
   SET_PROD    =  5, /* dim > 1, Cross Product of two or more sets */
   SET_MULTI   =  6, /* dim > 1, Multidimensional Subset */
   SET_UNION   =  7, /* dim > 1, Union of two sets */
   SET_INTER   =  8, /* dim > 1, Intersection of two sets */
   SET_MINUS   =  9, /* dim > 1, Subtraction of two sets */
   SET_SYMDIFF = 10, /* dim > 1, Symetric difference of two sets */
   SET_TYPES   = SET_TYPE_COUNT  /* marker */
};

typedef enum set_type      SetType;
typedef struct set_vtab    SetVTab;

typedef struct set_empty   SetEmpty;
typedef struct set_pseudo  SetPseudo;
typedef struct set_head    SetHead;
typedef struct set_list    SetList;
typedef struct set_range   SetRange;
typedef struct set_prod    SetProd;
typedef struct set_multi   SetMulti;

typedef struct set_empty_iter   SetEmptyIter;
typedef struct set_pseudo_iter  SetPseudoIter;
typedef struct set_list_iter    SetListIter;
typedef struct set_range_iter   SetRangeIter;
typedef struct set_prod_iter    SetProdIter;
typedef struct set_multi_iter   SetMultiIter;

struct set_head
{
   int        refc;
   int        dim;
   SetIterIdx members;
   SetType    type;
};

struct set_empty
{
   SetHead head;
   SID
};

struct set_pseudo
{
   SetHead head;
   SID
};

struct set_list
{
   SetHead head;   /** head.dim == 1 */
   int     size;
   Elem**  member; /** head.members gives the number */
   Hash*   hash;
   SID
};

struct set_range
{
   SetHead head;   /** head.dim == 1 */
   int     begin;
   int     end;
   int     step;
   SID
};

struct set_prod
{
   SetHead head;   /** head.dim > 1 */
   Set*    set_a;
   Set*    set_b;
   SID
};

/* set consists of dim list set that contain the elements of the tuples to be stored
 * subset is an array with consists of    
 */
struct set_multi
{
   SetHead      head;   /* head.dim > 1  */
   Set**        set;    /* dim times, type == SET_LIST */
   SetIterIdx*  subset; /* members * dim */
   SetIterIdx** order;  /* dim * members */
   SID
};

union set
{
   SetHead    head;
   SetEmpty   empty;
   SetPseudo  pseudo;
   SetList    list;
   SetRange   range;
   SetProd    prod;
   SetMulti   multi;
};

struct set_empty_iter
{
   int dummy;
   SID
};

struct set_pseudo_iter
{
   bool first;
   SID
};

struct set_list_iter
{
   SetIterIdx first;
   SetIterIdx last;
   SetIterIdx now;
   SID
};

struct set_range_iter
{
   SetIterIdx first;
   SetIterIdx last;
   SetIterIdx now;
   SID
};

struct set_prod_iter
{
   bool     first;
   SetIter* iter_a;
   SetIter* iter_b;
   Elem**   elem;
   SID
};

struct set_multi_iter
{
   int         dim;
   SetIterIdx  members;
   SetIterIdx  now;
   SetIterIdx* subset;
   SID
};

union set_iter
{
   SetEmptyIter   empty;
   SetPseudoIter  pseudo;
   SetListIter    list;
   SetRangeIter   range;
   SetProdIter    prod;
   SetMultiIter   multi;
};

struct set_vtab
{
   expects_NONNULL                  void       (*set_free)      (Set* set);
   expects_NONNULL  returns_NONNULL Set*       (*set_copy)      (Set const* set);
   expects_NONNULL                  SetIterIdx (*set_lookup_idx)(Set const* set, Tuple const* tuple, int offset);
   expects_NONNULL                  void       (*set_get_tuple) (Set const* set, SetIterIdx idx, Tuple* tuple, int offset);
   expects_NONNULL1 returns_NONNULL SetIter*   (*iter_init)     (Set const* set, Tuple const* pattern, int offset);
   expects_NONNULL                  bool       (*iter_next)     (SetIter* iter, Set const* set, Tuple* tuple, int offset);
   expects_NONNULL                  void       (*iter_exit)     (SetIter* iter, Set const* set);
   expects_NONNULL                  void       (*iter_reset)    (SetIter* iter, Set const* set);
                                    bool       (*set_is_valid)  (Set const* set);
};

#define SET_DEFAULT 0x0
#define SET_NO_HASH 0x1


#ifdef NDEBUG
extern SetVTab* set_vtab_global;
#endif

/* set4.c
 */
//lint -sem(        set_lookup_idx, 1p == 1, 2p == 1, chneg(3), @n >= -1) 
extern SetIterIdx   set_lookup_idx(Set const* set, Tuple const* tuple, int offset) expects_NONNULL;
//lint -sem(        set_get_tuple_intern, 1p == 1, chneg(2), 3p == 1, chneg(4)) 
extern void         set_get_tuple_intern(Set const* set, SetIterIdx idx, Tuple* tuple, int offset) expects_NONNULL;
//lint -sem(        set_iter_init_intern, 1p == 1, chneg(3), @P >= malloc(1)) 
extern SetIter*     set_iter_init_intern(Set const* set, Tuple const* pattern, int offset) expects_NONNULL1 returns_NONNULL;
//lint -sem(        set_iter_next_intern, inout(1), 1p == 1, 2p == 1, inout(3), 3p == 1, chneg(4)) 
extern bool         set_iter_next_intern(SetIter* iter, Set const* set, Tuple* tuple, int offset) expects_NONNULL;
//lint -sem(        set_iter_exit_intern, custodial(1), inout(1), 1p == 1, 2p == 1) 
extern void         set_iter_exit_intern(SetIter* iter, Set const* set) expects_NONNULL;
//lint -sem(        set_iter_reset_intern, inout(1), 1p == 1, 2p == 1) 
extern void         set_iter_reset_intern(SetIter* iter, Set const* set) expects_NONNULL;

/* setempty.c
 */
//lint -sem(        set_empty_init, 1p == SET_TYPE_COUNT) 
extern void         set_empty_init(SetVTab* vtab) expects_NONNULL;
   
/* setpseudo.c
 */
//lint -sem(        set_pseudo_init, 1p == SET_TYPE_COUNT) 
extern void         set_pseudo_init(SetVTab* vtab) expects_NONNULL;
   
/* setlist.c
 */
//lint -sem(        set_list_init, 1p == SET_TYPE_COUNT) 
extern void         set_list_init(SetVTab* vtab) expects_NONNULL;
//lint -sem(        set_list_new, 1n > 0, chneg(2n), @P >= malloc(1)) 
extern Set*         set_list_new(int size, int flags) returns_NONNULL;
//lint -sem(        set_list_add_elem, inout(1), 1p == 1, 2p == 1, @n >= -1) 
extern SetIterIdx   set_list_add_elem(Set* set, Elem const* elem, SetCheckType check) expects_NONNULL;
//lint -sem(        set_list_new_from_elems, 1p == 1, @P >= malloc(1)) 
extern Set*         set_list_new_from_elems(List const* list, SetCheckType check) expects_NONNULL returns_NONNULL;
//lint -sem(        set_list_new_from_tuples, 1p == 1, @P >= malloc(1)) 
extern Set*         set_list_new_from_tuples(List const* list, SetCheckType check) expects_NONNULL  returns_NONNULL;
//lint -sem(        set_list_new_from_entries, 1p == 1, @P >= malloc(1)) 
extern Set*         set_list_new_from_entries(List const* list, SetCheckType check) expects_NONNULL  returns_NONNULL;
//lint -sem(        set_list_get_elem, 1p == 1, chneg(2), @p == 1) 
extern Elem const*  set_list_get_elem(Set const* set, SetIterIdx idx) expects_NONNULL returns_NONNULL is_PURE;

/* setrange.c
 */
//lint -sem(        set_range_init, 1p == SET_TYPE_COUNT) 
extern void         set_range_init(SetVTab* vtab) expects_NONNULL;

/* setprod.c
 */
//lint -sem(        set_prod_init, 1p == SET_TYPE_COUNT) 
extern void         set_prod_init(SetVTab* vtab) expects_NONNULL;

/* set multi.c
 */
//lint -sem(        set_multi_init, 1p == SET_TYPE_COUNT) 
extern void         set_multi_init(SetVTab* vtab) expects_NONNULL;
//lint -sem(        set_multi_new_from_list, 1p == 1, @P >= malloc(1)) 
extern Set*         set_multi_new_from_list(List const* list, SetCheckType check) expects_NONNULL returns_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _SET4_H_ 










