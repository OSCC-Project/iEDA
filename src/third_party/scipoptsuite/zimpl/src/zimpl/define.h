/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: define.h                                                      */
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
#ifndef _DEFINE_H_
#define _DEFINE_H_

#ifndef _NUMB_H_
#error "Need to include numb.h before define.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum define_type     { DEF_ERR = 0, DEF_NUMB, DEF_STRG, DEF_BOOL, DEF_SET };

typedef enum define_type         DefineType;
typedef struct define            Define;

//lint -sem(        define_new, 1p, @P >= malloc(1)) */
extern Define*      define_new(char const* name, DefineType type) expects_NONNULL returns_NONNULL;
//lint -sem(        define_set_param, inout(1), 1p == 1, custodial(2), 2p == 1) */
extern void         define_set_param(Define* def, Tuple* param) expects_NONNULL;
//lint -sem(        define_set_code, inout(1), 1p == 1, 2p == 1) */
extern void         define_set_code(Define* def, CodeNode* code) expects_NONNULL;
extern void         define_exit(void);
//lint -sem(        define_lookup, 1p, r_null) */
extern Define*      define_lookup(char const* name) expects_NONNULL;
//lint -sem(        define_get_name, 1p == 1, @p) */
extern char const*  define_get_name(Define const* def) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        define_get_type, 1p == 1) */
extern DefineType   define_get_type(Define const* def) expects_NONNULL is_PURE;
//lint -sem(        define_get_param, 1p == 1, r_null) */
extern Tuple const* define_get_param(Define const* def) expects_NONNULL returns_NONNULL is_PURE;
//lint -sem(        define_get_code, 1p == 1, r_null) */
extern CodeNode*    define_get_code(Define const* def) expects_NONNULL is_PURE;

#ifdef __cplusplus
}
#endif
#endif // _DEFINE_H_
