/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: local.h                                                       */
/*   Name....: Local Parameter Functions                                     */
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
#ifndef _LOCAL_H_
#define _LOCAL_H_

#ifndef _ELEM_H_
#error "Need to include elem.h before local.h"
#endif

#ifndef _TUPLE_H_
#error "Need to include tuple.h before local.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern void         local_init(void);
extern void         local_exit(void);
extern void         local_drop_frame(void);
//lint -sem(        local_lookup, 1p, r_null) 
extern Elem const*  local_lookup(char const* name) expects_NONNULL is_PURE;
//lint -sem(        local_install_tuple, 1p == 1, 2p == 1) 
extern void         local_install_tuple(Tuple const* patt, Tuple const* vals) expects_NONNULL;
//lint -sem(        local_print_all, inout(1), 1p == 1) 
extern void         local_print_all(FILE* fp) expects_NONNULL;
//lint -sem(        local_tostrall, @P >= malloc(1)) 
extern char*        local_tostrall(void) returns_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _LOCAL_H_ 
