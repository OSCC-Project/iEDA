/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: prog.h                                                        */
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
#ifndef _PROG_H_
#define _PROG_H_

#ifndef _STMT_H_
#error "Need to include stmt.h before prog.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct program           Prog;

/* prog.c
 */
extern void*        prog_get_lp(void) is_PURE;
//lint -sem(        prog_new, @p) 
extern Prog*        prog_new(void) returns_NONNULL;
//lint -sem(        prog_free, custodial(1), inout(1), 1p == 1) 
extern void         prog_free(Prog* prog) expects_NONNULL;
//lint -sem(        prog_is_valid, pure, 1p == 1) 
extern bool         prog_is_valid(Prog const* prog) is_PURE;
//lint -sem(        prog_is_empty, pure, 1p == 1) 
extern bool         prog_is_empty(Prog const* prog) expects_NONNULL is_PURE;
//lint -sem(        prog_add_stmt, 1p == 1, custodial(2), 2p == 1) 
extern void         prog_add_stmt(Prog* prog, Stmt* stmt) expects_NONNULL;
//lint -sem(        prog_print, inout(1), 1p == 1, 2p == 1) 
extern void         prog_print(FILE* fp, Prog const* prog) expects_NONNULL;
//lint -sem(        prog_execute, 1p == 1) 
extern void         prog_execute(Prog const* prog, void* lp) expects_NONNULL;
//lint -sem(        prog_tostr, 1p == 1, 2p, 3p, @P >= malloc(1)) 
extern char*        prog_tostr(Prog const* prog, char const* prefix, char const* title, size_t max_output_line_len) expects_NONNULL returns_NONNULL;

/* load.c
 */
//lint -sem(        prog_load, inout(1), 1p == 1, 3p) 
extern void         prog_load(Prog* prog, char const* cmd, char const* filename);

#ifdef __cplusplus
}
#endif
#endif // _PROG_H_ 
