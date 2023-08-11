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
#ifndef _STRSTORE_H_
#define _STRSTORE_H_

#ifdef __cplusplus
extern "C" {
#endif

extern void         str_init(void);
extern void         str_exit(void);
//lint -sem(        str_new, 1p, @p) 
extern char const*  str_new(char const* s) returns_NONNULL;
//lint -sem(        str_hash, pure, 1p)           
extern unsigned int str_hash(char const* s) expects_NONNULL is_PURE;

#ifdef __cplusplus
}
#endif
#endif // _STRSTORE_H_ 
