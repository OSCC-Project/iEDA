/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: zimpllib.h                                                    */
/*   Name....: Zimpl library                                                 */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2005-2022 by Thorsten Koch <koch@zib.de>
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
#ifndef _ZIMPL_LIB_H_
#define _ZIMPL_LIB_H_

#ifdef __cplusplus
extern "C" {
#endif

//lint -sem(        zpl_add_parameter, 1p) 
extern void         zpl_add_parameter(char const* def) expects_NONNULL;
//lint -sem(        zpl_var_print, inout(1), 1p == 1, 2p == 1) 
extern void         zpl_var_print(FILE* fp, Var const* var) expects_NONNULL;
//lint -sem(        zpl_print_banner, inout(1), 1p == 1) 
extern void         zpl_print_banner(FILE* fp, bool with_license) expects_NONNULL;

//lint -sem(        zpl_read, 1p) 
extern bool         zpl_read(char const* filename, bool with_management, void* user_data) expects_NONNULL1;
//lint -sem(        zpl_read_with_args, 1p >= 2n, 2n > 0) 
extern bool         zpl_read_with_args(char** argv, int argc, bool with_management, void* user_data) expects_NONNULL1;

#ifdef __cplusplus
}
#endif

#endif // _ZIMPL_LIB_H_ 
