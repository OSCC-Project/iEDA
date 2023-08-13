/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: zlpglue.h                                                     */
/*   Name....: Glue between Zimpl command line and ratlp                     */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2003-2022 by Thorsten Koch <koch@zib.de>
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
#ifndef _ZLPGLUE_H_
#define _ZLPGLUE_H_

#ifndef _RATLPTYPES_H_
#error "Need to include ratlptypes.h before xlpglue.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

//lint -sem(    zlp_stat, 1p == 1) 
extern void     zlp_stat(Lps const* lp) expects_NONNULL;
//lint -sem(    zlp_setnamelen, inout(1), 1p == 1, chneg(2)) 
extern void     zlp_setnamelen(Lps* lp, int name_len) expects_NONNULL;
//lint -sem(    zlp_write, 1p == 1, inout(2), 2p == 1, 4p, 5p) 
extern void     zlp_write(Lps const* lp, FILE* fp, LpFormat format, char const* format_options, char const* title) expects_NONNULL;
//lint -sem(    zlp_transtable, 1p == 1, inout(2), 2p == 1) 
extern void     zlp_transtable(Lps const* lp, FILE* fp, LpFormat format) expects_NONNULL;
//lint -sem(    zlp_orderfile, 1p == 1, inout(2), 2p == 1) 
extern void     zlp_orderfile(Lps const* lp, FILE* fp, LpFormat format) expects_NONNULL;
//lint -sem(    zlp_mstfile, 1p == 1, inout(2), 2p == 1) 
extern void     zlp_mstfile(Lps const* lp, FILE* fp, LpFormat format) expects_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _ZLPGLUE_H 








