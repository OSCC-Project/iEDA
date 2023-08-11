/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: metaio.h                                                      */
/*   Name....: Meta Input/Output                                             */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2006-2022 by Thorsten Koch <koch@zib.de>
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
#ifndef _METAIO_H_
#define _METAIO_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct meta_file_ptr     MFP;

/* metaio.c
 */
//lint -sem(        mio_add_strg_file, 1p, 2p) 
extern void         mio_add_strg_file(char const* name, char const* content, bool use_copy) expects_NONNULL;
extern void         mio_init(void);
extern void         mio_exit(void);
//lint -sem(        mio_open, 1p, 2p) 
//lint -function(   fopen(1), mio_open(1)) 
//lint -function(   fopen(r), mio_open(r)) 
extern MFP*         mio_open(char const* name, char const* ext) expects_NONNULL;
//lint -sem(        mio_close, 1p == 1)   
//lint -function(   fclose, mio_close) 
extern void         mio_close(MFP* mfp) expects_NONNULL;
//lint -function(   fgetc, mio_getc) 
extern int          mio_getc(MFP const* mfp) expects_NONNULL;
//lint -function(   fgets(1), mio_gets(2)) 
//lint -function(   fgets(2), mio_gets(3)) 
//lint -function(   fgets(3), mio_gets(1)) 
//lint -function(   fgets(r), mio_gets(r)) 
extern char*        mio_gets(MFP const* mfp, char* buf, int len) expects_NONNULL;
//lint -sem(        mio_get_line, 1p == 1, r_null) 
extern char*        mio_get_line(MFP const* mfp) expects_NONNULL;

#ifdef __cplusplus
}
#endif
#endif // _METAIO_H_ 
