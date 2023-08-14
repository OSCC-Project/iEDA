#ifndef _UNISTD_H 
#define _UNISTD_H
 
/* This file intended to serve as a drop-in replacement for  
 *  unistd.h on Windows 
 *  Please add functionality as neeeded  
 */ 
 
#include <stdlib.h> 
#include <io.h> 
#include <getopt.h> /* getopt from: http://www.pwilson.net/sample.html. */ 

#ifdef __cplusplus
extern "C" {
#endif

#define srandom srand 
#define random rand 
 
#define W_OK 2
#define R_OK 4 
 
#define access _access 
#define ftruncate _chsize 

#define ssize_t int 
 
#ifdef __cplusplus
}
#endif
#endif /* unistd.h  */ 
