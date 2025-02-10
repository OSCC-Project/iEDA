/**
 * @file kernel_common.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief gpu kernel common functions file.
 * @version 0.1
 * @date 2024-10-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <cstdlib>
#include <cstdio>

inline void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        printf("error: cpu memory allocation failed at %s:%d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define MALLOC_Check(size) malloc_check(size, __FILE__, __LINE__)