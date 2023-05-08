/**
 * @file DisallowCopyAssign.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2020-12-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

// Disallow the copy constructor and operator= functions.
// This should be used in the private declarations for a class.
#define DISALLOW_COPY_AND_ASSIGN(type_name) \
  type_name(const type_name&) = delete;     \
  void operator=(const type_name&) = delete
