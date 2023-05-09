// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file StrMap.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2020-11-27
 */

#include "StrMap.hh"

#include "Str.hh"

namespace ieda {

/**
 * @brief The fuction for C-style string cmp.
 *
 * @param lhs
 * @param rhs
 * @return true if lhs < rhs
 * @return false if lhs >=rhs
 */
bool StrCmp::operator()(const char* const& lhs, const char* const& rhs) const
{
  return Str::caseCmp(lhs, rhs) < 0;  // if lhs == rhs, should be false to satify partial order.
}
}  // namespace ieda