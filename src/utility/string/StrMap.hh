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
 * @file StrMap.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2020-11-27
 */

#pragma once

#include "BTreeMap.hh"
#include "BTreeSet.hh"

namespace ieda {

/**
 * @brief The C-style string case lexical cmp.
 *
 */
struct StrCmp
{
  bool operator()(const char* const& lhs, const char* const& rhs) const;
};

/**
 * @brief The C-style string map.
 *
 * @tparam VALUE
 */
template <typename VALUE>
class StrMap : public BTreeMap<const char*, VALUE, StrCmp>
{
};

/**
 * @brief The C-style string set.
 *
 */
class StrSet : public BTreeSet<const char*, StrCmp>
{
};

}  // namespace ieda
