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
#pragma once

#include <time.h>

namespace idb {

// creation time and the last modification are now by default
class JsonTimestamp
{
 public:
  JsonTimestamp();
  explicit JsonTimestamp(time_t b, time_t l);

  void reset();

  time_t beg;
  time_t last;
};

/////////////// inline ///////////

inline JsonTimestamp::JsonTimestamp() : beg(time(nullptr)), last(time(nullptr))
{
}

inline JsonTimestamp::JsonTimestamp(time_t b, time_t l) : beg(b), last(l)
{
}

inline void JsonTimestamp::reset()
{
  beg = time(nullptr);
  last = time(nullptr);
}

}  // namespace idb