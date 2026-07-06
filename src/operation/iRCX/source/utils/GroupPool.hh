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

#include <iterator>
#include <span>
#include <vector>

#include "Types.hh"

namespace ircx {

template <typename Item>
class GroupPool
{
 public:
  GroupPool() = default;
  ~GroupPool() = default;

  auto append(std::vector<Item> items) -> void
  {
    group_ranges_.push_back({items_.size(), items.size()});
    items_.insert(items_.end(),
                  std::make_move_iterator(items.begin()),
                  std::make_move_iterator(items.end()));
  }

  auto items(Size group_id) const -> std::span<const Item>
  {
    if (group_id >= group_ranges_.size()) {
      return {};
    }

    const Range& range = group_ranges_[group_id];
    if (range.length == 0) {
      return {};
    }
    return std::span<const Item>(items_.data() + range.offset, range.length);
  }

  auto items(Size group_id) -> std::span<Item>
  {
    if (group_id >= group_ranges_.size()) {
      return {};
    }

    const Range& range = group_ranges_[group_id];
    if (range.length == 0) {
      return {};
    }
    return std::span<Item>(items_.data() + range.offset, range.length);
  }

  auto groups() const -> Size
  {
    return group_ranges_.size();
  }

  auto item_count() const -> Size
  {
    return items_.size();
  }

  auto empty() const -> bool
  {
    return group_ranges_.empty();
  }

  auto reserve_groups(Size count) -> void
  {
    group_ranges_.reserve(count);
  }

  auto reserve_items(Size count) -> void
  {
    items_.reserve(count);
  }

  auto clear() -> void
  {
    items_.clear();
    group_ranges_.clear();
  }

 private:
  struct Range
  {
    Size offset{0};
    Size length{0};
  };

  std::vector<Item> items_;
  std::vector<Range> group_ranges_;
};

}  // namespace ircx
