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

namespace irt {

class RAGCellNode
{
 public:
  RAGCellNode() = default;
  RAGCellNode(const irt_int gcell_idx, const irt_int result_idx)
  {
    _gcell_idx = gcell_idx;
    _result_idx = result_idx;
  }
  ~RAGCellNode() = default;
  // getter
  irt_int get_gcell_idx() const { return _gcell_idx; }
  irt_int get_result_idx() const { return _result_idx; }

  // setter

  // function

 private:
  irt_int _gcell_idx = -1;
  irt_int _result_idx = -1;
};

}  // namespace irt
