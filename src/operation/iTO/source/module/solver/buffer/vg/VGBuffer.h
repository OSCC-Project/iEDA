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

#include <iostream>

#include "BufferedOption.h"
#include "TreeBuild.h"

namespace ito {
class VGBuffer
{
 public:
  VGBuffer() = default;
  VGBuffer(TOLibertyCellSeq cells) : _available_lib_cell_sizes(cells){}
  ~VGBuffer() = default;

  BufferedOptionSeq VGBuffering(TreeBuild* tree);

 private:
  TOLibertyCellSeq _available_lib_cell_sizes;
  BufferedOptionSeq findBufferSolution(TreeBuild* tree, int curr_id, int prev_id);
  BufferedOptionSeq mergeBranch(BufferedOptionSeq buf_opt_left, BufferedOptionSeq buf_opt_right, Point curr_loc);
  BufferedOptionSeq addWire(BufferedOptionSeq buf_opt_seq, Point curr_loc, Point prev_loc);
  BufferedOptionSeq addBuffer(BufferedOptionSeq buf_opt_seq, Point prev_loc);
};

}  // namespace ito
