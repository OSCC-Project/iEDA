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
 * @file Characterization.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Public characterization module contract for CTS segment and H-tree characterization.
 */

#pragma once

// IWYU pragma: begin_exports
#include "characterization/buffer_cell/CharacterizationBufferCell.hh"
#include "characterization/builder/CharBuilder.hh"
#include "characterization/pattern/PatternCombiner.hh"
#include "characterization/pruning/Frontier.hh"
#include "characterization/pruning/HTreeTraits.hh"
#include "characterization/pruning/HashJoinEngine.hh"
#include "characterization/pruning/SegmentTraits.hh"
#include "characterization/table/HTreeTopologyCharTable.hh"
#include "characterization/table/SegmentCharTable.hh"
// IWYU pragma: end_exports
