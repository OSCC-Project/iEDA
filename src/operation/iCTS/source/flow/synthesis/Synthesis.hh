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
 * @file Synthesis.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS synthesis entry facade.
 */

#pragma once

#include "synthesis/trace/SynthesisTrace.hh"

namespace icts {

class CharacterizationLibrary;
class ClockLayout;
class Config;
class Design;
class FastSTA;
class SchemaWriter;
class Wrapper;

struct SynthesisInput
{
  const Config* config = nullptr;
  Design* design = nullptr;
  Wrapper* wrapper = nullptr;
  FastSTA* fast_sta = nullptr;
  SchemaWriter* reporter = nullptr;
  ClockLayout* clock_layout = nullptr;
  CharacterizationLibrary* characterization_library = nullptr;
};

class Synthesis
{
 public:
  Synthesis() = delete;

  static auto run(const SynthesisInput& input) -> SynthesisTraceSummary;
};

}  // namespace icts
