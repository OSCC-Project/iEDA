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
 * @file GdsStream.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Minimal binary GDSII stream writer for CTS clock-tree visualization files.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "spatial/Point.hh"

namespace icts::visualization {

struct GdsLayerKey
{
  int16_t layer = 0;
  int16_t datatype = 0;
};

struct GdsLayerProperty
{
  GdsLayerKey key;
  std::string name;
  std::string color;
};

struct GdsPath
{
  GdsLayerKey key;
  int32_t width_dbu = 1;
  std::vector<Point<int>> points;
};

struct GdsBoundary
{
  GdsLayerKey key;
  std::vector<Point<int>> points;
};

struct GdsText
{
  GdsLayerKey key;
  Point<int> origin = Point<int>(0, 0);
  std::string text;
};

struct GdsLibrary
{
  std::string library_name = "CTS_REPORT";
  std::string structure_name = "CTS";
  int32_t dbu_per_um = 1000;
  std::vector<GdsPath> paths;
  std::vector<GdsBoundary> boundaries;
  std::vector<GdsText> texts;
};

class GdsStream
{
 public:
  GdsStream() = delete;

  static auto writeBinary(const std::filesystem::path& path, const GdsLibrary& library) -> bool;
  static auto writeLayerProperties(const std::filesystem::path& path, const std::vector<GdsLayerProperty>& layers) -> bool;
};

}  // namespace icts::visualization
