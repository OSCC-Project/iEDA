/*
 * @FilePath: density_db.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ieval {

struct DensityPin
{
  std::string type;
  int32_t lx;
  int32_t ly;
};

struct DensityCell
{
  std::string type;
  int32_t lx;
  int32_t ly;
  int32_t width;
  int32_t height;
  int id;
};

struct DensityNet
{
  int32_t lx;
  int32_t ly;
  int32_t ux;
  int32_t uy;
  int id; 
};

struct DensityRegion
{
  int32_t lx;
  int32_t ly;
  int32_t ux;
  int32_t uy;
};

using DensityPins = std::vector<DensityPin>;
using DensityCells = std::vector<DensityCell>;
using DensityNets = std::vector<DensityNet>;

struct CellMapSummary
{
  std::string macro_density;
  std::string stdcell_density;
  std::string allcell_density;
};

struct PinMapSummary
{
  std::string macro_pin_density;
  std::string stdcell_pin_density;
  std::string allcell_pin_density;
};

struct NetMapSummary
{
  std::string local_net_density;
  std::string global_net_density;
  std::string allnet_density;
};

struct DensityMapSummary
{
  CellMapSummary cell_map_summary;
  PinMapSummary pin_map_summary;
  NetMapSummary net_map_summary;
};

struct MacroMarginSummary
{
  std::string horizontal_margin;
  std::string vertical_margin;
  std::string union_margin;
};

struct MacroCustomizedSummary
{
  MacroMarginSummary margin_summary;
  std::string macro_channel;
  std::string max_continuous_space;
  std::string macro_hierarchy;
};

}  // namespace ieval