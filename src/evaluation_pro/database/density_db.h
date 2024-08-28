#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ieval {

using namespace ::std;

struct DensityPin
{
  string type;
  int32_t lx;
  int32_t ly;
};

struct DensityCell
{
  string type;
  int32_t lx;
  int32_t ly;
  int32_t width;
  int32_t height;
};

struct DensityNet
{
  int32_t lx;
  int32_t ly;
  int32_t ux;
  int32_t uy;
};

struct DensityRegion
{
  int32_t lx;
  int32_t ly;
  int32_t ux;
  int32_t uy;
};

using DensityPins = vector<DensityPin>;
using DensityCells = vector<DensityCell>;
using DensityNets = vector<DensityNet>;

struct CellMapSummary
{
  string macro_density;
  string stdcell_density;
  string allcell_density;
};

struct PinMapSummary
{
  string macro_pin_density;
  string stdcell_pin_density;
  string allcell_pin_density;
};

struct NetMapSummary
{
  string local_net_density;
  string global_net_density;
  string allnet_density;
};

struct CellReportSummary
{
  string macro_density;
  string stdcell_density;
  string allcell_density;
};

struct PinReportSummary
{
  string macro_pin_density;
  string stdcell_pin_density;
  string allcell_pin_density;
};

struct NetReportSummary
{
  string local_net_density;
  string global_net_density;
  string all_net_density;
};

}  // namespace ieval