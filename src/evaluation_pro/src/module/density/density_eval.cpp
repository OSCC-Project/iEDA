#include "density_eval.h"

namespace ieval {

DensityEval::DensityEval()
{
}

DensityEval::~DensityEval()
{
}

std::string DensityEval::evalCellDensity()
{
  return "cell_density";
}

std::string DensityEval::evalPinDensity()
{
  return "pin_density";
}

std::string DensityEval::evalNetDensity()
{
  return "net_density";
}

std::string DensityEval::evalChannelDensity()
{
  return "channel_density";
}

std::string DensityEval::evalWhitespaceDensity()
{
  return "whitespace_density";
}
}  // namespace ieval
