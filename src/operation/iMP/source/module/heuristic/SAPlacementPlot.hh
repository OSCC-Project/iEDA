/**
 * @file SAPlacementPlot.hh
 * @author Yuezuo Liu (yuezuoliu@163.com)
 * @brief
 * @version 0.1
 * @date 2023-11-24
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_SAPlacement_PLOT_H
#define IMP_SAPlacement_PLOT_H

#include <vector>

#include "SAPlacement.hh"

namespace imp {

template <typename CoordType, typename RepresentType>
struct SAPlacementPlot
{
  bool operator()(const std::string& filename, SAPlacement<CoordType, RepresentType>& SAPlacement);
  CoordType _ylim;
};

}  // namespace imp

#include "SAPlacementPlot.tpp"
#endif