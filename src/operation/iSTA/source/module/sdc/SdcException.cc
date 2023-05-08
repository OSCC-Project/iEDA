/**
 * @file SdcException.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2022-07-18
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "SdcException.hh"

namespace ista {

SdcMulticyclePath::SdcMulticyclePath(int path_multiplier)
    : _path_multiplier(path_multiplier) {}

}  // namespace ista