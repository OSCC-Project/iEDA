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

// #include <boost/gil/extension/io/bmp.hpp>
#include <boost/gil.hpp>

#include <boost/gil/extension/toolbox/image_types/indexed_image.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/typedefs.hpp>
#include <random>

namespace iplf {
const static int PREDEFINED_COLOR_NUM = 10;
const std::array<boost::gil::rgb8_pixel_t, PREDEFINED_COLOR_NUM> PREDEFINED_COLORS{
    boost::gil::rgb8_pixel_t(50, 205, 50),      boost::gil::rgb8_pixel_t(205, 104, 57), boost::gil::rgb8_pixel_t(148, 0, 211),
    boost::gil::rgb8_pixel_t(71, 60, 139),      boost::gil::rgb8_pixel_t(0, 139, 139),  boost::gil::rgb8_pixel_t(105, 139, 34),
    boost::gil::rgb8_pixel_t(218, 165, 32),     boost::gil::rgb8_pixel_t(255, 0, 255),  boost::gil::rgb8_pixel_t(244, 164, 96),
    boost::gil::rgb8_pixel_t(0xfa, 0xbe, 0xd4),
};

std::vector<boost::gil::rgb8_pixel_t> getColorList(int n)
{
  if (n <= PREDEFINED_COLOR_NUM) {
    return {PREDEFINED_COLORS.begin(), PREDEFINED_COLORS.begin() + n};
  }

  std::vector<boost::gil::rgb8_pixel_t> colors(n);
  for (size_t i = 0; i < PREDEFINED_COLORS.size(); ++i) {
    colors[i] = PREDEFINED_COLORS[i];
  }
  std::minstd_rand0 rand;
  auto rnd = [&rand]() { return rand() % 256; };
  for (int i = PREDEFINED_COLOR_NUM; i < n; ++i) {
    colors[i] = boost::gil::rgb8_pixel_t(rnd(), rnd(), rnd());
  }
  return colors;
}

}  // namespace iplf
