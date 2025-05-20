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
#include <boost/gil/typedefs.hpp>

#include "IdbInstance.h"
#include "idm.h"
#include "plot.hpp"
#include "report_basic.h"
#include "trie.hpp"
namespace iplf {
class ReportPlace : public ReportBase
{
 public:
  explicit ReportPlace(const std::string& report_name) : ReportBase(report_name) {}
  void createInstLevelReport(const std::string& prefix, int level, int num_threshold)
  {
    Trie<IdbInstance*> trie;
    auto* instlist = dmInst->get_idb_design()->get_instance_list();
    for (auto* inst : instlist->get_instance_list()) {
      trie.insert(inst->get_name(), inst);
    }
    auto instances = trie.nextLevel(prefix, level, num_threshold);
    for (auto& [prefix, cnt] : instances) {
      std::cout << cnt << "\t" << prefix << std::endl;
    }
  }
  void createInstDistributionReport(const std::vector<std::string>& prefixes, const std::string& file)
  {
    Trie<IdbInstance*> trie;

    auto* instlist = dmInst->get_idb_design()->get_instance_list();
    using namespace boost::gil;
    int scale = 500;
    auto height = dmInst->get_idb_layout()->get_die()->get_height();
    auto width = dmInst->get_idb_layout()->get_die()->get_width();

    int extend = width / 20;
    rgb8_image_t img((width + extend) / scale, height / scale);
    rgb8_pixel_t background(255, 255, 255);
    rgb8_pixel_t gray(245, 245, 245);
    rgb8_pixel_t border(220, 220, 220);
    fill_pixels(view(img), background);
    // auto rect_view = subimage_view(view(img), 10,10, 100,100);
    // fill_pixels(rect_view, rgb8_pixel_t(0,255,0));
    for (auto* inst : instlist->get_instance_list()) {
      trie.insert(inst->get_name(), inst);
      plotInstance(img, inst, scale, gray, border);
    }
    auto colors = getColorList(prefixes.size());
    for (size_t i = 0; i < prefixes.size(); ++i) {
      rgb8_pixel_t color = colors[i];
      trie.operateOnPrefix(prefixes[i], [&](IdbInstance* inst) { plotInstance(img, inst, scale, color); });
    }
    for (size_t i = 0; i < colors.size(); ++i) {
      int x = width;
      int y = height - (i + 1) * 200 * scale;
      drawRect(view(img), x / scale, y / scale, 200, 60, colors[i]);
    }
    std::string bmpfile = file.empty()? "inst_distro.bmp" : file;
    // write_view(bmpfile, flipped_up_down_view(view(img)), bmp_tag{});
    std::cout << "result saved to "<< bmpfile << std::endl;
  }

  template <typename View, typename Color>
  void drawRect(View img_view, int x, int y, int width, int height, Color color)
  {
    auto rect_view = subimage_view(img_view, x, y, width, height);
    fill_pixels(rect_view, color);
  }
  template <typename View, typename Color>
  void drawBorder(View img_view, int x, int y, int width, int height, Color color)
  {
    for (int i = x; i < x + width; ++i) {
      img_view(i, y) = color;
      img_view(i, y + height - 1) = color;
    }
    for (int i = y; i < y + height; ++i) {
      img_view(x, i) = color;
      img_view(x + width - 1, i) = color;
    }
  }

  template <typename Image, typename Color>
  void plotInstance(Image& img, IdbInstance* inst, int scale, Color fill)
  {
    plotInstance(img, inst, scale, fill, fill);
  }
  template <typename Image, typename Color>
  void plotInstance(Image& img, IdbInstance* inst, int scale, Color fill, Color border)
  {
    auto x0 = inst->get_bounding_box()->get_low_x();
    auto width = inst->get_bounding_box()->get_width();
    auto y0 = inst->get_bounding_box()->get_low_y();
    auto height = inst->get_bounding_box()->get_height();
    x0 = x0 / scale;
    width = (width + scale - 1) / scale;
    y0 = y0 / scale;
    height = (height + scale - 1) / scale;

    drawRect(view(img), x0, y0, width, height, fill);
    if (fill != border) {
      drawBorder(view(img), x0, y0, width, height, border);
    }
  }
};
}  // namespace iplf
