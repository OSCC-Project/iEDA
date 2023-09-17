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
 * @File Name: guide_parser.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "guide_parser.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "contest_guide.h"
#include "util.h"

namespace ieda_contest {

bool GuideParser::parse(std::string guide_file, std::vector<ContestGuideNet>& guide_nets)
{
  guide_nets.clear();
  std::vector<ContestGuideNet>().swap(guide_nets);

  std::ifstream* guide_file_stream = getInputFileStream(guide_file);
  if (guide_file_stream == nullptr) {
    return false;
  }

  std::string new_line;
  bool is_new_net = true;

  std::string net_name, layer_name;
  while (getline(*guide_file_stream, new_line)) {
    int lb_x, lb_y, rt_x, rt_y;
    if (is_new_net) {
      net_name = new_line;
      is_new_net = false;
    } else if (new_line == "(") {
      std::vector<ContestGuide> guide_list;
      while (getline(*guide_file_stream, new_line)) {
        if (new_line == ")") {
          is_new_net = true;

          ContestGuideNet net;
          net.set_net_name(net_name);
          net.set_guide_list(guide_list);
          guide_nets.push_back(net);

          static int net_num = 0;
          net_num++;
          if (net_num % 10000 == 0) {
            std::cout << "read " << net_num << " nets" << std::endl;
          }
          break;
        }
        std::istringstream str(new_line);
        str >> lb_x >> lb_y >> rt_x >> rt_y >> layer_name;

        ContestGuide guide;
        guide.set_lb_x(lb_x);
        guide.set_lb_y(lb_y);
        guide.set_rt_x(rt_x);
        guide.set_rt_y(rt_y);
        guide.set_layer_name(layer_name);
        guide_list.push_back(guide);
      }
    }
  }
  closeFileStream(guide_file_stream);
  std::cout << "read guide file end" << std::endl;
  return true;
}

bool GuideParser::save(std::string guide_file, std::vector<ContestGuideNet>& guide_nets)
{
  std::ofstream* guide_file_stream = getOutputFileStream(guide_file);
  if (guide_file_stream == nullptr) {
    return false;
  }

  for (size_t i = 0; i < guide_nets.size(); i++) {
    ContestGuideNet& net = guide_nets[i];
    (*guide_file_stream) << net.get_net_name() << "\n(\n";
    for (ContestGuide& guide : net.get_guide_list()) {
      (*guide_file_stream) << guide.get_lb_x() << " " << guide.get_lb_y() << " " << guide.get_rt_x() << " " << guide.get_rt_y() << " "
                           << guide.get_layer_name() << "\n";
    }
    (*guide_file_stream) << ")\n";
    if ((i + 1) % 10000 == 0) {
      std::cout << "write " << (i + 1) << " nets" << std::endl;
    }
  }
  closeFileStream(guide_file_stream);
  std::cout << "write guide file end" << std::endl;
  return true;
}

}  // namespace ieda_contest
