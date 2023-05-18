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

#include "GDSPlotter.hh"

namespace ipl::imp {
void GDSPlotter::plotInst(ofstream& gds_file, FPInst* inst, int layer)
{
  int llx = int(inst->get_x());
  int lly = int(inst->get_y());
  int w = int(inst->get_width());
  int h = int(inst->get_height());
  gds_file << "TEXT" << std::endl;
  gds_file << "LAYER 1000" << std::endl;
  gds_file << "TEXTTYPE 0" << std::endl;
  gds_file << "XY" << std::endl;
  gds_file << inst->get_center_x() << " : " << inst->get_center_y() << std::endl;
  gds_file << "STRING " << inst->get_name() << std::endl;
  gds_file << "ENDEL" << std::endl;
  gds_file << "BOUNDARY" << std::endl;
  gds_file << "LAYER " << layer << std::endl;
  gds_file << "DATATYPE 0" << std::endl;
  gds_file << "XY" << std::endl;
  gds_file << llx << " : " << lly << std::endl;
  gds_file << llx + w << " : " << lly << std::endl;
  gds_file << llx + w << " : " << lly + h << std::endl;
  gds_file << llx << " : " << lly + h << std::endl;
  gds_file << llx << " : " << lly << std::endl;
  gds_file << "ENDEL" << std::endl;
}

}  // namespace ipl::imp