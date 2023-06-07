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

GDSPlotter::GDSPlotter(std::string path)
{
  _gds_file.open(path);
  _gds_file << "HEADER 600" << std::endl;
  _gds_file << "BGNLIB" << std::endl;
  _gds_file << "LIBNAME DensityLib" << std::endl;
  _gds_file << "UNITS 0.001 1e-9" << std::endl;
  _gds_file << "BGNSTR" << std::endl;
  _gds_file << "STRNAME Die" << std::endl;
}

GDSPlotter::~GDSPlotter()
{
  _gds_file << "ENDSTR" << std::endl;
  _gds_file << "ENDLIB" << std::endl;
  _gds_file.close();
}

void GDSPlotter::plotInstList(std::vector<FPInst*> inst_list, int layer)
{
  for (FPInst* inst : inst_list) {
    plotInst(inst, layer);
  }
}

void GDSPlotter::plotNetList(std::vector<FPNet*> net_list, int layer)
{
  for (FPNet* net : net_list) {
    std::vector<FPPin*> pin_list = net->get_pin_list();
    FPPin* pin0 = pin_list[0];
    for (size_t i = 1; i < pin_list.size(); ++i) {
      plotLine(pin0, pin_list[i], layer);
    }
  }
}

void GDSPlotter::plotInst(FPInst* inst, int layer)
{
  _gds_file << "TEXT" << std::endl;
  _gds_file << "LAYER 1000" << std::endl;
  _gds_file << "TEXTTYPE 0" << std::endl;
  _gds_file << "XY" << std::endl;
  _gds_file << inst->get_center_x() << " : " << inst->get_center_y() << std::endl;
  _gds_file << "STRING " << inst->get_name() << std::endl;
  _gds_file << "ENDEL" << std::endl;
  plotRect(inst, layer);
}

void GDSPlotter::plotRect(FPRect* rect, int layer)
{
  int llx = int(rect->get_x());
  int lly = int(rect->get_y());
  int w = int(rect->get_width());
  int h = int(rect->get_height());
  _gds_file << "BOUNDARY" << std::endl;
  _gds_file << "LAYER " << layer << std::endl;
  _gds_file << "DATATYPE 0" << std::endl;
  _gds_file << "XY" << std::endl;
  _gds_file << llx << " : " << lly << std::endl;
  _gds_file << llx + w << " : " << lly << std::endl;
  _gds_file << llx + w << " : " << lly + h << std::endl;
  _gds_file << llx << " : " << lly + h << std::endl;
  _gds_file << llx << " : " << lly << std::endl;
  _gds_file << "ENDEL" << std::endl;
}

void GDSPlotter::plotLine(FPPin* start, FPPin* end, int layer)
{
  _gds_file << "PATH" << std::endl;
  _gds_file << "LAYER " << layer << std::endl;
  _gds_file << "DATATYPE 0" << std::endl;
  _gds_file << "WIDTH " << 20 << std::endl;
  _gds_file << "XY" << std::endl;
  _gds_file << int(start->get_x()) << ":" << int(start->get_y()) << std::endl;
  _gds_file << int(end->get_x()) << ":" << int(end->get_y()) << std::endl;
  _gds_file << "ENDEL" << std::endl;
}

}  // namespace ipl::imp