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
#include <fstream>

#include "MPDB.hh"

namespace ipl::imp {
class GDSPlotter
{
 public:
  GDSPlotter(std::string path);
  ~GDSPlotter();
  void plotDie(FPRect* die, int layer = 0) { plotRect(die, layer); }
  void plotCore(FPRect* core, int layer = 1) { plotRect(core, layer); }
  void plotInstList(std::vector<FPInst*> inst_list, int layer);
  void plotNetList(std::vector<FPNet*> net_list, int layer);

  void plotInst(FPInst* inst, int layer);
  void plotRect(FPRect* rect, int layer);
  void plotLine(FPPin* start, FPPin* end, int layer);

 private:
  ofstream _gds_file;
};
}  // namespace ipl::imp