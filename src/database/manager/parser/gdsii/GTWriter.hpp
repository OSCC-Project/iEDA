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

#include "GdsAref.hpp"
#include "GdsBoundary.hpp"
#include "GdsBox.hpp"
#include "GdsData.hpp"
#include "GdsNode.hpp"
#include "GdsPath.hpp"
#include "GdsSref.hpp"
#include "GdsText.hpp"

namespace idb {

// GDS-TXT writer
// GDS-TXT is a human-readable version of the GDSII file format.
class GdsiiTextWriter
{
 public:
  // constructor
  GdsiiTextWriter();
  GdsiiTextWriter(GdsData* data, const std::string txt = "");
  ~GdsiiTextWriter();

  // getter
  bool init(std::string txt, GdsData* data);
  bool close();

  bool begin();
  bool finish();

  // setter
  void writeTopStruct();
  void writeStruct();
  void write_endlib() const;
  // function
  //   bool write();
  std::string fmt_time(time_t) const;

 private:
  // members
  GdsData* _data = nullptr;
  std::ofstream* _stream = nullptr;

  void flush();

  void write_header() const;
  void write_bgnlib() const;
  void write_libname() const;
  void write_reflibs() const;
  void write_fonts() const;
  void write_attrtable() const;
  void write_generations() const;
  void write_format() const;
  void write_units() const;

  void write_bgnstr(GdsStruct*) const;
  void write_strname(GdsStruct*) const;
  void write_strclass(GdsStruct*) const;
  void write_struct_element(GdsElemBase*) const;
  void write_endstr() const;
  void write_element(GdsElement*) const;
  void write_boundary(GdsBoundary*) const;
  void write_path(GdsPath*) const;
  void write_sref(GdsSref*) const;
  void write_aref(GdsAref*) const;
  void write_text(GdsText*) const;
  void write_node(GdsNode*) const;
  void write_box(GdsBox*) const;
  void write_elflags(const GdsElemBase*) const;
  void write_plex(const GdsElemBase*) const;
  void write_property(GdsElemBase*) const;
  void write_xy(GdsElemBase*) const;
  void write_endel() const;
  void write_strans(const GdsStrans&) const;
  void write_layer(GdsLayer) const;
};

}  // namespace idb