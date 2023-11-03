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

#include "JsonAref.hpp"
#include "JsonBoundary.hpp"
#include "JsonBox.hpp"
#include "JsonData.hpp"
#include "JsonNode.hpp"
#include "JsonPath.hpp"
#include "JsonSref.hpp"
#include "JsonText.hpp"
// #include "../builder/"

namespace idb {

// JSON-TXT writer
// JSON-TXT is a human-readable version of the JSON file format.
// To verify the grammar in the output file, KLayout is recommended.
class JsonTextWriter
{
 public:
  // constructor
  JsonTextWriter();
  JsonTextWriter(JsonData* data, const std::string txt = "");
  ~JsonTextWriter();

  // getter
  bool init(std::string txt, JsonData* data);
  bool close();

  bool begin(int i);
  bool finish(int i,std::vector<std::string> discard);

  // setter
  void writeTopStruct(int i);
  void writeStruct(int i);
  void write_sref_expension(int i,int num);
  void write_endlib(int i) const;
  void write_layerinfo(std::vector<std::string> layer_name,int i,int num);
  // function
  //   bool write();
  std::string fmt_time(time_t) const;
  void write_diearea(int i) const;

 private:
  // members
  JsonData* _data = nullptr;
  std::ofstream* _stream = nullptr;

  void flush();

  void write_header(int i) const;
  void write_bgnlib(int i) const;
  void write_libname(int i) const;
  void write_reflibs(int i) const;
  void write_fonts(int i) const;
  void write_attrtable(int i) const;
  void write_generations(int i) const;
  void write_format(int i) const;
  void write_units(int i) const;

  void write_bgnstr(JsonStruct*,int i) const;
  void write_strname(JsonStruct*,int i) const;
  void write_strclass(JsonStruct*,int i) const;
  void write_struct_element(JsonElemBase*,int i,bool out) const;
  void write_endstr(int i,int t) const;
  void write_element(JsonElement*,int i,bool out)const;
  void write_boundary(JsonBoundary*,int i,bool out) const;
  void write_path(JsonPath*,int i,bool out) const;
  void write_sref(JsonSref*,int i,bool out) const;
  void write_aref(JsonAref*,int i,bool out) const;
  void write_text(JsonText*,int i,bool out) const;
  void write_node(JsonNode*,int i,bool out) const;
  void write_box(JsonBox*,int i,bool out) const;
  void write_elflags(const JsonElemBase*,int i) const;
  void write_plex(const JsonElemBase*,int i) const;
  void write_property(JsonElemBase*,int i) const;
  void write_xy(JsonElemBase*,int i) const;
  void write_endel(int i) const;
  void write_strans(const JsonStrans&,int i) const;
  void write_layer(JsonLayer,int i) const;
};

}  // namespace idb