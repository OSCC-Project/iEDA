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
#include "GTWriter.hpp"

#include <fstream>

namespace idb {

GdsiiTextWriter::GdsiiTextWriter()
{
}

GdsiiTextWriter::GdsiiTextWriter(GdsData* data, const std::string txt)
{
  _data = data;
  _stream = new std::ofstream(txt, std::ios::out);
  if (_stream != nullptr && _stream->is_open()) {
    _stream->close();
    _stream = nullptr;
  }
}

GdsiiTextWriter::~GdsiiTextWriter()
{
  _data = nullptr;
}

bool GdsiiTextWriter::init(std::string txt, GdsData* data)
{
  if (txt.empty() || data == nullptr) {
    return false;
  }

  if (_stream != nullptr) {
    delete _stream;
  }

  _stream = new std::ofstream(txt, std::ios::out);
  if (_stream != nullptr && !_stream->is_open()) {
    _stream->close();
    _stream = nullptr;
    return false;
  }

  if (data != _data) {
    delete _data;
    _data = data;
  }

  return true;
}

bool GdsiiTextWriter::close()
{
  _stream->close();

  return true;
}

void GdsiiTextWriter::flush()
{
  _stream->flush();

  _data->clear_struct_list();
}

bool GdsiiTextWriter::begin()
{
  // <stream format>
  write_header();
  write_bgnlib();
  write_libname();
  write_reflibs();
  write_fonts();
  write_attrtable();
  write_generations();
  write_format();
  write_units();

  flush();

  return true;
}

bool GdsiiTextWriter::finish()
{
  writeStruct();

  /// @brief write top struct
  writeTopStruct();

  write_endlib();

  flush();

  close();

  return true;
}

// // @param data gds data
// // @param txt the name of GDS-TXT file
// bool GdsiiTextWriter::write()
// {
//   write_struct();

//   return true;
// }

void GdsiiTextWriter::write_header() const
{
  (*_stream) << "HEADER " << std::to_string(_data->get_header()) << std::endl;
}

void GdsiiTextWriter::write_bgnlib() const
{
  (*_stream) << "BGNLIB " << fmt_time(_data->get_bgn_lib()) << " " << fmt_time(_data->get_last_mod()) << std::endl;
}

std::string GdsiiTextWriter::fmt_time(time_t t) const
{
  const char* time_fmt = "%m/%d/%Y %H:%M:%S";
  char buf[64] = {0};
  struct tm bd_time;  // broken-down time
  localtime_r(&t, &bd_time);
  strftime(buf, sizeof(buf), time_fmt, &bd_time);
  return buf;
}

void GdsiiTextWriter::write_libname() const
{
  (*_stream) << "LIBNAME " << _data->get_lib_name() << std::endl;
}

void GdsiiTextWriter::write_reflibs() const
{
  auto libs = _data->get_ref_libs();
  if (libs.size() == 0)
    return;

  (*_stream) << "REFLIBS ";
  for (auto lib : libs) {
    (*_stream) << lib << "\n";
  }
}

void GdsiiTextWriter::write_fonts() const
{
  auto fonts = _data->get_fonts();
  if (fonts.size() == 0)
    return;

  (*_stream) << "FONTS ";
  for (auto font : fonts) {
    (*_stream) << font << "\n";
  }
}

void GdsiiTextWriter::write_attrtable() const
{
  auto attrtable = _data->get_attrtable();
  if (attrtable.size() == 0)
    return;

  (*_stream) << "ATTRTABLE " << attrtable << std::endl;
}

// This record contains a value to indicate
// the number of copies of deleted or back-up structures to retain.
// This numbermust be at least 2 and not more than 99.
// If the GENERATION record is omitted, a value of 3 is assumed.
void GdsiiTextWriter::write_generations() const
{
  auto g = _data->get_generations();
  if (g == 3)
    return;

  (*_stream) << "GENERATIONS " << g << std::endl;
}

void GdsiiTextWriter::write_format() const
{
  auto fmt = _data->get_format();
  if (fmt.type == GdsFormatType::kGDSII_Archive)
    return;

  (*_stream) << "FORMAT " << (int) fmt.type << "\n";

  if (!fmt.is_filtered())
    return;

  (*_stream) << "MASK\n"
             << fmt.mask << "\n"
             << "ENDMASKS" << std::endl;
}

void GdsiiTextWriter::write_units() const
{
  (*_stream) << "UNITS " << _data->get_unit().dbu_in_user() << " " << _data->get_unit().dbu_in_meter() << std::endl;
}

void GdsiiTextWriter::writeTopStruct()
{
  GdsStruct* str = _data->get_top_struct();
  write_bgnstr(str);
  write_strname(str);
  // write_strclass( str);
  for (GdsElemBase* e : str->get_element_list()) {
    write_struct_element(e);
  }
  write_endstr();

  /// @brief clear
  flush();
}

void GdsiiTextWriter::writeStruct()
{
  for (GdsStruct* str : _data->get_struct_list()) {
    // <structure>
    write_bgnstr(str);
    write_strname(str);
    // write_strclass( str);
    for (GdsElemBase* e : str->get_element_list()) {
      write_struct_element(e);
    }
    write_endstr();
  }

  /// @brief clear
  flush();
}

void GdsiiTextWriter::write_endlib() const
{
  (*_stream) << "ENDLIB" << std::endl;
}

void GdsiiTextWriter::write_bgnstr(GdsStruct* str) const
{
  if (!str)
    return;

  (*_stream) << "\nBGNSTR " << fmt_time(str->get_bgn_str()) << " " << fmt_time(str->get_last_mod()) << std::endl;
}

void GdsiiTextWriter::write_strname(GdsStruct* str) const
{
  if (!str)
    return;

  (*_stream) << "STRNAME " << str->get_name() << std::endl;
}

// Not used
// https://www.boolean.klaasholwerda.nl/interface/bnf/gdsformat.html#rec_strclass
void GdsiiTextWriter::write_strclass(GdsStruct* str) const
{
  if (!str)
    return;
  (*_stream) << "STRCLASS " << std::endl;
}

void GdsiiTextWriter::write_endstr() const
{
  (*_stream) << "ENDSTR" << std::endl;
}

void GdsiiTextWriter::write_struct_element(GdsElemBase* e) const
{
  if (!e)
    return;
  (*_stream) << std::endl;

  switch (e->get_elem_type()) {
    case GdsElemType::kElement:
      write_element(dynamic_cast<GdsElement*>(e));
      break;
    case GdsElemType::kBoundary:
      write_boundary(dynamic_cast<GdsBoundary*>(e));
      break;
    case GdsElemType::kPath:
      write_path(dynamic_cast<GdsPath*>(e));
      break;
    case GdsElemType::kSref:
      write_sref(dynamic_cast<GdsSref*>(e));
      break;
    case GdsElemType::kAref:
      write_aref(dynamic_cast<GdsAref*>(e));
      break;
    case GdsElemType::kText:
      write_text(dynamic_cast<GdsText*>(e));
      break;
    case GdsElemType::kNode:
      write_node(dynamic_cast<GdsNode*>(e));
      break;
    case GdsElemType::kBox:
      write_box(dynamic_cast<GdsBox*>(e));
      break;

    default:
      break;
  }
}

void GdsiiTextWriter::write_element(GdsElement* e) const
{
  if (!e)
    return;

  write_property(e);
  write_endel();
}

void GdsiiTextWriter::write_boundary(GdsBoundary* e) const
{
  if (!e)
    return;

  (*_stream) << "BOUNDARY\n";
  write_elflags(e);
  write_plex(e);
  write_layer(e->layer);
  (*_stream) << "DATATYPE " << e->data_type << "\n";
  write_xy(e);
  write_property(e);
  write_endel();
}

void GdsiiTextWriter::write_path(GdsPath* e) const
{
  if (!e)
    return;

  (*_stream) << "PATH\n";
  write_elflags(e);
  write_plex(e);
  write_layer(e->layer);
  (*_stream) << "DATATYPE " << e->data_type << "\n"
             << "PATHTYPE " << (int) e->path_type << "\n"
             << "WIDTH " << e->width << "\n";
  write_xy(e);
  write_property(e);
  write_endel();
}

void GdsiiTextWriter::write_sref(GdsSref* e) const
{
  if (!e)
    return;

  (*_stream) << "SREF\n";
  write_elflags(e);
  write_plex(e);
  (*_stream) << "SNAME " << e->sname << "\n";
  write_strans(e->strans);
  write_xy(e);
  write_property(e);
  write_endel();
}

void GdsiiTextWriter::write_aref(GdsAref* e) const
{
  if (!e)
    return;

  (*_stream) << "AREF\n";
  write_elflags(e);
  write_plex(e);
  (*_stream) << "SNAME " << e->sname << "\n";
  write_strans(e->strans);
  (*_stream) << "COLROW " << e->col << " " << e->row << "\n";
  write_xy(e);
  write_property(e);
  write_endel();
}

void GdsiiTextWriter::write_text(GdsText* e) const
{
  if (!e)
    return;

  (*_stream) << "TEXT\n";
  write_elflags(e);
  write_plex(e);
  write_layer(e->layer);
  (*_stream) << "TEXTTYPE " << e->text_type << "\n"
             << "PRESENTATION " << int(e->presentation) << "\n"
             << "PATHTYPE " << int(e->path_type) << "\n"
             << "WIDTH " << e->width << "\n";
  write_strans(e->strans);
  write_xy(e);
  (*_stream) << "STRING " << e->str << "\n";
  write_property(e);
  write_endel();
}

void GdsiiTextWriter::write_node(GdsNode* e) const
{
  if (!e)
    return;

  (*_stream) << "NODE\n";
  write_elflags(e);
  write_plex(e);
  write_layer(e->layer);
  (*_stream) << "NODETYPE " << e->node_type << "\n";
  write_xy(e);
  write_property(e);
  write_endel();
}

void GdsiiTextWriter::write_box(GdsBox* e) const
{
  if (!e)
    return;

  (*_stream) << "BOX\n";
  write_elflags(e);
  write_plex(e);
  write_layer(e->layer);
  (*_stream) << "BOXTYPE " << e->box_type << "\n";
  write_xy(e);
  write_property(e);
  write_endel();
}

// The document recommends setting attributes from 1 to 127,
// Since the "PROPATTR" is a two-byte signed integer,
// those property whose PROPATTR < 0 will not be written.
void GdsiiTextWriter::write_property(GdsElemBase* e) const
{
  if (!e)
    return;

  for (const auto& [attr, value] : e->get_property_map()) {
    if (attr < 0)
      continue;

    (*_stream) << "PROPATTR " << attr << "\n"
               << "PROPVALUE " << value << "\n";
  }
}

// quantity check in terms of
// https://www.boolean.klaasholwerda.nl/interface/bnf/gdsformat.html#rec_xy
void GdsiiTextWriter::write_xy(GdsElemBase* e) const
{
  if (!e)
    return;

  int min = 0;
  int max = 0;
  switch (e->get_elem_type()) {
    case GdsElemType::kElement:
      min = 0;
      max = 0;
      return;
    case GdsElemType::kBoundary:
      min = 4;
      max = 200;
      break;
    case GdsElemType::kPath:
      min = 2;
      max = 200;
      break;
    case GdsElemType::kSref:
      min = 1;
      max = 1;
      break;
    case GdsElemType::kAref:
      min = 3;
      max = 1;
      break;
    case GdsElemType::kText:
      min = 1;
      max = 1;
      break;
    case GdsElemType::kNode:
      min = 1;
      max = 50;
      break;
    case GdsElemType::kBox:
      min = 5;
      max = 5;
      break;

    default:
      return;
  }

  int num = e->get_xy().get_nums();
  assert(num);

  if (min > num)
    std::cout << "Warn: coordinate total is less than the expected"
              << ", GdsElemType =" << (int) e->get_elem_type() << std::endl;

  if (max < num)
    std::cout << "Warn: coordinate total is more than the expected"
              << ", GdsElemType =" << (int) e->get_elem_type() << std::endl;

  (*_stream) << "XY ";
  for (auto& xy : e->get_xy().get_coords()) {
    (*_stream) << xy.x << ": " << xy.y << "\n";
  }
}

void GdsiiTextWriter::write_endel() const
{
  (*_stream) << "ENDEL" << std::endl;
}

void GdsiiTextWriter::write_strans(const GdsStrans& strans) const
{
  (*_stream) << "STRANS " << strans.bit_flag << "\n"
             << "MAG " << strans.mag << "\n"
             << "ANGLE " << strans.angle << "\n";
}

void GdsiiTextWriter::write_elflags(const GdsElemBase* e) const
{
  if (!e)
    return;

  auto value = e->get_flags().get_value();
  if (value == 0)
    return;

  (*_stream) << "ELFLAGS " << value << std::endl;
}

void GdsiiTextWriter::write_plex(const GdsElemBase* e) const
{
  if (!e)
    return;

  auto value = e->get_plex();
  if (value == 0)
    return;

  (*_stream) << "PLEX " << value << std::endl;
}

// The document allows setting layer-value in the range of 0 to 255.
// So layer < 0 will be assert.
void GdsiiTextWriter::write_layer(GdsLayer layer) const
{
  assert(layer >= 0);
  (*_stream) << "LAYER " << layer << std::endl;
}

}  // namespace idb