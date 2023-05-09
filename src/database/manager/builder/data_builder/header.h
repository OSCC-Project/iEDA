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
/**
 * @project		iDB
 * @file		file_manager.h
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
 * @description


    There is a file manager to provides information description of binary files and read-write function of buffer level.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <string>
#include <vector>

#include "def_service.h"

namespace idb {

using std::string;
using std::vector;

#define kDbSuccess 0
#define kDbFail 1

#define CLOCKS_PER_MS 1000

enum class IdbFileHeaderType : uint8_t
{
  kNone = 0,
  kManufactureGrid,
  kUnits,
  kDie,
  kCore,
  kLayers,
  kSites,
  kRows,
  kGCellGridList,
  kTrackGridList,
  kCellMasterList,
  kVias,
  kViaRuleList,
  kMax
};

class IdbHeader
{
 public:
  IdbHeader() = default;
  ~IdbHeader() = default;

  // save header
  virtual void save_header() = 0;
  virtual void save_data() = 0;

  // getter
  inline const IdbFileHeaderType get_file_header_type() { return _type; }
  inline const char* get_file_path() { return _file_path; }

  // setter
  void set_type(IdbFileHeaderType type) { _type = type; }
  void set_file_path(const char* file_path) { _file_path = file_path; }

 private:
  IdbFileHeaderType _type;
  const char* _file_path;
};

class IdbManufactureGridHeader : public IdbHeader
{
 public:
  IdbManufactureGridHeader(IdbFileHeaderType type, const char* file_path, int32_t* manufacture_grid);
  ~IdbManufactureGridHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

  // operater
  void close_file() { fclose(_file_ptr); }

 private:
  FILE* _file_ptr;

  uint32_t _size;

  int32_t* _manufacture_grid;
};

class IdbUnitsHeader : public IdbHeader
{
 public:
  IdbUnitsHeader(IdbFileHeaderType type, const char* file_path, IdbUnits* units);
  ~IdbUnitsHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  uint32_t _size;

  IdbUnits* _units;
};

class IdbDieHeader : public IdbHeader
{
 public:
  IdbDieHeader(IdbFileHeaderType type, const char* file_path, IdbDie* die);
  ~IdbDieHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  IdbDie* _die;
};

class IdbCoreHeader : public IdbHeader
{
 public:
  IdbCoreHeader(IdbFileHeaderType type, const char* file_path, IdbCore* core);
  ~IdbCoreHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  IdbCore* _core;
};

class IdbLayersHeader : public IdbHeader
{
 public:
  IdbLayersHeader(IdbFileHeaderType type, const char* file_path, IdbLayers* layers);
  ~IdbLayersHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  uint32_t _layers_num;
  vector<uint32_t> _name_size;

  IdbLayers* _layers;
};

class IdbSitesHeader : public IdbHeader
{
 public:
  IdbSitesHeader(IdbFileHeaderType type, const char* file_path, IdbSites* sites);
  ~IdbSitesHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  uint32_t _site_num;
  vector<uint32_t> _site_name_size;

  IdbSites* _sites;
};

class IdbRowsHeader : public IdbHeader
{
 public:
  IdbRowsHeader(IdbFileHeaderType type, const char* file_path, IdbRows* rows, IdbSites* sites);
  ~IdbRowsHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  uint32_t _rows_num;

  vector<uint32_t> _site_name_size;

  vector<uint32_t> _row_name_size;

  IdbRows* _rows;
  IdbSites* _sites;
};

class IdbGcellGridHeader : public IdbHeader
{
 public:
  IdbGcellGridHeader(IdbFileHeaderType type, const char* file_path, IdbGCellGridList* gcell_grid);
  ~IdbGcellGridHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  uint32_t _grid_num = 0;

  IdbGCellGridList* _gcell_grid;
};

class IdbTrackGridHeader : public IdbHeader
{
 public:
  IdbTrackGridHeader(IdbFileHeaderType type, const char* file_path, IdbTrackGridList* track_grid_list, IdbLayers* layers);
  ~IdbTrackGridHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  uint32_t _track_grid_num;
  vector<uint8_t> _layers_num_list;
  vector<uint8_t> _layer_name_size;

  IdbTrackGridList* _track_grid_list;

  IdbLayers* _layers;
};

class IdbCellMasterHeader : public IdbHeader
{
 public:
  IdbCellMasterHeader(IdbFileHeaderType type, const char* file_path, IdbCellMasterList* cell_master_list, IdbLayers* layers);
  ~IdbCellMasterHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  uint32_t _master_num;

  vector<uint8_t> _masters_name_size;
  vector<uint32_t> _term_num_list;
  vector<uint32_t> _obs_num_list;

  vector<uint8_t> _term_name_size;
  vector<uint32_t> _term_port_num_list;

  vector<uint32_t> _term_layer_shape_num_list;

  vector<uint8_t> _term_layer_shape_layer_name_size;
  vector<uint32_t> _term_layer_shape_rect_num_list;

  vector<uint32_t> _obs_layer_num_list;

  vector<uint8_t> _obs_layer_shape_layer_name_size;
  vector<uint32_t> _obs_layer_shape_rect_num_list;

  IdbCellMasterList* _cell_master_list;
  IdbLayers* _layers;
};

class IdbViaListHeader : public IdbHeader
{
 public:
  IdbViaListHeader(IdbFileHeaderType type, const char* file_path, IdbVias* vias, IdbLayers* layers);
  ~IdbViaListHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  uint32_t _vias_num;
  vector<uint8_t> _via_name_size;
  vector<uint8_t> _via_instance_name_size;

  vector<uint8_t> _rule_name_size;

  vector<uint8_t> _rule_generate_name_size;

  vector<uint8_t> _rule_generate_layer_bottom_name_size;
  vector<uint8_t> _rule_generate_layer_cut_name_size;
  vector<uint8_t> _rule_generate_layer_top_name_size;

  vector<uint8_t> _master_generate_layer_bottom_name_size;
  vector<uint8_t> _master_generate_layer_cut_name_size;
  vector<uint8_t> _master_generate_layer_top_name_size;

  vector<uint8_t> _cut_rect_num_list;

  vector<uint32_t> _via_instance_master_fixed_num_list;

  vector<uint8_t> _master_fixed_layer_name_size;
  vector<uint32_t> _master_fixed_rect_num_list;

  vector<uint8_t> _master_instance_layer_shape_bottom_layer_name_size;
  vector<uint32_t> _master_instance_layer_shape_bottom_rect_num_list;
  vector<uint8_t> _master_instance_layer_shape_cut_layer_name_size;
  vector<uint32_t> _master_instance_layer_shape_cut_rect_num_list;
  vector<uint8_t> _master_instance_layer_shape_top_layer_name_size;
  vector<uint32_t> _master_instance_layer_shape_top_rect_num_list;

  IdbVias* _vias;
  IdbLayers* _layers;
};

class IdbViaRuleListHeader : public IdbHeader
{
 public:
  IdbViaRuleListHeader(IdbFileHeaderType type, const char* file_path, IdbViaRuleList* via_rule, IdbLayers* layers);
  ~IdbViaRuleListHeader() = default;

  // saver
  void save_header();
  void save_data();

  // loader
  void load_header();
  void load_data();

 private:
  FILE* _file_ptr;

  uint32_t _via_rule_generate_num;

  vector<uint8_t> _via_rule_name_size;

  vector<uint8_t> _layer_bottom_name_size;
  vector<uint8_t> _layer_cut_name_size;
  vector<uint8_t> _layer_top_name_size;

  IdbViaRuleList* _via_rule;
  IdbLayers* _layers;
};

}  // namespace idb
