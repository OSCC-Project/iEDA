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
 * @project		iDB
 * @file		file_manager.cpp
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
* @description


        There is a file manager to provides information description of binary files and read-write function of buffer level.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "header.h"

namespace idb {

// ManufactureGridHeader

IdbManufactureGridHeader::IdbManufactureGridHeader(IdbFileHeaderType type, const char* file_path, int32_t* manufacture_grid) : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_manufacture_grid = manufacture_grid;
}

void IdbManufactureGridHeader::save_header()
{
  _size = sizeof(_manufacture_grid);

  _file_ptr = fopen(this->get_file_path(), "wb");

  fwrite(&_size, sizeof(_size), 1, _file_ptr);
}

void IdbManufactureGridHeader::save_data()
{
  fwrite(_manufacture_grid, sizeof(_manufacture_grid), 1, _file_ptr);

  fclose(_file_ptr);
}

void IdbManufactureGridHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");

  fread(&_size, sizeof(uint32_t), 1, _file_ptr);
}

void IdbManufactureGridHeader::load_data()
{
  fread(_manufacture_grid, sizeof(uint32_t), 1, _file_ptr);

  fclose(_file_ptr);
}

// IdbUnitsHeader

IdbUnitsHeader::IdbUnitsHeader(IdbFileHeaderType type, const char* file_path, IdbUnits* units) : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_units = units;
}

void IdbUnitsHeader::save_header()
{
  _size = sizeof(IdbUnits);

  _file_ptr = fopen(this->get_file_path(), "wb");

  fwrite(&_size, sizeof(_size), 1, _file_ptr);
}

void IdbUnitsHeader::save_data()
{
  fwrite(_units, sizeof(IdbUnits), 1, _file_ptr);

  fclose(_file_ptr);
}

void IdbUnitsHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");

  fread(&_size, sizeof(_size), 1, _file_ptr);
}

void IdbUnitsHeader::load_data()
{
  fread(_units, sizeof(IdbUnits), 1, _file_ptr);

  fclose(_file_ptr);
}

// IdbDieHeader
IdbDieHeader::IdbDieHeader(IdbFileHeaderType type, const char* file_path, IdbDie* die) : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_die = die;
}

void IdbDieHeader::save_header()
{
  _file_ptr = fopen(this->get_file_path(), "wb");
}

void IdbDieHeader::save_data()
{
  uint32_t point_num = _die->get_points().size();
  fwrite(&point_num, sizeof(int32_t), 1, _file_ptr);

  for (uint32_t i = 0; i < point_num; i++) {
    fwrite((*_die).get_points()[i], sizeof(IdbCoordinate<int32_t>), 1, _file_ptr);
  }

  uint64_t area = _die->get_area();
  fwrite(&area, sizeof(uint64_t), 1, _file_ptr);

  int32_t width = _die->get_width();
  fwrite(&width, sizeof(int32_t), 1, _file_ptr);

  int32_t height = _die->get_height();
  fwrite(&height, sizeof(int32_t), 1, _file_ptr);

  float utilization = _die->get_utilization();
  fwrite(&utilization, sizeof(float), 1, _file_ptr);

  fclose(_file_ptr);
}

void IdbDieHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");
}

void IdbDieHeader::load_data()
{
  int32_t point_num;
  fread(&point_num, sizeof(int32_t), 1, _file_ptr);
  _die->set_point_num(point_num);

  vector<IdbCoordinate<int32_t>*> points;
  for (int32_t i = 0; i < point_num; i++) {
    IdbCoordinate<int32_t>* point = new IdbCoordinate<int32_t>();
    fread(point, sizeof(IdbCoordinate<int32_t>), 1, _file_ptr);
    points.push_back(point);
  }
  _die->set_points(std::move(points));

  uint64_t area;
  fread(&area, sizeof(uint64_t), 1, _file_ptr);
  _die->set_area(area);

  int32_t width;
  fread(&width, sizeof(int32_t), 1, _file_ptr);
  _die->set_width(width);

  int32_t height;
  fread(&height, sizeof(int32_t), 1, _file_ptr);
  _die->set_height(height);

  _die->set_bounding_box();

  fclose(_file_ptr);
}

// IdbCore
IdbCoreHeader::IdbCoreHeader(IdbFileHeaderType type, const char* file_path, IdbCore* core) : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_core = core;
}

void IdbCoreHeader::save_header()
{
  _file_ptr = fopen(this->get_file_path(), "wb");
}

void IdbCoreHeader::save_data()
{
  IdbRect* bounding_box = _core->get_bounding_box();
  fwrite(bounding_box, sizeof(IdbRect), 1, _file_ptr);

  uint64_t id = _core->get_id();
  fwrite(&id, sizeof(uint64_t), 1, _file_ptr);

  fclose(_file_ptr);
}

void IdbCoreHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");
}

void IdbCoreHeader::load_data()
{
  IdbRect* bouding_box = new IdbRect();
  fread(bouding_box, sizeof(IdbRect), 1, _file_ptr);

  _core->set_bounding_box(bouding_box);

  uint64_t id;
  fread(&id, sizeof(uint64_t), 1, _file_ptr);
  _core->set_id(id);

  fclose(_file_ptr);
}

// IdbLayers
IdbLayersHeader::IdbLayersHeader(IdbFileHeaderType type, const char* file_path, IdbLayers* layers) : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_layers = layers;
}

void IdbLayersHeader::save_header()
{
  _file_ptr = fopen(this->get_file_path(), "wb");

  _layers_num = _layers->get_layers_num();
  fwrite(&_layers_num, sizeof(_layers_num), 1, _file_ptr);

  for (IdbLayer* layer : _layers->get_layers()) {
    uint32_t name_size = layer->get_name().size();
    fwrite(&name_size, sizeof(uint32_t), 1, _file_ptr);
  }
}

void IdbLayersHeader::save_data()
{
  for (IdbLayer* layer : _layers->get_layers()) {
    string name = layer->get_name();
    fwrite(name.c_str(), name.size(), 1, _file_ptr);

    IdbLayerType type = layer->get_type();
    fwrite(&type, sizeof(IdbLayerType), 1, _file_ptr);

    int8_t id = layer->get_id();
    fwrite(&id, sizeof(int8_t), 1, _file_ptr);

    switch (layer->get_type()) {
      case IdbLayerType::kLayerRouting: {
        IdbLayerRouting* routing = dynamic_cast<IdbLayerRouting*>(layer);

        int32_t width = routing->get_width();
        fwrite(&width, sizeof(int32_t), 1, _file_ptr);

        int32_t min_width = routing->get_min_width();
        fwrite(&min_width, sizeof(int32_t), 1, _file_ptr);

        int32_t max_width = routing->get_max_width();
        fwrite(&max_width, sizeof(int32_t), 1, _file_ptr);

        IdbLayerOrientValue pitch = routing->get_pitch();
        fwrite(&pitch, sizeof(IdbLayerOrientValue), 1, _file_ptr);

        IdbLayerOrientValue offset = routing->get_offset();
        fwrite(&offset, sizeof(IdbLayerOrientValue), 1, _file_ptr);

        IdbLayerDirection direction = routing->get_direction();
        fwrite(&direction, sizeof(IdbLayerDirection), 1, _file_ptr);

        int32_t wire_extension = routing->get_wire_extension();
        fwrite(&wire_extension, sizeof(wire_extension), 1, _file_ptr);

        int32_t thickness = routing->get_thickness();
        fwrite(&thickness, sizeof(int32_t), 1, _file_ptr);

        int32_t height = routing->get_height();
        fwrite(&height, sizeof(int32_t), 1, _file_ptr);

        int32_t area = routing->get_area();
        fwrite(&area, sizeof(int32_t), 1, _file_ptr);

        double resistance = routing->get_resistance();
        fwrite(&resistance, sizeof(double), 1, _file_ptr);

        double capacitance = routing->get_capacitance();
        fwrite(&capacitance, sizeof(double), 1, _file_ptr);

        double edge_capacitance = routing->get_edge_capacitance();
        fwrite(&edge_capacitance, sizeof(double), 1, _file_ptr);

        double min_density = routing->get_min_density();
        fwrite(&min_density, sizeof(double), 1, _file_ptr);

        double max_density = routing->get_max_density();
        fwrite(&max_density, sizeof(double), 1, _file_ptr);

        int32_t density_check_length = routing->get_density_check_length();
        fwrite(&density_check_length, sizeof(int32_t), 1, _file_ptr);

        int32_t density_check_width = routing->get_density_check_width();
        fwrite(&density_check_width, sizeof(int32_t), 1, _file_ptr);

        int32_t density_check_step = routing->get_density_check_step();
        fwrite(&density_check_step, sizeof(int32_t), 1, _file_ptr);

        int32_t min_cut_num = routing->get_min_cut_num();
        fwrite(&min_cut_num, sizeof(int32_t), 1, _file_ptr);

        int32_t min_cut_width = routing->get_min_cut_width();
        fwrite(&min_cut_width, sizeof(int32_t), 1, _file_ptr);

        uint32_t spacing_list_num = routing->get_spacing_list()->get_spacing_list_num();
        fwrite(&spacing_list_num, sizeof(uint32_t), 1, _file_ptr);

        for (IdbLayerSpacing* spacing : routing->get_spacing_list()->get_spacing_list()) {
          fwrite(spacing, sizeof(IdbLayerSpacing), 1, _file_ptr);
        }

        uint32_t area_list_num = routing->get_min_enclose_area_list()->get_min_area_list_num();
        fwrite(&area_list_num, sizeof(uint32_t), 1, _file_ptr);

        for (IdbMinEncloseArea min_enclose_area : routing->get_min_enclose_area_list()->get_min_area_list()) {
          fwrite(&min_enclose_area, sizeof(IdbMinEncloseArea), 1, _file_ptr);
        }

        break;
      }
      case IdbLayerType::kLayerCut: {
        IdbLayerCut* cut = dynamic_cast<IdbLayerCut*>(layer);

        int32_t width = cut->get_width();
        fwrite(&width, sizeof(int32_t), 1, _file_ptr);

        int32_t spacing = cut->get_spacing();
        fwrite(&spacing, sizeof(int32_t), 1, _file_ptr);

        IdbLayerCutArraySpacing* array_spacing = cut->get_array_spacing();

        bool is_long_array = array_spacing->is_long_array();
        fwrite(&is_long_array, sizeof(bool), 1, _file_ptr);

        int32_t cut_spacing = array_spacing->get_cut_spacing();
        fwrite(&cut_spacing, sizeof(int32_t), 1, _file_ptr);

        int32_t num_array_cut = array_spacing->get_array_cut_number();

        // TODO::PA have data error
        if (name == "PA") {
          num_array_cut = 0;
        }
        // TODO::PA have data error

        fwrite(&num_array_cut, sizeof(int32_t), 1, _file_ptr);

        for (IdbArrayCut array_cut : array_spacing->get_array_cut_list()) {
          fwrite(&array_cut, sizeof(IdbArrayCut), 1, _file_ptr);
        }

        IdbLayerCutEnclosure* enclosure_below = cut->get_enclosure_below();

        fwrite(enclosure_below, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);

        IdbLayerCutEnclosure* enclosure_above = cut->get_enclosure_above();

        fwrite(enclosure_above, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);

        break;
      }
      default:
        break;
    }
  }

  fclose(_file_ptr);
}

void IdbLayersHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");

  fread(&_layers_num, sizeof(_layers_num), 1, _file_ptr);

  for (uint32_t i = 0; i < _layers_num; i++) {
    uint32_t name_size;
    fread(&name_size, sizeof(uint32_t), 1, _file_ptr);
    _name_size.push_back(name_size);
  }
}

void IdbLayersHeader::load_data()
{
  for (uint32_t i = 0; i < _layers_num; i++) {
    IdbLayer* layer = new IdbLayer();

    char name[_name_size[i]];
    fread(name, _name_size[i], 1, _file_ptr);
    string name_str = name;
    name_str.resize(_name_size[i]);
    layer->set_name(name_str);

    IdbLayerType layer_type;
    fread(&layer_type, sizeof(IdbLayerType), 1, _file_ptr);
    layer->set_type(layer_type);

    int8_t layer_id;
    fread(&layer_id, sizeof(int8_t), 1, _file_ptr);
    layer->set_id(layer_id);

    switch (layer_type) {
      case IdbLayerType::kLayerRouting: {
        IdbLayerRouting* routing = new IdbLayerRouting();

        routing->set_name(layer->get_name());
        routing->set_type(layer->get_type());
        routing->set_id(layer->get_id());

        int32_t width;
        fread(&width, sizeof(int32_t), 1, _file_ptr);
        routing->set_width(width);

        int32_t min_width;
        fread(&min_width, sizeof(int32_t), 1, _file_ptr);
        routing->set_min_width(min_width);

        int32_t max_width;
        fread(&max_width, sizeof(int32_t), 1, _file_ptr);
        routing->set_max_width(max_width);

        IdbLayerOrientValue pitch;
        fread(&pitch, sizeof(IdbLayerOrientValue), 1, _file_ptr);
        routing->set_pitch(pitch);

        IdbLayerOrientValue offset;
        fread(&offset, sizeof(IdbLayerOrientValue), 1, _file_ptr);
        routing->set_offset(offset);

        IdbLayerDirection direction;
        fread(&direction, sizeof(IdbLayerDirection), 1, _file_ptr);
        routing->set_direction(direction);

        int32_t wire_extension;
        fread(&wire_extension, sizeof(wire_extension), 1, _file_ptr);
        routing->set_wire_extension(wire_extension);

        int32_t thickness;
        fread(&thickness, sizeof(int32_t), 1, _file_ptr);
        routing->set_thickness(thickness);

        int32_t height;
        fread(&height, sizeof(int32_t), 1, _file_ptr);
        routing->set_height(height);

        int32_t area;
        fread(&area, sizeof(int32_t), 1, _file_ptr);
        routing->set_area(area);

        double resistance;
        fread(&resistance, sizeof(double), 1, _file_ptr);
        routing->set_resistance(resistance);

        double capacitance;
        fread(&capacitance, sizeof(double), 1, _file_ptr);
        routing->set_capacitance(capacitance);

        double edge_capacitance;
        fread(&edge_capacitance, sizeof(double), 1, _file_ptr);
        routing->set_edge_capacitance(edge_capacitance);

        double min_density;
        fread(&min_density, sizeof(double), 1, _file_ptr);
        routing->set_min_density(min_density);

        double max_density;
        fread(&max_density, sizeof(double), 1, _file_ptr);
        routing->set_max_density(max_density);

        int32_t density_check_length;
        fread(&density_check_length, sizeof(int32_t), 1, _file_ptr);
        routing->set_density_check_length(density_check_length);

        int32_t density_check_width;
        fread(&density_check_width, sizeof(int32_t), 1, _file_ptr);
        routing->set_density_check_width(density_check_width);

        int32_t density_check_step;
        fread(&density_check_step, sizeof(int32_t), 1, _file_ptr);
        routing->set_density_check_step(density_check_step);

        int32_t min_cut_num;
        fread(&min_cut_num, sizeof(int32_t), 1, _file_ptr);
        routing->set_min_cut_num(min_cut_num);

        int32_t min_cut_width;
        fread(&min_cut_width, sizeof(int32_t), 1, _file_ptr);
        routing->set_min_cut_width(min_cut_width);

        uint32_t spacing_list_num;
        fread(&spacing_list_num, sizeof(uint32_t), 1, _file_ptr);

        IdbLayerSpacingList* spacing_list = new IdbLayerSpacingList();
        for (uint32_t j = 0; j < spacing_list_num; j++) {
          IdbLayerSpacing* spacing = new IdbLayerSpacing();
          fread(spacing, sizeof(IdbLayerSpacing), 1, _file_ptr);
          spacing_list->add_spacing(spacing);
        }
        routing->set_spacing_list(spacing_list);

        uint32_t area_list_num;
        fread(&area_list_num, sizeof(uint32_t), 1, _file_ptr);

        IdbMinEncloseAreaList* min_enclose_area_list = new IdbMinEncloseAreaList();
        for (uint32_t j = 0; j < area_list_num; j++) {
          IdbMinEncloseArea* min_enclose_area = new IdbMinEncloseArea();
          fread(min_enclose_area, sizeof(IdbMinEncloseArea), 1, _file_ptr);
          min_enclose_area_list->add_min_area(min_enclose_area->_area, min_enclose_area->_width);
        }
        routing->set_min_enclose_area_list(min_enclose_area_list);

        _layers->add_routing_layer(routing);

        IdbLayer* routing_to_layer = dynamic_cast<IdbLayer*>(routing);
        _layers->get_layers().push_back(routing_to_layer);

        delete layer;
        layer = nullptr;

        break;
      }
      case IdbLayerType::kLayerCut: {
        IdbLayerCut* cut = new IdbLayerCut();

        cut->set_name(layer->get_name());
        cut->set_type(layer->get_type());
        cut->set_id(layer->get_id());

        int32_t width;
        fread(&width, sizeof(int32_t), 1, _file_ptr);
        cut->set_width(width);

        int32_t spacing;
        fread(&spacing, sizeof(int32_t), 1, _file_ptr);
        cut->set_spacing(spacing);

        IdbLayerCutArraySpacing* array_spacing = new IdbLayerCutArraySpacing();

        bool is_long_array;
        fread(&is_long_array, sizeof(bool), 1, _file_ptr);
        array_spacing->set_long_array(is_long_array);

        int32_t cut_spacing;
        fread(&cut_spacing, sizeof(int32_t), 1, _file_ptr);
        array_spacing->set_cut_spacing(cut_spacing);

        int32_t num_array_cut;
        fread(&num_array_cut, sizeof(int32_t), 1, _file_ptr);
        array_spacing->set_array_cut_num(num_array_cut);

        for (int32_t j = 0; j < num_array_cut; j++) {
          IdbArrayCut array_cut;
          fread(&array_cut, sizeof(IdbArrayCut), 1, _file_ptr);
          array_spacing->get_array_cut_list()[j] = array_cut;
        }

        cut->set_array_spacing(array_spacing);

        IdbLayerCutEnclosure* enclosure_below = new IdbLayerCutEnclosure();

        fread(enclosure_below, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);

        cut->set_enclosure_below(enclosure_below);

        IdbLayerCutEnclosure* enclosure_above = new IdbLayerCutEnclosure();

        fread(enclosure_above, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);

        cut->set_enclosure_above(enclosure_above);

        _layers->add_cut_layer(cut);

        IdbLayer* cut_to_layer = dynamic_cast<IdbLayer*>(cut);
        _layers->get_layers().push_back(cut_to_layer);

        delete layer;
        layer = nullptr;

        break;
      }
      default:
        _layers->get_layers().push_back(layer);
        break;
    }
  }

  fclose(_file_ptr);
}

// IdbSites
IdbSitesHeader::IdbSitesHeader(IdbFileHeaderType type, const char* file_path, IdbSites* sites) : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_sites = sites;
}

void IdbSitesHeader::save_header()
{
  _file_ptr = fopen(this->get_file_path(), "wb");

  _site_num = _sites->get_sites_num();

  fwrite(&_site_num, sizeof(uint32_t), 1, _file_ptr);

  for (IdbSite* site : _sites->get_site_list()) {
    uint32_t length = site->get_name().size();
    fwrite(&length, sizeof(uint32_t), 1, _file_ptr);
    _site_name_size.push_back(length);
  }
}

void IdbSitesHeader::save_data()
{
  for (IdbSite* site : _sites->get_site_list()) {
    int32_t width = site->get_width();
    fwrite(&width, sizeof(int32_t), 1, _file_ptr);

    int32_t height = site->get_height();
    fwrite(&height, sizeof(int32_t), 1, _file_ptr);

    bool overlap = site->is_overlap();
    fwrite(&overlap, sizeof(bool), 1, _file_ptr);

    IdbSiteClass site_class = site->get_site_class();
    fwrite(&site_class, sizeof(IdbSiteClass), 1, _file_ptr);

    IdbSymmetry symmetry = site->get_symmetry();
    fwrite(&symmetry, sizeof(IdbSymmetry), 1, _file_ptr);

    IdbOrient orient = site->get_orient();
    fwrite(&orient, sizeof(IdbOrient), 1, _file_ptr);

    string name = site->get_name();
    fwrite(name.c_str(), name.size(), 1, _file_ptr);
  }

  fclose(_file_ptr);
}

void IdbSitesHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");

  fread(&_site_num, sizeof(uint32_t), 1, _file_ptr);

  for (uint32_t i = 0; i < _site_num; i++) {
    uint32_t size;
    fread(&size, sizeof(uint32_t), 1, _file_ptr);
    _site_name_size.push_back(size);
  }
}

void IdbSitesHeader::load_data()
{
  for (uint32_t i = 0; i < _site_num; i++) {
    IdbSite* site = new IdbSite();

    int32_t width;
    fread(&width, sizeof(int32_t), 1, _file_ptr);
    site->set_width(width);

    int32_t height;
    fread(&height, sizeof(int32_t), 1, _file_ptr);
    site->set_height(height);

    bool overlap;
    fread(&overlap, sizeof(bool), 1, _file_ptr);
    site->set_occupied(overlap);

    IdbSiteClass site_class;
    fread(&site_class, sizeof(IdbSiteClass), 1, _file_ptr);
    site->set_class(site_class);

    IdbSymmetry symmetry;
    fread(&symmetry, sizeof(IdbSymmetry), 1, _file_ptr);
    site->set_symmetry(symmetry);

    IdbOrient orient;
    fread(&orient, sizeof(IdbOrient), 1, _file_ptr);
    site->set_orient(orient);

    char name[_site_name_size[i]];
    fread(name, _site_name_size[i], 1, _file_ptr);
    string name_str = name;
    name_str.resize(_site_name_size[i]);
    site->set_name(name_str);

    _sites->add_site_list(site);
  }

  fclose(_file_ptr);
}

// IdbRows
IdbRowsHeader::IdbRowsHeader(IdbFileHeaderType type, const char* file_path, IdbRows* rows, IdbSites* sites) : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_rows = rows;
  this->_sites = sites;
}

void IdbRowsHeader::save_header()
{
  _file_ptr = fopen(this->get_file_path(), "wb");

  _rows_num = _rows->get_row_num();
  fwrite(&_rows_num, sizeof(uint32_t), 1, _file_ptr);

  for (IdbRow* row : _rows->get_row_list()) {
    uint32_t site_name_size = row->get_site()->get_name().size();
    fwrite(&site_name_size, sizeof(uint32_t), 1, _file_ptr);

    uint32_t row_name_size = row->get_name().size();
    fwrite(&row_name_size, sizeof(uint32_t), 1, _file_ptr);
  }
}

void IdbRowsHeader::save_data()
{
  for (IdbRow* row : _rows->get_row_list()) {
    string site_name = row->get_site()->get_name();
    fwrite(site_name.c_str(), site_name.size(), 1, _file_ptr);

    string name = row->get_name();
    fwrite(name.c_str(), name.size(), 1, _file_ptr);

    IdbCoordinate<int32_t>* original = row->get_original_coordinate();
    fwrite(original, sizeof(IdbCoordinate<int32_t>), 1, _file_ptr);

    int32_t row_num_x = row->get_row_num_x();
    fwrite(&row_num_x, sizeof(int32_t), 1, _file_ptr);

    int32_t row_num_y = row->get_row_num_y();
    fwrite(&row_num_y, sizeof(int32_t), 1, _file_ptr);

    int32_t step_x = row->get_step_x();
    fwrite(&step_x, sizeof(int32_t), 1, _file_ptr);

    int32_t step_y = row->get_step_y();
    fwrite(&step_y, sizeof(int32_t), 1, _file_ptr);
  }

  fclose(_file_ptr);
}

void IdbRowsHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");

  fread(&_rows_num, sizeof(uint32_t), 1, _file_ptr);

  for (uint32_t i = 0; i < _rows_num; i++) {
    uint32_t site_name_size;
    fread(&site_name_size, sizeof(uint32_t), 1, _file_ptr);
    _site_name_size.push_back(site_name_size);

    uint32_t row_name_size;
    fread(&row_name_size, sizeof(uint32_t), 1, _file_ptr);
    _row_name_size.push_back(row_name_size);
  }
}

void IdbRowsHeader::load_data()
{
  vector<IdbRow*> rows;
  for (uint32_t i = 0; i < _rows_num; i++) {
    IdbRow* row = new IdbRow();

    char site_name[_site_name_size[i]];
    fread(site_name, _site_name_size[i], 1, _file_ptr);
    string site_name_str = site_name;
    site_name_str.resize(_site_name_size[i]);
    row->set_site(_sites->find_site(site_name_str));

    char row_name[_row_name_size[i]];
    fread(row_name, _row_name_size[i], 1, _file_ptr);
    string row_name_str = row_name;
    row_name_str.resize(_row_name_size[i]);
    row->set_name(row_name_str);

    IdbCoordinate<int32_t>* original = new IdbCoordinate<int32_t>();
    fread(original, sizeof(IdbCoordinate<int32_t>), 1, _file_ptr);
    row->set_original_coordinate(original);

    int32_t row_num_x;
    fread(&row_num_x, sizeof(int32_t), 1, _file_ptr);
    row->set_row_num_x(row_num_x);

    int32_t row_num_y;
    fread(&row_num_y, sizeof(int32_t), 1, _file_ptr);
    row->set_row_num_y(row_num_y);

    int32_t step_x;
    fread(&step_x, sizeof(int32_t), 1, _file_ptr);
    row->set_step_x(step_x);

    int32_t step_y;
    fread(&step_y, sizeof(int32_t), 1, _file_ptr);
    row->set_step_y(step_y);

    row->set_bounding_box();

    _rows->add_row_list(row);
  }

  fclose(_file_ptr);
}

// IdbGcellGrid
IdbGcellGridHeader::IdbGcellGridHeader(IdbFileHeaderType type, const char* file_path, IdbGCellGridList* gcell_grid) : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_gcell_grid = gcell_grid;
}

void IdbGcellGridHeader::save_header()
{
  _file_ptr = fopen(this->get_file_path(), "wb");
}

void IdbGcellGridHeader::save_data()
{
  _grid_num = _gcell_grid->get_gcell_grid_num();
  fwrite(&_grid_num, sizeof(uint32_t), 1, _file_ptr);

  for (IdbGCellGrid* gcell_grid : _gcell_grid->get_gcell_grid_list()) {
    fwrite(gcell_grid, sizeof(IdbGCellGrid), 1, _file_ptr);
  }

  fclose(_file_ptr);
}

void IdbGcellGridHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");
}

void IdbGcellGridHeader::load_data()
{
  fread(&_grid_num, sizeof(uint32_t), 1, _file_ptr);

  for (int i = 0; i < _grid_num; i++) {
    IdbGCellGrid* gcell_grid = _gcell_grid->add_gcell_grid(nullptr);
    fread(gcell_grid, sizeof(IdbGCellGrid), 1, _file_ptr);
  }

  fclose(_file_ptr);
}

// IdbTrackGrid
IdbTrackGridHeader::IdbTrackGridHeader(IdbFileHeaderType type, const char* file_path, IdbTrackGridList* track_grid_list, IdbLayers* layers)
    : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_track_grid_list = track_grid_list;
  this->_layers = layers;
}

void IdbTrackGridHeader::save_header()
{
  _file_ptr = fopen(this->get_file_path(), "wb");

  _track_grid_num = _track_grid_list->get_track_grid_num();
  fwrite(&_track_grid_num, sizeof(uint32_t), 1, _file_ptr);

  for (IdbTrackGrid* track_grid : _track_grid_list->get_track_grid_list()) {
    uint8_t layers_num = track_grid->get_layer_list().size();
    fwrite(&layers_num, sizeof(uint8_t), 1, _file_ptr);

    for (IdbLayer* layer : track_grid->get_layer_list()) {
      uint8_t name_length = layer->get_name().size();
      fwrite(&name_length, sizeof(uint8_t), 1, _file_ptr);
    }
  }
}

void IdbTrackGridHeader::save_data()
{
  for (IdbTrackGrid* track_grid : _track_grid_list->get_track_grid_list()) {
    IdbTrack* track = track_grid->get_track();
    fwrite(track, sizeof(IdbTrack), 1, _file_ptr);

    uint32_t track_num = track_grid->get_track_num();
    fwrite(&track_num, sizeof(uint32_t), 1, _file_ptr);

    for (IdbLayer* layer : track_grid->get_layer_list()) {
      string name = layer->get_name();
      fwrite(name.c_str(), name.size(), 1, _file_ptr);
    }
  }

  fclose(_file_ptr);
}

void IdbTrackGridHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");

  fread(&_track_grid_num, sizeof(uint32_t), 1, _file_ptr);

  for (uint32_t i = 0; i < _track_grid_num; i++) {
    uint8_t layers_num;
    fread(&layers_num, sizeof(uint8_t), 1, _file_ptr);
    _layers_num_list.push_back(layers_num);

    for (uint32_t j = 0; j < layers_num; j++) {
      uint8_t name_length;
      fread(&name_length, sizeof(uint8_t), 1, _file_ptr);
      _layer_name_size.push_back(name_length);
    }
  }
}

void IdbTrackGridHeader::load_data()
{
  uint32_t layer_num_index = 0;

  for (uint32_t i = 0; i < _track_grid_num; i++) {
    IdbTrackGrid* track_grid = new IdbTrackGrid();

    IdbTrack* track = new IdbTrack();
    fread(track, sizeof(IdbTrack), 1, _file_ptr);
    track_grid->set_track(track);

    uint32_t track_num;
    fread(&track_num, sizeof(uint32_t), 1, _file_ptr);
    track_grid->set_track_number(track_num);

    for (uint8_t j = 0; j < _layers_num_list[i]; j++) {
      char layer_name[_layer_name_size[layer_num_index]];
      fread(layer_name, _layer_name_size[layer_num_index], 1, _file_ptr);
      string layer_name_str = layer_name;
      layer_name_str.resize(_layer_name_size[layer_num_index]);
      ++layer_num_index;

      IdbLayer* layer = _layers->find_layer(layer_name_str);
      track_grid->add_layer_list(layer);
      IdbLayerRouting* routing = dynamic_cast<IdbLayerRouting*>(layer);
      routing->add_track_grid(track_grid);
    }
    _track_grid_list->add_track_grid(track_grid);
  }

  fclose(_file_ptr);
}

// IdbCellMasterList
IdbCellMasterHeader::IdbCellMasterHeader(IdbFileHeaderType type, const char* file_path, IdbCellMasterList* cell_master_list,
                                         IdbLayers* layers)
    : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_cell_master_list = cell_master_list;
  this->_layers = layers;
}

void IdbCellMasterHeader::save_header()
{
  _file_ptr = fopen(this->get_file_path(), "wb");

  _master_num = _cell_master_list->get_cell_master_num();
  fwrite(&_master_num, sizeof(uint32_t), 1, _file_ptr);

  for (IdbCellMaster* cell_master : _cell_master_list->get_cell_master()) {
    uint8_t cell_master_name_length = cell_master->get_name().size();
    fwrite(&cell_master_name_length, sizeof(uint8_t), 1, _file_ptr);

    uint32_t term_num = cell_master->get_term_num();
    fwrite(&term_num, sizeof(term_num), 1, _file_ptr);

    for (IdbTerm* term : cell_master->get_term_list()) {
      uint8_t term_name_length = term->get_name().size();
      fwrite(&term_name_length, sizeof(uint8_t), 1, _file_ptr);

      uint32_t port_num = term->get_port_number();
      fwrite(&port_num, sizeof(uint32_t), 1, _file_ptr);

      for (IdbPort* port : term->get_port_list()) {
        uint32_t shape_num = port->get_layer_shape().size();
        fwrite(&shape_num, sizeof(uint32_t), 1, _file_ptr);

        for (IdbLayerShape* shape : port->get_layer_shape()) {
          uint8_t layer_name_size;
          if (shape->get_layer() == nullptr) {
            layer_name_size = 0;
          } else {
            layer_name_size = shape->get_layer()->get_name().size();
          }
          fwrite(&layer_name_size, sizeof(uint8_t), 1, _file_ptr);

          uint32_t rect_num = shape->get_rect_list_num();
          fwrite(&rect_num, sizeof(uint32_t), 1, _file_ptr);
        }
      }
    }

    uint32_t obs_num = cell_master->get_obs_list().size();
    fwrite(&obs_num, sizeof(uint32_t), 1, _file_ptr);

    for (IdbObs* obs : cell_master->get_obs_list()) {
      uint32_t obs_layer_num = obs->get_obs_layer_num();
      fwrite(&obs_layer_num, sizeof(uint32_t), 1, _file_ptr);

      for (IdbObsLayer* obs_layer : obs->get_obs_layer_list()) {
        uint8_t layer_shape_layer_name_size;
        if (obs_layer->get_shape()->get_layer() == nullptr) {
          layer_shape_layer_name_size = 0;
        } else {
          layer_shape_layer_name_size = obs_layer->get_shape()->get_layer()->get_name().size();
        }
        fwrite(&layer_shape_layer_name_size, sizeof(uint8_t), 1, _file_ptr);

        uint32_t rect_num = obs_layer->get_shape()->get_rect_list_num();
        fwrite(&rect_num, sizeof(uint32_t), 1, _file_ptr);
      }
    }
  }
}

void IdbCellMasterHeader::save_data()
{
  for (IdbCellMaster* cell_master : _cell_master_list->get_cell_master()) {
    string cell_master_name = cell_master->get_name();
    fwrite(cell_master_name.c_str(), cell_master_name.size(), 1, _file_ptr);

    CellMasterType type = cell_master->get_type();
    fwrite(&type, sizeof(CellMasterType), 1, _file_ptr);

    bool symmetry_x = cell_master->is_symmetry_x();
    fwrite(&symmetry_x, sizeof(bool), 1, _file_ptr);

    bool symmetry_y = cell_master->is_symmetry_y();
    fwrite(&symmetry_y, sizeof(bool), 1, _file_ptr);

    bool symmetry_R90 = cell_master->is_symmetry_R90();
    fwrite(&symmetry_R90, sizeof(bool), 1, _file_ptr);

    int64_t origin_x = cell_master->get_origin_x();
    fwrite(&origin_x, sizeof(int64_t), 1, _file_ptr);

    int64_t origin_y = cell_master->get_origin_y();
    fwrite(&origin_y, sizeof(int64_t), 1, _file_ptr);

    uint32_t width = cell_master->get_width();
    fwrite(&width, sizeof(uint32_t), 1, _file_ptr);

    uint32_t height = cell_master->get_height();
    fwrite(&height, sizeof(uint32_t), 1, _file_ptr);

    for (IdbTerm* term : cell_master->get_term_list()) {
      string term_name = term->get_name();
      fwrite(term_name.c_str(), term_name.size(), 1, _file_ptr);

      IdbConnectDirection direction = term->get_direction();
      fwrite(&direction, sizeof(IdbConnectDirection), 1, _file_ptr);

      IdbConnectType type = term->get_type();
      fwrite(&type, sizeof(IdbConnectType), 1, _file_ptr);

      IdbTermShape shape = term->get_shape();
      fwrite(&shape, sizeof(IdbTermShape), 1, _file_ptr);

      IdbPlacementStatus placement_status = term->get_placement_status();
      fwrite(&placement_status, sizeof(IdbPlacementStatus), 1, _file_ptr);

      for (IdbPort* port : term->get_port_list()) {
        for (IdbLayerShape* shape : port->get_layer_shape()) {
          IdbLayerShapeType type = shape->get_type();
          fwrite(&type, sizeof(IdbLayerShapeType), 1, _file_ptr);

          if (shape->get_layer() != nullptr) {
            string layer_name = shape->get_layer()->get_name();
            fwrite(layer_name.c_str(), layer_name.size(), 1, _file_ptr);
          }

          for (IdbRect* rect : shape->get_rect_list()) {
            fwrite(rect, sizeof(IdbRect), 1, _file_ptr);
          }
        }

        IdbPortClass port_class = port->get_port_class();
        fwrite(&port_class, sizeof(IdbPortClass), 1, _file_ptr);
      }

      IdbCoordinate<int32_t> average_position = term->get_average_position();
      fwrite(&average_position, sizeof(IdbCoordinate<int32_t>), 1, _file_ptr);

      IdbRect* bouding_box = term->get_bounding_box();
      fwrite(bouding_box, sizeof(IdbRect), 1, _file_ptr);
    }

    for (IdbObs* obs : cell_master->get_obs_list()) {
      for (IdbObsLayer* obs_layer : obs->get_obs_layer_list()) {
        IdbLayerShapeType layer_shape_type = obs_layer->get_shape()->get_type();
        fwrite(&layer_shape_type, sizeof(IdbLayerShapeType), 1, _file_ptr);

        if (obs_layer->get_shape()->get_layer() != nullptr) {
          string layer_shape_layer_name = obs_layer->get_shape()->get_layer()->get_name();
          fwrite(layer_shape_layer_name.c_str(), layer_shape_layer_name.size(), 1, _file_ptr);
        }

        for (IdbRect* rect : obs_layer->get_shape()->get_rect_list()) {
          fwrite(rect, sizeof(IdbRect), 1, _file_ptr);
        }
      }
    }
  }

  fclose(_file_ptr);
}

void IdbCellMasterHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");

  fread(&_master_num, sizeof(uint32_t), 1, _file_ptr);
  _cell_master_list->set_number(_master_num);

  for (uint32_t i = 0; i < _master_num; i++) {
    uint8_t cell_master_name_size;
    fread(&cell_master_name_size, sizeof(uint8_t), 1, _file_ptr);
    _masters_name_size.push_back(cell_master_name_size);

    uint32_t term_num;
    fread(&term_num, sizeof(uint32_t), 1, _file_ptr);
    _term_num_list.push_back(term_num);

    for (uint32_t j = 0; j < term_num; j++) {
      uint8_t term_name_size;
      fread(&term_name_size, sizeof(uint8_t), 1, _file_ptr);
      _term_name_size.push_back(term_name_size);

      uint32_t port_num;
      fread(&port_num, sizeof(uint32_t), 1, _file_ptr);
      _term_port_num_list.push_back(port_num);

      for (uint32_t k = 0; k < port_num; k++) {
        uint32_t shape_num;
        fread(&shape_num, sizeof(uint32_t), 1, _file_ptr);
        _term_layer_shape_num_list.push_back(shape_num);

        for (uint32_t l = 0; l < shape_num; l++) {
          uint8_t layer_name_size;
          fread(&layer_name_size, sizeof(uint8_t), 1, _file_ptr);
          _term_layer_shape_layer_name_size.push_back(layer_name_size);

          uint32_t rect_num;
          fread(&rect_num, sizeof(uint32_t), 1, _file_ptr);
          _term_layer_shape_rect_num_list.push_back(rect_num);
        }
      }
    }

    uint32_t obs_num;
    fread(&obs_num, sizeof(uint32_t), 1, _file_ptr);
    _obs_num_list.push_back(obs_num);

    for (uint32_t j = 0; j < obs_num; j++) {
      uint32_t obs_layer_num;
      fread(&obs_layer_num, sizeof(uint32_t), 1, _file_ptr);
      _obs_layer_num_list.push_back(obs_layer_num);

      for (uint32_t k = 0; k < obs_layer_num; k++) {
        uint8_t obs_layer_shape_layer_name_size;
        fread(&obs_layer_shape_layer_name_size, sizeof(uint8_t), 1, _file_ptr);
        _obs_layer_shape_layer_name_size.push_back(obs_layer_shape_layer_name_size);

        uint32_t obs_layer_shape_rect_num;
        fread(&obs_layer_shape_rect_num, sizeof(uint32_t), 1, _file_ptr);
        _obs_layer_shape_rect_num_list.push_back(obs_layer_shape_rect_num);
      }
    }
  }
}

void IdbCellMasterHeader::load_data()
{
  uint32_t term_name_index, term_port_num_index, term_layer_shape_num_index, term_layer_shape_layer_name_size_index, term_rect_num_index,
      obs_layer_num_index, obs_layer_shape_layer_name_size_index, _obs_layer_shape_rect_num_index;
  term_name_index = term_port_num_index = term_layer_shape_num_index = term_layer_shape_layer_name_size_index = term_rect_num_index
      = obs_layer_num_index = obs_layer_shape_layer_name_size_index = _obs_layer_shape_rect_num_index = 0;

  for (uint32_t i = 0; i < _master_num; i++) {
    IdbCellMaster* cell_master = new IdbCellMaster();

    char cell_master_name[_masters_name_size[i]];
    fread(cell_master_name, _masters_name_size[i], 1, _file_ptr);
    string cell_master_name_str = cell_master_name;
    cell_master_name_str.resize(_masters_name_size[i]);
    cell_master->set_name(cell_master_name_str);

    CellMasterType type;
    fread(&type, sizeof(CellMasterType), 1, _file_ptr);
    cell_master->set_type(type);

    bool symmetry_x;
    fread(&symmetry_x, sizeof(bool), 1, _file_ptr);
    cell_master->set_symmetry_x(symmetry_x);

    bool symmetry_y;
    fread(&symmetry_y, sizeof(bool), 1, _file_ptr);
    cell_master->set_symmetry_y(symmetry_y);

    bool symmetry_R90;
    fread(&symmetry_R90, sizeof(bool), 1, _file_ptr);
    cell_master->set_symmetry_R90(symmetry_R90);

    int64_t origin_x;
    fread(&origin_x, sizeof(int64_t), 1, _file_ptr);
    cell_master->set_origin_x(origin_x);

    int64_t origin_y;
    fread(&origin_y, sizeof(int64_t), 1, _file_ptr);
    cell_master->set_origin_y(origin_y);

    uint32_t width;
    fread(&width, sizeof(uint32_t), 1, _file_ptr);
    cell_master->set_width(width);

    uint32_t height;
    fread(&height, sizeof(uint32_t), 1, _file_ptr);
    cell_master->set_height(height);

    for (uint32_t j = 0; j < _term_num_list[i]; j++) {
      IdbTerm* term = new IdbTerm();

      char term_name[_term_name_size[term_name_index]];
      fread(term_name, _term_name_size[term_name_index], 1, _file_ptr);
      string term_name_str = term_name;
      term_name_str.resize(_term_name_size[term_name_index]);
      ++term_name_index;

      term->set_name(term_name_str);

      IdbConnectDirection direction;
      fread(&direction, sizeof(IdbConnectDirection), 1, _file_ptr);
      term->set_direction(direction);

      IdbConnectType type;
      fread(&type, sizeof(IdbConnectType), 1, _file_ptr);
      term->set_type(type);

      IdbTermShape shape;
      fread(&shape, sizeof(IdbTermShape), 1, _file_ptr);
      term->set_shape(shape);

      IdbPlacementStatus placement_status;
      fread(&placement_status, sizeof(IdbPlacementStatus), 1, _file_ptr);
      term->set_placement_status(placement_status);

      for (uint32_t k = 0; k < _term_port_num_list[term_port_num_index]; k++) {
        IdbPort* port = new IdbPort();

        for (uint32_t l = 0; l < _term_layer_shape_num_list[term_layer_shape_num_index]; l++) {
          IdbLayerShape* layer_shape = new IdbLayerShape();

          IdbLayerShapeType type;
          fread(&type, sizeof(IdbLayerShapeType), 1, _file_ptr);
          layer_shape->set_type_rect();

          char layer_shape_layer_name[_term_layer_shape_layer_name_size[term_layer_shape_layer_name_size_index]];
          fread(layer_shape_layer_name, _term_layer_shape_layer_name_size[term_layer_shape_layer_name_size_index], 1, _file_ptr);
          string layer_shape_layer_name_str = layer_shape_layer_name;
          layer_shape_layer_name_str.resize(_term_layer_shape_layer_name_size[term_layer_shape_layer_name_size_index]);
          ++term_layer_shape_layer_name_size_index;

          layer_shape->set_layer(_layers->find_layer(layer_shape_layer_name_str));

          for (uint32_t m = 0; m < _term_layer_shape_rect_num_list[term_rect_num_index]; m++) {
            IdbRect* rect = new IdbRect();

            fread(rect, sizeof(IdbRect), 1, _file_ptr);

            layer_shape->add_rect(rect);
          }
          ++term_rect_num_index;

          port->add_layer_shape(layer_shape);
        }

        IdbPortClass port_class;
        fread(&port_class, sizeof(IdbPortClass), 1, _file_ptr);
        port->set_port_class(port_class);

        term->add_port(port);
      }
      ++term_port_num_index;

      IdbCoordinate<int32_t> average_position;
      fread(&average_position, sizeof(IdbCoordinate<int32_t>), 1, _file_ptr);
      term->set_average_position(average_position.get_x(), average_position.get_y());

      IdbRect* bouding_box = new IdbRect();
      fread(bouding_box, sizeof(IdbRect), 1, _file_ptr);
      term->set_bounding_box(bouding_box->get_low_x(), bouding_box->get_low_y(), bouding_box->get_high_x(), bouding_box->get_high_y());

      cell_master->add_term(term);
    }

    for (uint32_t j = 0; j < _obs_num_list[i]; j++) {
      IdbObs* obs = new IdbObs();

      for (uint32_t k = 0; k < _obs_layer_num_list[obs_layer_num_index]; k++) {
        IdbObsLayer* obs_layer = new IdbObsLayer();

        IdbLayerShape* layer_shape = new IdbLayerShape();

        IdbLayerShapeType layer_shape_type;
        fread(&layer_shape_type, sizeof(IdbLayerShapeType), 1, _file_ptr);
        layer_shape->set_type_rect();

        char layer_shape_layer_name[_obs_layer_shape_layer_name_size[obs_layer_shape_layer_name_size_index]];
        fread(layer_shape_layer_name, _obs_layer_shape_layer_name_size[obs_layer_shape_layer_name_size_index], 1, _file_ptr);
        string layer_shape_layer_name_str = layer_shape_layer_name;
        layer_shape_layer_name_str.resize(_obs_layer_shape_layer_name_size[obs_layer_shape_layer_name_size_index]);
        ++obs_layer_shape_layer_name_size_index;

        layer_shape->set_layer(_layers->find_layer(layer_shape_layer_name_str));

        for (uint32_t l = 0; l < _obs_layer_shape_rect_num_list[_obs_layer_shape_rect_num_index]; l++) {
          IdbRect* rect = new IdbRect();

          fread(rect, sizeof(IdbRect), 1, _file_ptr);

          layer_shape->add_rect(rect);
        }
        ++_obs_layer_shape_rect_num_index;

        obs_layer->set_shape(layer_shape);

        obs->add_obs_layer(obs_layer);
      }
      ++obs_layer_num_index;

      cell_master->add_obs(obs);
    }

    _cell_master_list->get_cell_master().push_back(cell_master);
  }

  fclose(_file_ptr);
}

// IdbVias
IdbViaListHeader::IdbViaListHeader(IdbFileHeaderType type, const char* file_path, IdbVias* vias, IdbLayers* layers) : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_vias = vias;
  this->_layers = layers;
}

void IdbViaListHeader::save_header()
{
  _file_ptr = fopen(this->get_file_path(), "wb");

  _vias_num = _vias->get_num_via();
  fwrite(&_vias_num, sizeof(uint32_t), 1, _file_ptr);

  for (IdbVia* via : _vias->get_via_list()) {
    uint8_t via_name_size = via->get_name().size();
    fwrite(&via_name_size, sizeof(uint8_t), 1, _file_ptr);

    uint8_t via_instance_name_size = via->get_instance()->get_name().size();
    fwrite(&via_instance_name_size, sizeof(uint8_t), 1, _file_ptr);

    uint8_t rule_name_size = via->get_instance()->get_master_generate()->get_rule_name().size();
    fwrite(&rule_name_size, sizeof(uint8_t), 1, _file_ptr);

    bool is_rule_generate_null = via->get_instance()->get_master_generate()->get_rule_generate() == nullptr ? 1 : 0;
    uint8_t rule_generate_name_size, rule_generate_layer_bottom_name_size, rule_generate_layer_cut_name_size,
        rule_generate_layer_top_name_size;
    if (is_rule_generate_null) {
      rule_generate_name_size = rule_generate_layer_bottom_name_size = rule_generate_layer_cut_name_size = rule_generate_layer_top_name_size
          = 0;
    } else {
      rule_generate_name_size = via->get_instance()->get_master_generate()->get_rule_generate()->get_name().size();

      rule_generate_layer_bottom_name_size
          = via->get_instance()->get_master_generate()->get_rule_generate()->get_layer_bottom()->get_name().size();

      rule_generate_layer_cut_name_size
          = via->get_instance()->get_master_generate()->get_rule_generate()->get_layer_cut()->get_name().size();

      rule_generate_layer_top_name_size
          = via->get_instance()->get_master_generate()->get_rule_generate()->get_layer_top()->get_name().size();
    }
    fwrite(&rule_generate_name_size, sizeof(uint8_t), 1, _file_ptr);
    fwrite(&_rule_generate_layer_bottom_name_size, sizeof(uint8_t), 1, _file_ptr);
    fwrite(&_rule_generate_layer_cut_name_size, sizeof(uint8_t), 1, _file_ptr);
    fwrite(&_rule_generate_layer_top_name_size, sizeof(uint8_t), 1, _file_ptr);

    bool is_layer_bottom_null = via->get_instance()->get_master_generate()->get_layer_bottom() == nullptr ? 1 : 0;
    uint8_t master_generate_layer_bottom_name_size;
    if (is_layer_bottom_null) {
      master_generate_layer_bottom_name_size = 0;
    } else {
      master_generate_layer_bottom_name_size = via->get_instance()->get_master_generate()->get_layer_bottom()->get_name().size();
    }
    fwrite(&master_generate_layer_bottom_name_size, sizeof(uint8_t), 1, _file_ptr);

    bool is_layer_cut_null = via->get_instance()->get_master_generate()->get_layer_cut() == nullptr ? 1 : 0;
    uint8_t master_generate_layer_cut_name_size;
    if (is_layer_cut_null) {
      master_generate_layer_cut_name_size = 0;
    } else {
      master_generate_layer_cut_name_size = via->get_instance()->get_master_generate()->get_layer_cut()->get_name().size();
    }
    fwrite(&master_generate_layer_cut_name_size, sizeof(uint8_t), 1, _file_ptr);

    bool is_layer_top_null = via->get_instance()->get_master_generate()->get_layer_top() == nullptr ? 1 : 0;
    uint8_t master_generate_layer_top_name_size;
    if (is_layer_top_null) {
      master_generate_layer_top_name_size = 0;
    } else {
      master_generate_layer_top_name_size = via->get_instance()->get_master_generate()->get_layer_top()->get_name().size();
    }
    fwrite(&master_generate_layer_top_name_size, sizeof(uint8_t), 1, _file_ptr);

    uint32_t cut_rect_num = via->get_instance()->get_master_generate()->get_cut_rect_list().size();
    fwrite(&cut_rect_num, sizeof(uint32_t), 1, _file_ptr);

    uint32_t master_fixed_num = via->get_instance()->get_master_fixed_list().size();
    fwrite(&master_fixed_num, sizeof(uint32_t), 1, _file_ptr);

    for (IdbViaMasterFixed* master_fixed : via->get_instance()->get_master_fixed_list()) {
      uint8_t master_fixed_layer_name_size = master_fixed->get_layer()->get_name().size();
      fwrite(&master_fixed_layer_name_size, sizeof(uint8_t), 1, _file_ptr);

      uint32_t master_fixed_rect_num = master_fixed->get_layer_shape()->get_rect_list_num();
      fwrite(&master_fixed_rect_num, sizeof(uint32_t), 1, _file_ptr);
    }
  }
}

void IdbViaListHeader::save_data()
{
  for (IdbVia* via : _vias->get_via_list()) {
    string via_name = via->get_name();
    fwrite(via_name.c_str(), via_name.size(), 1, _file_ptr);

    string via_instance_name = via->get_instance()->get_name();
    fwrite(via_instance_name.c_str(), via_instance_name.size(), 1, _file_ptr);

    IdbRect* cut_rect = via->get_instance()->get_cut_rect();
    fwrite(cut_rect, sizeof(IdbRect), 1, _file_ptr);

    IdbViaMasterGenerate* master_generate = via->get_instance()->get_master_generate();

    string rule_name = master_generate->get_rule_name();
    fwrite(rule_name.c_str(), rule_name.size(), 1, _file_ptr);

    IdbViaRuleGenerate* rule_generate = master_generate->get_rule_generate();

    bool is_rule_generate_null = (rule_generate == nullptr) ? 1 : 0;
    fwrite(&is_rule_generate_null, sizeof(bool), 1, _file_ptr);

    if (!is_rule_generate_null) {
      string rule_generate_name = rule_generate->get_name();
      fwrite(rule_generate_name.c_str(), rule_generate_name.size(), 1, _file_ptr);

      string rule_generate_layer_bottom_name = rule_generate->get_layer_bottom()->get_name();
      fwrite(rule_generate_layer_bottom_name.c_str(), rule_generate_layer_bottom_name.size(), 1, _file_ptr);

      IdbLayerCutEnclosure* layer_cut_enclosure_bottom = rule_generate->get_enclosure_bottom();
      fwrite(layer_cut_enclosure_bottom, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);

      string rule_generate_layer_cut_name = rule_generate->get_layer_cut()->get_name();
      fwrite(rule_generate_layer_cut_name.c_str(), rule_generate_layer_cut_name.size(), 1, _file_ptr);

      IdbRect* cut_rect = rule_generate->get_cut_rect();
      fwrite(cut_rect, sizeof(IdbRect), 1, _file_ptr);

      int32_t cut_spacing_x = rule_generate->get_spacing_x();
      fwrite(&cut_spacing_x, sizeof(int32_t), 1, _file_ptr);

      int32_t cut_spacing_y = rule_generate->get_spacing_y();
      fwrite(&cut_spacing_y, sizeof(int32_t), 1, _file_ptr);

      string rule_generate_layer_top_name = rule_generate->get_layer_top()->get_name();
      fwrite(rule_generate_layer_top_name.c_str(), rule_generate_layer_top_name.size(), 1, _file_ptr);

      IdbLayerCutEnclosure* layer_cut_enclosure_top = rule_generate->get_enclosure_top();
      fwrite(layer_cut_enclosure_top, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);
    }

    int32_t cut_size_x = master_generate->get_cut_size_x();
    fwrite(&cut_size_x, sizeof(int32_t), 1, _file_ptr);

    int32_t cut_size_y = master_generate->get_cut_size_y();
    fwrite(&cut_size_y, sizeof(int32_t), 1, _file_ptr);

    bool is_layer_bottom_null = master_generate->get_layer_bottom() == nullptr ? 1 : 0;
    fwrite(&is_layer_bottom_null, sizeof(bool), 1, _file_ptr);
    if (!is_layer_bottom_null) {
      string layer_bottom_name = master_generate->get_layer_bottom()->get_name();
      fwrite(layer_bottom_name.c_str(), layer_bottom_name.size(), 1, _file_ptr);
    }

    bool is_layer_cut_null = master_generate->get_layer_cut() == nullptr ? 1 : 0;
    fwrite(&is_layer_cut_null, sizeof(bool), 1, _file_ptr);
    if (!is_layer_cut_null) {
      string layer_cut_name = master_generate->get_layer_cut()->get_name();
      fwrite(layer_cut_name.c_str(), layer_cut_name.size(), 1, _file_ptr);
    }

    bool is_layer_top_null = master_generate->get_layer_top() == nullptr ? 1 : 0;
    fwrite(&is_layer_top_null, sizeof(bool), 1, _file_ptr);
    if (!is_layer_top_null) {
      string layer_top_name = master_generate->get_layer_top()->get_name();
      fwrite(layer_top_name.c_str(), layer_top_name.size(), 1, _file_ptr);
    }

    int32_t cut_spacing_x = master_generate->get_cut_spcing_x();
    fwrite(&cut_spacing_x, sizeof(int32_t), 1, _file_ptr);
    int32_t cut_spacing_y = master_generate->get_cut_spcing_y();
    fwrite(&cut_spacing_y, sizeof(int32_t), 1, _file_ptr);

    int32_t enclosure_bottom_x = master_generate->get_enclosure_bottom_x();
    fwrite(&enclosure_bottom_x, sizeof(int32_t), 1, _file_ptr);
    int32_t enclosure_bottom_y = master_generate->get_enclosure_bottom_y();
    fwrite(&enclosure_bottom_y, sizeof(int32_t), 1, _file_ptr);
    int32_t enclosure_top_x = master_generate->get_enclosure_top_x();
    fwrite(&enclosure_top_x, sizeof(int32_t), 1, _file_ptr);
    int32_t enclosure_top_y = master_generate->get_enclosure_top_y();
    fwrite(&enclosure_top_y, sizeof(int32_t), 1, _file_ptr);

    int32_t num_cut_rows = master_generate->get_cut_rows();
    fwrite(&num_cut_rows, sizeof(int32_t), 1, _file_ptr);
    int32_t num_cut_cols = master_generate->get_cut_cols();
    fwrite(&num_cut_cols, sizeof(int32_t), 1, _file_ptr);

    int32_t original_offset_x = master_generate->get_original_offset_x();
    fwrite(&original_offset_x, sizeof(int32_t), 1, _file_ptr);
    int32_t original_offset_y = master_generate->get_original_offset_y();
    fwrite(&original_offset_y, sizeof(int32_t), 1, _file_ptr);

    int32_t offset_bottom_x = master_generate->get_offset_bottom_x();
    fwrite(&offset_bottom_x, sizeof(int32_t), 1, _file_ptr);
    int32_t offset_bottom_y = master_generate->get_offset_bottom_y();
    fwrite(&offset_bottom_y, sizeof(int32_t), 1, _file_ptr);
    int32_t offset_top_x = master_generate->get_offset_top_x();
    fwrite(&offset_top_x, sizeof(int32_t), 1, _file_ptr);
    int32_t offset_top_y = master_generate->get_offset_top_y();
    fwrite(&offset_top_y, sizeof(int32_t), 1, _file_ptr);

    for (IdbRect* rect : master_generate->get_cut_rect_list()) {
      fwrite(rect, sizeof(IdbRect), 1, _file_ptr);
    }

    IdbRect* cut_bounding_rect = master_generate->get_cut_bouding_rect();
    fwrite(cut_bounding_rect, sizeof(IdbRect), 1, _file_ptr);

    bool is_default = via->get_instance()->is_default();
    fwrite(&is_default, sizeof(bool), 1, _file_ptr);

    IdbViaMaster::IdbViaMasterType type = via->get_instance()->get_type();
    fwrite(&type, sizeof(IdbViaMaster::IdbViaMasterType), 1, _file_ptr);

    for (IdbViaMasterFixed* master_fixed : via->get_instance()->get_master_fixed_list()) {
      string layer_name = master_fixed->get_layer()->get_name();
      fwrite(layer_name.c_str(), layer_name.size(), 1, _file_ptr);

      for (IdbRect* rect : master_fixed->get_rect_list()) {
        fwrite(rect, sizeof(IdbRect), 1, _file_ptr);
      }
    }

    IdbCoordinate<int32_t>* coordinate = via->get_coordinate();
    fwrite(coordinate, sizeof(IdbCoordinate<int32_t>), 1, _file_ptr);
  }

  fclose(_file_ptr);
}

void IdbViaListHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");

  fread(&_vias_num, sizeof(uint32_t), 1, _file_ptr);

  for (uint32_t i = 0; i < _vias_num; i++) {
    uint8_t via_name_size;
    fread(&via_name_size, sizeof(uint8_t), 1, _file_ptr);
    _via_name_size.push_back(via_name_size);

    uint8_t via_instance_name_size;
    fread(&via_instance_name_size, sizeof(uint8_t), 1, _file_ptr);
    _via_instance_name_size.push_back(via_instance_name_size);

    uint8_t rule_name_size;
    fread(&rule_name_size, sizeof(uint8_t), 1, _file_ptr);
    _rule_name_size.push_back(rule_name_size);

    uint8_t rule_generate_name_size;
    fread(&rule_generate_name_size, sizeof(uint8_t), 1, _file_ptr);
    _rule_generate_name_size.push_back(rule_generate_name_size);

    uint8_t rule_generate_layer_bottom_size;
    fread(&rule_generate_layer_bottom_size, sizeof(uint8_t), 1, _file_ptr);
    _rule_generate_layer_bottom_name_size.push_back(rule_generate_layer_bottom_size);

    uint8_t rule_generate_layer_cut_size;
    fread(&rule_generate_layer_cut_size, sizeof(uint8_t), 1, _file_ptr);
    _rule_generate_layer_cut_name_size.push_back(rule_generate_layer_cut_size);

    uint8_t rule_generate_layer_top_size;
    fread(&rule_generate_layer_top_size, sizeof(uint8_t), 1, _file_ptr);
    _rule_generate_layer_top_name_size.push_back(rule_generate_layer_top_size);

    uint8_t master_generate_layer_bottom_name_size;
    fread(&master_generate_layer_bottom_name_size, sizeof(uint8_t), 1, _file_ptr);
    _master_generate_layer_bottom_name_size.push_back(master_generate_layer_bottom_name_size);

    uint8_t master_generate_layer_cut_name_size;
    fread(&master_generate_layer_cut_name_size, sizeof(uint8_t), 1, _file_ptr);
    _master_generate_layer_cut_name_size.push_back(master_generate_layer_cut_name_size);

    uint8_t master_generate_layer_top_name_size;
    fread(&master_generate_layer_top_name_size, sizeof(uint8_t), 1, _file_ptr);
    _master_generate_layer_top_name_size.push_back(master_generate_layer_top_name_size);

    uint32_t cut_rect_num;
    fread(&cut_rect_num, sizeof(uint32_t), 1, _file_ptr);
    _cut_rect_num_list.push_back(cut_rect_num);

    uint32_t master_fixed_num;
    fread(&master_fixed_num, sizeof(uint32_t), 1, _file_ptr);
    _via_instance_master_fixed_num_list.push_back(master_fixed_num);

    for (uint32_t j = 0; j < master_fixed_num; j++) {
      uint8_t master_fixed_layer_name_size;
      fread(&master_fixed_layer_name_size, sizeof(uint8_t), 1, _file_ptr);
      _master_fixed_layer_name_size.push_back(master_fixed_layer_name_size);

      uint32_t master_fixed_rect_num;
      fread(&master_fixed_rect_num, sizeof(uint32_t), 1, _file_ptr);
      _master_fixed_rect_num_list.push_back(master_fixed_rect_num);
    }
  }
}

void IdbViaListHeader::load_data()
{
  uint32_t master_fixed_layer_name_size_index, master_fixed_rect_num_index;
  master_fixed_layer_name_size_index = master_fixed_rect_num_index = 0;

  for (uint32_t i = 0; i < _vias_num; i++) {
    IdbVia* via = new IdbVia();

    char via_name[_via_name_size[i]];
    fread(via_name, _via_name_size[i], 1, _file_ptr);
    string via_name_str = via_name;
    via_name_str.resize(_via_name_size[i]);
    via->set_name(via_name_str);

    IdbViaMaster* via_master = new IdbViaMaster();

    char via_instance_name[_via_instance_name_size[i]];
    fread(via_instance_name, _via_instance_name_size[i], 1, _file_ptr);
    string via_instance_name_str = via_instance_name;
    via_instance_name_str.resize(_via_instance_name_size[i]);
    via_master->set_name(via_instance_name_str);

    IdbRect* cut_rect = new IdbRect();
    fread(cut_rect, sizeof(IdbRect), 1, _file_ptr);
    via_master->set_cut_rect(cut_rect->get_low_x(), cut_rect->get_low_y(), cut_rect->get_high_x(), cut_rect->get_high_y());

    IdbViaMasterGenerate* master_generate = new IdbViaMasterGenerate();

    char rule_name[_rule_name_size[i]];
    fread(rule_name, _rule_name_size[i], 1, _file_ptr);
    string rule_name_str = rule_name;
    rule_name_str.resize(_rule_name_size[i]);
    master_generate->set_rule_name(rule_name_str);

    bool is_rule_generate_null;
    fread(&is_rule_generate_null, sizeof(bool), 1, _file_ptr);

    if (is_rule_generate_null) {
      master_generate->set_rule_generate(nullptr);
    } else {
      IdbViaRuleGenerate* via_rule_generate = new IdbViaRuleGenerate();

      char rule_generate_name[_rule_generate_name_size[i]];
      fread(rule_generate_name, _rule_generate_name_size[i], 1, _file_ptr);
      string rule_generate_name_str = rule_generate_name;
      rule_generate_name_str.resize(_rule_generate_name_size[i]);
      via_rule_generate->set_name(rule_generate_name_str);

      char layer_bottom_name[_rule_generate_layer_bottom_name_size[i]];
      fread(layer_bottom_name, _rule_generate_layer_bottom_name_size[i], 1, _file_ptr);
      string layer_bottom_name_str = layer_bottom_name;
      layer_bottom_name_str.resize(_rule_generate_layer_bottom_name_size[i]);
      IdbLayerRouting* layer_bottom = dynamic_cast<IdbLayerRouting*>(_layers->find_layer(layer_bottom_name_str));
      via_rule_generate->set_layer_bottom(layer_bottom);

      IdbLayerCutEnclosure* layer_cut_enclosure_bottom = new IdbLayerCutEnclosure();
      fread(layer_cut_enclosure_bottom, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);
      via_rule_generate->set_enclosure_bottom(layer_cut_enclosure_bottom);

      char layer_cut_name[_rule_generate_layer_cut_name_size[i]];
      fread(layer_cut_name, _rule_generate_layer_cut_name_size[i], 1, _file_ptr);
      string layer_cut_name_str = layer_cut_name;
      layer_cut_name_str.resize(_rule_generate_layer_cut_name_size[i]);
      IdbLayerCut* layer_cut = dynamic_cast<IdbLayerCut*>(_layers->find_layer(layer_cut_name_str));
      via_rule_generate->set_layer_cut(layer_cut);

      IdbRect* cut_rect = new IdbRect();
      fread(cut_rect, sizeof(IdbRect), 1, _file_ptr);
      via_rule_generate->set_cut_rect(cut_rect);

      int32_t cut_spacing_x;
      fread(&cut_spacing_x, sizeof(int32_t), 1, _file_ptr);
      int32_t cut_spacing_y;
      fread(&cut_spacing_y, sizeof(int32_t), 1, _file_ptr);
      via_rule_generate->set_spacing(cut_spacing_x, cut_spacing_y);

      char layer_top_name[_rule_generate_layer_top_name_size[i]];
      fread(layer_top_name, _rule_generate_layer_top_name_size[i], 1, _file_ptr);
      string layer_top_name_str = layer_top_name;
      layer_top_name_str.resize(_rule_generate_layer_top_name_size[i]);
      IdbLayerRouting* layer_top = dynamic_cast<IdbLayerRouting*>(_layers->find_layer(layer_top_name_str));
      via_rule_generate->set_layer_top(layer_top);

      IdbLayerCutEnclosure* layer_cut_enclosure_top = new IdbLayerCutEnclosure();
      fread(layer_cut_enclosure_top, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);
      via_rule_generate->set_enclosure_top(layer_cut_enclosure_top);
    }

    int32_t cut_size_x;
    fread(&cut_size_x, sizeof(int32_t), 1, _file_ptr);
    int32_t cut_size_y;
    fread(&cut_size_y, sizeof(int32_t), 1, _file_ptr);
    master_generate->set_cut_size(cut_size_x, cut_size_y);

    bool is_layer_bottom_null;
    fread(&is_layer_bottom_null, sizeof(bool), 1, _file_ptr);
    if (is_layer_bottom_null) {
      master_generate->set_layer_bottom(nullptr);
    } else {
      char layer_bottom_name[_master_generate_layer_bottom_name_size[i]];
      fread(layer_bottom_name, _master_generate_layer_bottom_name_size[i], 1, _file_ptr);
      string layer_bottom_name_str = layer_bottom_name;
      layer_bottom_name_str.resize(_master_generate_layer_bottom_name_size[i]);
      IdbLayerRouting* layer_bottom = dynamic_cast<IdbLayerRouting*>(_layers->find_layer(layer_bottom_name_str));
      master_generate->set_layer_bottom(layer_bottom);
    }

    bool is_layer_cut_null;
    fread(&is_layer_cut_null, sizeof(bool), 1, _file_ptr);
    if (is_layer_cut_null) {
      master_generate->set_layer_cut(nullptr);
    } else {
      char layer_cut_name[_master_generate_layer_cut_name_size[i]];
      fread(layer_cut_name, _master_generate_layer_cut_name_size[i], 1, _file_ptr);
      string layer_cut_name_str = layer_cut_name;
      layer_cut_name_str.resize(_master_generate_layer_cut_name_size[i]);
      IdbLayerCut* layer_cut = dynamic_cast<IdbLayerCut*>(_layers->find_layer(layer_cut_name_str));
      master_generate->set_layer_cut(layer_cut);
    }

    bool is_layer_top_null;
    fread(&is_layer_top_null, sizeof(bool), 1, _file_ptr);
    if (is_layer_top_null) {
      master_generate->set_layer_top(nullptr);
    } else {
      char layer_top_name[_master_generate_layer_top_name_size[i]];
      fread(layer_top_name, _master_generate_layer_top_name_size[i], 1, _file_ptr);
      string layer_top_name_str = layer_top_name;
      layer_top_name_str.resize(_master_generate_layer_top_name_size[i]);
      IdbLayerRouting* layer_top = dynamic_cast<IdbLayerRouting*>(_layers->find_layer(layer_top_name_str));
      master_generate->set_layer_top(layer_top);
    }

    int32_t cut_spacing_x;
    fread(&cut_spacing_x, sizeof(int32_t), 1, _file_ptr);
    int32_t cut_spacing_y;
    fread(&cut_spacing_y, sizeof(int32_t), 1, _file_ptr);
    master_generate->set_cut_spacing(cut_spacing_x, cut_spacing_y);

    int32_t enclosure_bottom_x;
    fread(&enclosure_bottom_x, sizeof(int32_t), 1, _file_ptr);
    int32_t enclosure_bottom_y;
    fread(&enclosure_bottom_y, sizeof(int32_t), 1, _file_ptr);
    int32_t enclosure_top_x;
    fread(&enclosure_top_x, sizeof(int32_t), 1, _file_ptr);
    int32_t enclosure_top_y;
    fread(&enclosure_top_y, sizeof(int32_t), 1, _file_ptr);
    master_generate->set_enclosure_bottom(enclosure_bottom_x, enclosure_bottom_y);
    master_generate->set_enclosure_top(enclosure_top_x, enclosure_top_y);

    int32_t num_cut_rows;
    fread(&num_cut_rows, sizeof(int32_t), 1, _file_ptr);
    int32_t num_cut_cols;
    fread(&num_cut_cols, sizeof(int32_t), 1, _file_ptr);
    master_generate->set_cut_row_col(num_cut_rows, num_cut_cols);

    int32_t original_offset_x;
    fread(&original_offset_x, sizeof(int32_t), 1, _file_ptr);
    int32_t original_offset_y;
    fread(&original_offset_y, sizeof(int32_t), 1, _file_ptr);
    master_generate->set_original(original_offset_x, original_offset_y);

    int32_t offset_bottom_x;
    fread(&offset_bottom_x, sizeof(int32_t), 1, _file_ptr);
    int32_t offset_bottom_y;
    fread(&offset_bottom_y, sizeof(int32_t), 1, _file_ptr);
    int32_t offset_top_x;
    fread(&offset_top_x, sizeof(int32_t), 1, _file_ptr);
    int32_t offset_top_y;
    fread(&offset_top_y, sizeof(int32_t), 1, _file_ptr);
    master_generate->set_offset_bottom(offset_bottom_x, offset_bottom_y);
    master_generate->set_offset_top(offset_top_x, offset_top_y);

    for (uint32_t j = 0; j < _cut_rect_num_list[i]; j++) {
      IdbRect* rect = new IdbRect();
      fread(rect, sizeof(IdbRect), 1, _file_ptr);
      master_generate->add_cut_rect(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
    }

    IdbRect* bounding_rect = new IdbRect();
    fread(bounding_rect, sizeof(IdbRect), 1, _file_ptr);

    master_generate->set_cut_bouding_rect(bounding_rect->get_low_x(), bounding_rect->get_low_y(), bounding_rect->get_high_x(),
                                          bounding_rect->get_high_y());

    via_master->set_master_generate(master_generate);

    bool is_default;
    fread(&is_default, sizeof(bool), 1, _file_ptr);
    via_master->set_default(is_default);

    IdbViaMaster::IdbViaMasterType type;
    fread(&type, sizeof(IdbViaMaster::IdbViaMasterType), 1, _file_ptr);
    via_master->set_type(type);

    for (uint32_t j = 0; j < _via_instance_master_fixed_num_list[i]; j++) {
      char layer_name[_master_fixed_layer_name_size[master_fixed_layer_name_size_index]];
      fread(layer_name, _master_fixed_layer_name_size[master_fixed_layer_name_size_index], 1, _file_ptr);
      string layer_name_str = layer_name;
      layer_name_str.resize(_master_fixed_layer_name_size[master_fixed_layer_name_size_index]);
      ++master_fixed_layer_name_size_index;

      IdbViaMasterFixed* master_fixed = via_master->add_fixed(layer_name_str);
      master_fixed->set_layer(_layers->find_layer(layer_name_str));

      for (uint32_t k = 0; k < _master_fixed_rect_num_list[master_fixed_rect_num_index]; k++) {
        IdbRect* rect = master_fixed->add_rect();
        fread(rect, sizeof(IdbRect), 1, _file_ptr);
      }
      ++master_fixed_rect_num_index;

      via_master->set_bottom_layer_shape();
      via_master->set_cut_layer_shape();
      via_master->set_top_layer_shape();
    }

    via->set_instance(via_master);

    IdbCoordinate<int32_t>* coordinate = new IdbCoordinate<int32_t>();
    fread(coordinate, sizeof(IdbCoordinate<int32_t>), 1, _file_ptr);
    via->set_coordinate(coordinate);

    _vias->add_via(via);
  }

  fclose(_file_ptr);
}  // TODO::set_bounding_box

// IdbViaRuleList
IdbViaRuleListHeader::IdbViaRuleListHeader(IdbFileHeaderType type, const char* file_path, IdbViaRuleList* via_rule, IdbLayers* layers)
    : IdbHeader()
{
  this->set_type(type);
  this->set_file_path(file_path);
  this->_via_rule = via_rule;
  this->_layers = layers;
}

void IdbViaRuleListHeader::save_header()
{
  _file_ptr = fopen(this->get_file_path(), "wb");

  _via_rule_generate_num = _via_rule->get_num_via_rule_generate();
  fwrite(&_via_rule_generate_num, sizeof(int32_t), 1, _file_ptr);

  for (IdbViaRuleGenerate* via_rule : _via_rule->get_rule_list()) {
    uint8_t via_rule_name_size = via_rule->get_name().size();
    fwrite(&via_rule_name_size, sizeof(uint8_t), 1, _file_ptr);

    uint8_t layer_bottom_name_size = via_rule->get_layer_bottom()->get_name().size();
    fwrite(&layer_bottom_name_size, sizeof(uint8_t), 1, _file_ptr);

    uint8_t layer_cut_name_size = via_rule->get_layer_cut()->get_name().size();
    fwrite(&layer_cut_name_size, sizeof(uint8_t), 1, _file_ptr);

    uint8_t layer_top_name_size = via_rule->get_layer_top()->get_name().size();
    fwrite(&layer_top_name_size, sizeof(uint8_t), 1, _file_ptr);
  }
}

void IdbViaRuleListHeader::save_data()
{
  for (IdbViaRuleGenerate* via_rule : _via_rule->get_rule_list()) {
    string via_rule_name = via_rule->get_name();
    fwrite(via_rule_name.c_str(), via_rule_name.size(), 1, _file_ptr);

    string layer_bottom_name = via_rule->get_layer_bottom()->get_name();
    fwrite(layer_bottom_name.c_str(), layer_bottom_name.size(), 1, _file_ptr);

    IdbLayerCutEnclosure* enclosure_bottom = via_rule->get_enclosure_bottom();
    fwrite(enclosure_bottom, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);

    string layer_cut_name = via_rule->get_layer_cut()->get_name();
    fwrite(layer_cut_name.c_str(), layer_cut_name.size(), 1, _file_ptr);

    IdbRect* cut_rect = via_rule->get_cut_rect();
    fwrite(cut_rect, sizeof(IdbRect), 1, _file_ptr);

    int32_t spacing_x = via_rule->get_spacing_x();
    fwrite(&spacing_x, sizeof(int32_t), 1, _file_ptr);

    int32_t spacing_y = via_rule->get_spacing_y();
    fwrite(&spacing_y, sizeof(int32_t), 1, _file_ptr);

    string layer_top_name = via_rule->get_layer_top()->get_name();
    fwrite(layer_top_name.c_str(), layer_top_name.size(), 1, _file_ptr);

    IdbLayerCutEnclosure* enclosure_top = via_rule->get_enclosure_top();
    fwrite(enclosure_top, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);
  }

  fclose(_file_ptr);
}

void IdbViaRuleListHeader::load_header()
{
  _file_ptr = fopen(this->get_file_path(), "rb");

  fread(&_via_rule_generate_num, sizeof(int32_t), 1, _file_ptr);

  for (uint32_t i = 0; i < _via_rule_generate_num; i++) {
    uint8_t via_rule_name_size;
    fread(&via_rule_name_size, sizeof(uint8_t), 1, _file_ptr);
    _via_rule_name_size.push_back(via_rule_name_size);

    uint8_t layer_bottom_name_size;
    fread(&layer_bottom_name_size, sizeof(uint8_t), 1, _file_ptr);
    _layer_bottom_name_size.push_back(layer_bottom_name_size);

    uint8_t layer_cut_name_size;
    fread(&layer_cut_name_size, sizeof(uint8_t), 1, _file_ptr);
    _layer_cut_name_size.push_back(layer_cut_name_size);

    uint8_t layer_top_name_size;
    fread(&layer_top_name_size, sizeof(uint8_t), 1, _file_ptr);
    _layer_top_name_size.push_back(layer_top_name_size);
  }
}

void IdbViaRuleListHeader::load_data()
{
  _via_rule->set_num_via_rule(_via_rule_generate_num);

  for (uint32_t i = 0; i < _via_rule_generate_num; i++) {
    IdbViaRuleGenerate* via_rule = new IdbViaRuleGenerate();

    char via_rule_name[_via_rule_name_size[i]];
    fread(via_rule_name, _via_rule_name_size[i], 1, _file_ptr);
    string via_rule_name_str = via_rule_name;
    via_rule_name_str.resize(_via_rule_name_size[i]);
    via_rule->set_name(via_rule_name_str);

    char layer_bottom_name[_layer_bottom_name_size[i]];
    fread(layer_bottom_name, _layer_bottom_name_size[i], 1, _file_ptr);
    string layer_bottom_name_str = layer_bottom_name;
    layer_bottom_name_str.resize(_layer_bottom_name_size[i]);
    IdbLayerRouting* layer_bottom = dynamic_cast<IdbLayerRouting*>(_layers->find_layer(layer_bottom_name_str));
    via_rule->set_layer_bottom(layer_bottom);

    IdbLayerCutEnclosure* enclosuer_bottom = new IdbLayerCutEnclosure();
    fread(enclosuer_bottom, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);
    via_rule->set_enclosure_bottom(enclosuer_bottom);

    char layer_cut_name[_layer_cut_name_size[i]];
    fread(layer_cut_name, _layer_cut_name_size[i], 1, _file_ptr);
    string layer_cut_name_str = layer_cut_name;
    layer_cut_name_str.resize(_layer_cut_name_size[i]);
    IdbLayerCut* layer_cut = dynamic_cast<IdbLayerCut*>(_layers->find_layer(layer_cut_name_str));
    via_rule->set_layer_cut(layer_cut);

    IdbRect* cut_rect = new IdbRect();
    fread(cut_rect, sizeof(IdbRect), 1, _file_ptr);
    via_rule->set_cut_rect(cut_rect);

    int32_t cut_spacing_x, cut_spacing_y;
    fread(&cut_spacing_x, sizeof(int32_t), 1, _file_ptr);
    fread(&cut_spacing_y, sizeof(int32_t), 1, _file_ptr);
    via_rule->set_spacing(cut_spacing_x, cut_spacing_y);

    char layer_top_name[_layer_top_name_size[i]];
    fread(layer_top_name, _layer_top_name_size[i], 1, _file_ptr);
    string layer_top_name_str = layer_top_name;
    layer_top_name_str.resize(_layer_top_name_size[i]);
    IdbLayerRouting* layer_top = dynamic_cast<IdbLayerRouting*>(_layers->find_layer(layer_top_name_str));
    via_rule->set_layer_top(layer_top);

    IdbLayerCutEnclosure* enclosuer_top = new IdbLayerCutEnclosure();
    fread(enclosuer_top, sizeof(IdbLayerCutEnclosure), 1, _file_ptr);
    via_rule->set_enclosure_top(enclosuer_top);

    _via_rule->get_rule_list().push_back(via_rule);
  }

  fclose(_file_ptr);
}

}  // namespace idb
