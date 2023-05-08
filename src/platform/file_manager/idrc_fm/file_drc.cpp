/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @project		iplf
 * @file		file_cts.h
 * @copyright	(c) 2021 All Rights Reserved.
 * @date		25/05/2021
 * @version		0.1
 * @description


        Process file.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "file_drc.h"

#include <cstring>
#include <iostream>

#include "idrc_io/idrc_io.h"

using namespace std;

namespace iplf {

void FileDrcManager::wrapDrcStruct(idrc::DrcViolationSpot* spot, DrcDetailResult& detail_result)
{
  memset(&detail_result, 0, sizeof(DrcDetailResult));
  detail_result.violation_type = (int) spot->get_violation_type();
  std::memcpy(detail_result.layer_name, spot->get_layer().c_str(), spot->get_layer().length());
  detail_result.layer_id = spot->get_layer_id();
  detail_result.net_id = spot->get_net_id();
  detail_result.min_x = spot->get_min_x();
  detail_result.min_y = spot->get_min_y();
  detail_result.max_x = spot->get_max_x();
  detail_result.max_y = spot->get_max_y();
}

void FileDrcManager::parseDrcStruct(DrcDetailResult& detail_result, idrc::DrcViolationSpot* spot)
{
  spot->set_vio_type((idrc::ViolationType) detail_result.violation_type);
  spot->set_layer_name(detail_result.layer_name);
  spot->set_layer_id(detail_result.layer_id);
  spot->set_net_id(detail_result.net_id);
  spot->setCoordinate(detail_result.min_x, detail_result.min_y, detail_result.max_x, detail_result.max_y);
}

bool FileDrcManager::parseFileData()
{
  //   uint64_t size = get_data_size();
  //   if (size == 0) {
  //     return false;
  //   }

  /// parse cts data header
  DrcFileHeader data_header;
  get_fstream().read((char*) &data_header, sizeof(DrcFileHeader));
  get_fstream().seekp(sizeof(DrcFileHeader), ios::cur);

  auto& detail_rule_map = drcInst->get_detail_drc();
  detail_rule_map.clear();

  char* data_buf = new char[max_size];
  std::memset(data_buf, 0, max_size);

  for (int i = 0; i < data_header.module_num; ++i) {
    /// parse header
    DrcResultHeader drc_header;
    get_fstream().read((char*) &drc_header, sizeof(DrcResultHeader));
    get_fstream().seekp(sizeof(DrcResultHeader), ios::cur);

    vector<idrc::DrcViolationSpot*> spot_list;
    spot_list.reserve(drc_header.drc_num);

    std::memset(data_buf, 0, max_size);
    char* buf_ref = data_buf;
    uint64_t total_num = 0;

    while (total_num < drc_header.drc_num) {
      /// calculate spot number read from file
      int read_num = drc_header.drc_num - total_num >= max_num ? max_num : drc_header.drc_num - total_num;

      get_fstream().read(data_buf, sizeof(DrcDetailResult) * read_num);
      get_fstream().seekp(sizeof(DrcDetailResult) * read_num, ios::cur);

      for (int j = 0; j < read_num; j++) {
        idrc::DrcViolationSpot* spot = new idrc::DrcViolationSpot();
        /// parse single unit
        DrcDetailResult detail_result;
        std::memcpy(&detail_result, buf_ref, sizeof(DrcDetailResult));
        parseDrcStruct(detail_result, spot);

        /// add to spot list
        spot_list.push_back(spot);
        buf_ref += sizeof(DrcDetailResult);

        total_num++;
      }

      std::memset(data_buf, 0, max_size);
      buf_ref = data_buf;
    }

    detail_rule_map.insert(std::make_pair(std::string(drc_header.rule_name), spot_list));
  }

  delete[] data_buf;
  data_buf = nullptr;

  return true;
}

int32_t FileDrcManager::getBufferSize()
{
  return drcInst->get_buffer_size();
}

bool FileDrcManager::saveFileData()
{
  //   int size = getBufferSize();
  //   assert(size != 0);
  auto& detail_rule_map = drcInst->get_detail_drc();
  /// save cts data header
  DrcFileHeader file_header;
  file_header.module_num = detail_rule_map.size();

  get_fstream().write((char*) &file_header, sizeof(DrcFileHeader));
  get_fstream().seekp(sizeof(DrcFileHeader), ios::cur);

  char* data_buf = new char[max_size];
  std::memset(data_buf, 0, max_size);

  for (auto [rule_name, drc_list] : detail_rule_map) {
    /// save drc header
    DrcResultHeader drc_header;
    std::memset(&drc_header, 0, sizeof(DrcResultHeader));
    std::memcpy(drc_header.rule_name, rule_name.c_str(), rule_name.length());
    drc_header.drc_num = drc_list.size();
    // std::memcpy(buf_ref, &drc_header, sizeof(DrcResultHeader));
    // buf_ref += sizeof(DrcResultHeader);
    // mem_size += sizeof(DrcResultHeader);
    get_fstream().write((char*) &drc_header, sizeof(DrcResultHeader));
    get_fstream().seekp(sizeof(DrcResultHeader), ios::cur);

    char* buf_ref = data_buf;
    int index = 0;
    uint64_t total_num = 0;
    for (auto drc_spot : drc_list) {
      /// wrap drc file struct
      DrcDetailResult detail_result;
      wrapDrcStruct(drc_spot, detail_result);
      /// save drc data
      std::memcpy(buf_ref, &detail_result, sizeof(DrcDetailResult));
      buf_ref += sizeof(DrcDetailResult);
      index++;
      total_num++;

      if (index == max_num || total_num >= drc_list.size()) {
        /// write file
        get_fstream().write(data_buf, sizeof(DrcDetailResult) * index);
        get_fstream().seekp(sizeof(DrcDetailResult) * index, ios::cur);

        /// reset
        std::memset(data_buf, 0, max_size);
        buf_ref = data_buf;
        index = 0;
      }
    }

    buf_ref = nullptr;
  }

  delete[] data_buf;
  data_buf = nullptr;

  return true;
}

}  // namespace iplf
