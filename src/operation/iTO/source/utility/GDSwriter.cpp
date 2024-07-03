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
#include "GDSwriter.h"

#include "IdbBlockages.h"
#include "IdbDesign.h"
#include "IdbInstance.h"
#include "ToApi.hpp"
#include "builder.h"

namespace ito {
void GDSwriter::writeGDS(idb::IdbBuilder* idb_builder, std::string path)
{
  idb::IdbDesign* idb_design = idb_builder->get_def_service()->get_design();

  // IdbLayout *idb_layout = idb_builder->get_lef_service()->get_layout();

  std::ofstream gds_file(path);
  if (gds_file.is_open()) {
    gds_file << "HEADER 600" << std::endl;
    gds_file << "BGNLIB" << std::endl;
    gds_file << "LIBNAME TOLib" << std::endl;
    gds_file << "UNITS 0.001 1e-9" << std::endl;
    writeAllInstance(gds_file, idb_design);
    gds_file << "ENDLIB" << endl;

    gds_file.close();
  }
}

void GDSwriter::writeAllInstance(ofstream& gds_file, idb::IdbDesign* idb_design)
{
  cout << "Write instance to GDS starting" << endl;
  if (gds_file.is_open()) {
    int count = 0;
    for (auto inst : idb_design->get_instance_list()->get_instance_list()) {
      writeInstance(gds_file, inst, 0);
      count++;
    }

    int block_named = 1;
    for (auto block : idb_design->get_blockage_list()->get_blockage_list()) {
      if (block->is_palcement_blockage()) {
        writeBlockage(gds_file, block, 3, block_named);
        block_named++;
      }
    }

    // top
    gds_file << "BGNSTR" << std::endl;
    gds_file << "STRNAME top" << std::endl;

    for (idb::IdbInstance* instance : idb_design->get_instance_list()->get_instance_list()) {
      gds_file << "SREF" << std::endl;
      gds_file << "SNAME " << instance->get_name() << std::endl;
      gds_file << "XY 0:0" << std::endl;
      gds_file << "ENDEL" << std::endl;
    }

    block_named = 1;
    for (auto block : idb_design->get_blockage_list()->get_blockage_list()) {
      if (block->is_palcement_blockage()) {
        gds_file << "SREF" << std::endl;
        gds_file << "SNAME "
                 << "blockage_" << block_named << std::endl;
        gds_file << "XY 0:0" << std::endl;
        gds_file << "ENDEL" << std::endl;
        block_named++;
      }
    }

    gds_file << "ENDSTR" << endl;
    // gds_file << "ENDLIB" << endl;
    // gds_file.close();
    cout << "Write success! Total of " << count << " instances." << endl;
  }
}

void GDSwriter::writeInstance(ofstream& gds_file, idb::IdbInstance* instance, int layer)
{
  int factor = 1;
  int llx = instance->get_bounding_box()->get_low_x() * factor;
  int lly = instance->get_bounding_box()->get_low_y() * factor;
  int urx = instance->get_bounding_box()->get_high_x() * factor;
  int ury = instance->get_bounding_box()->get_high_y() * factor;

  auto inst_name = instance->get_name().c_str();
  if (strstr(inst_name, "DRV_buffer_") != NULL) {
    layer = 1;
  }
  if (strstr(inst_name, "hold_buf_") != NULL) {
    layer = 2;
  }
  // instance->get_id();
  gds_file << "BGNSTR" << std::endl;
  gds_file << "STRNAME " << instance->get_name() << std::endl;
  gds_file << "BOUNDARY" << std::endl;
  gds_file << "LAYER " << layer << std::endl;
  gds_file << "DATATYPE 0" << std::endl;
  gds_file << "XY" << std::endl;
  gds_file << llx << " : " << lly << std::endl;
  gds_file << urx << " : " << lly << std::endl;
  gds_file << urx << " : " << ury << std::endl;
  gds_file << llx << " : " << ury << std::endl;
  gds_file << llx << " : " << lly << std::endl;
  gds_file << "ENDEL" << std::endl;
  gds_file << "ENDSTR" << std::endl;
}

void GDSwriter::writeBlockage(ofstream& gds_file, idb::IdbBlockage* block, int layer, int named)
{
  int factor = 1;
  idb::IdbRect* rect = block->get_rect_list()[0];

  int llx = rect->get_low_x() * factor;
  int lly = rect->get_low_y() * factor;
  int urx = rect->get_high_x() * factor;
  int ury = rect->get_high_y() * factor;

  gds_file << "BGNSTR" << std::endl;
  gds_file << "STRNAME "
           << "blockage_" << named << std::endl;
  gds_file << "BOUNDARY" << std::endl;
  gds_file << "LAYER " << layer << std::endl;
  gds_file << "DATATYPE 0" << std::endl;
  gds_file << "XY" << std::endl;
  gds_file << llx << " : " << lly << std::endl;
  gds_file << urx << " : " << lly << std::endl;
  gds_file << urx << " : " << ury << std::endl;
  gds_file << llx << " : " << ury << std::endl;
  gds_file << llx << " : " << lly << std::endl;
  gds_file << "ENDEL" << std::endl;
  gds_file << "ENDSTR" << std::endl;
}
}  // namespace ito
