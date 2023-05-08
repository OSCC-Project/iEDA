#include "tapcell.h"

#include "idm.h"

using namespace std;

namespace ifp {

int32_t TapCellPlacer::find_macro_top_row(idb::IdbInstance* ins)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_rows = idb_layout->get_rows();

  int32_t height = ins->get_cell_master()->get_height();
  int32_t width = ins->get_cell_master()->get_width();
  int32_t lly = ins->get_coordinate()->get_y();
  int32_t ury = 0;
  idb::IdbOrient orient = ins->get_orient();
  if (orient == idb::IdbOrient::kW_R90 || orient == idb::IdbOrient::kE_R270 || orient == idb::IdbOrient::kFW_MX90
      || orient == idb::IdbOrient::kFE_MY90) {
    ury = lly + width;
  } else {
    ury = lly + height;
  }

  int i = 0;
  for (auto row : idb_rows->get_row_list()) {
    row->set_bounding_box();
    if (row->get_original_coordinate()->get_y() + row->get_site()->get_height() >= ury) {
      return i;
    }
    i++;
  }

  return -1;
}

/**
 * @brief 寻找macro的上边缘属于的row的序号
 *
 * @param ins
 * @return int32_t
 */
int32_t TapCellPlacer::find_macro_bottom_row(idb::IdbInstance* ins)
{
  auto idb_rows = dmInst->get_idb_design()->get_layout()->get_rows();
  int32_t lly = ins->get_coordinate()->get_y();
  int i = 0;
  for (auto row : idb_rows->get_row_list()) {
    row->set_bounding_box();
    if (row->get_original_coordinate()->get_y() + row->get_site()->get_height() > lly) {
      return i;
    }
    i++;
  }
  return -1;
}

/**
 * @brief 找到被占用的site的位置
 *
 * @param row_fills
 * @param lly
 * @return int32_t
 */
int32_t TapCellPlacer::find_index_in_row_fills(std::vector<std::pair<int32_t, std::vector<std::pair<int32_t, int32_t>>>> row_fills,
                                               int32_t lly)
{
  for (size_t i = 0; i < row_fills.size(); ++i) {
    if (row_fills[i].first == lly) {
      return i;
    }
  }
  return -1;
}

// /**
//  * @brief 找到macro占用的site
//  *
//  * @param ins
//  * @return int32_t
//  */
// int32_t TapCellPlacer::find_macro_overlap_site(idb::IdbInstance* ins)
// {
//   auto idb_design = dmInst->get_idb_design();
//   auto idb_layout = idb_design->get_layout();
//   auto idb_rows = idb_layout->get_rows();

//   auto idb_core = idb_layout->get_core();

//   int32_t core_llx = idb_core->get_bounding_box()->get_low_x();
//   int32_t core_urx = idb_core->get_bounding_box()->get_high_x();

//   ins->set_bounding_box();
//   int32_t llx = ins->get_coordinate()->get_x();
//   int32_t lly = ins->get_coordinate()->get_y();
//   int32_t urx = ins->get_bounding_box()->get_high_x();
//   int32_t ury = ins->get_bounding_box()->get_high_y();

//   int32_t bottom_row = find_macro_bottom_row(ins);
//   int32_t top_row = find_macro_top_row(ins);

//   for (int i = bottom_row; i <= top_row; ++i) {
//     IdbSite* site = idb_rows->get_row_list()[i]->get_site();
//     int32_t site_width = site->get_width();
//     int32_t row_y = idb_rows->get_row_list()[i]->get_bounding_box()->get_low_y();
//     int32_t cur_row_site_llx = 0, num = 0, cur_row_site_urx;
//     num = (llx - core_llx) / site_width;
//     cur_row_site_llx = num * site_width + core_llx;
//     num = (urx - core_llx) / site_width + 1;
//     cur_row_site_urx = num * site_width + core_llx;
//     _filled_sites.push_back(make_pair(row_y, make_pair(cur_row_site_llx, cur_row_site_urx)));
//   }
// }

/**
 * @brief 查找矩形上边缘所处的row的序号
 *
 * @param rect
 * @return int32_t
 */
int32_t TapCellPlacer::find_rect_top_row(idb::IdbRect* rect)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_rows = idb_layout->get_rows();
  auto idb_core = idb_layout->get_core();

  int32_t core_ury = idb_core->get_bounding_box()->get_high_y();
  int32_t ury = rect->get_high_y();

  if (core_ury < ury) {
    return idb_rows->get_row_list().size() - 1;
  }

  for (size_t i = 0; i < idb_rows->get_row_list().size(); ++i) {
    idb::IdbRow* row = idb_rows->get_row_list()[i];
    row->set_bounding_box();
    if (row->get_original_coordinate()->get_y() + row->get_site()->get_height() >= ury) {
      return i;
    }
  }
  //   std::cout << "No Overlap Row" << std::endl;
  return -1;
}

/**
 * @brief 查找矩形下边缘所在的row的序号
 *
 * @param rect
 * @return int32_t
 */
int32_t TapCellPlacer::find_rect_bottom_row(idb::IdbRect* rect)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_rows = idb_layout->get_rows();
  auto idb_core = idb_layout->get_core();

  int32_t core_lly = idb_core->get_bounding_box()->get_low_y();
  int32_t lly = rect->get_low_y();

  if (core_lly > lly) {
    return 0;
  }

  for (size_t i = 0; i < idb_rows->get_row_list().size(); ++i) {
    idb::IdbRow* row = idb_rows->get_row_list()[i];
    row->set_bounding_box();
    if (row->get_original_coordinate()->get_y() + row->get_site()->get_height() > lly) {
      return i;
    }
  }
  //   std::cout << "No Overlap Row" << std::endl;
  return -1;
}

/**
 * @brief 查找布局障碍占据的site
 *
 * @param blk
 * @return int32_t
 */
int32_t TapCellPlacer::find_placement_blockage_overlap_site(IdbBlockage* blk)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_rows = idb_layout->get_rows();
  auto idb_core = idb_layout->get_core();

  int32_t core_llx = idb_core->get_bounding_box()->get_low_x();

  idb::IdbRect* rect = blk->get_rect_list()[0];
  int32_t llx = rect->get_low_x();
  int32_t urx = rect->get_high_x();

  int32_t bottom_row = find_rect_bottom_row(rect);
  int32_t top_row = find_rect_top_row(rect);

  for (int i = bottom_row; i <= top_row; ++i) {
    IdbSite* site = idb_rows->get_row_list()[i]->get_site();
    int32_t site_width = site->get_width();
    idb_rows->get_row_list()[i]->set_bounding_box();
    int32_t row_y = idb_rows->get_row_list()[i]->get_bounding_box()->get_low_y();
    int32_t cur_row_site_llx = 0, num = 0, cur_row_site_urx;
    num = (llx - core_llx) / site_width;
    cur_row_site_llx = num * site_width + core_llx;
    num = (urx - core_llx) / site_width + 1;
    cur_row_site_urx = num * site_width + core_llx;
    _filled_sites.push_back(make_pair(row_y, make_pair(cur_row_site_llx, cur_row_site_urx)));
  }

  return _filled_sites.size();
}

/**
 * @brief 读取所有布局障碍，并完成占据site的查找
 *
 */
void TapCellPlacer::initMacroLocation()
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_blockage_list = idb_design->get_blockage_list();

  for (auto blk : idb_blockage_list->get_blockage_list()) {
    if (blk->is_palcement_blockage()) {
      find_placement_blockage_overlap_site(blk);
    }
  }
}
/**
 * @brief  creating endcap cell instances and place them in design
 * @param  endcap_master
 */
void TapCellPlacer::insertEndCaps(std::string endcap_master)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();
  auto idb_rows = idb_layout->get_rows();
  auto idb_cellmaster_list = idb_layout->get_cell_master_list();

  if (!_init_macro) {
    initMacroLocation();
    _init_macro = true;
  }

  std::vector<std::pair<int32_t, std::vector<std::pair<int32_t, int32_t>>>> row_fills;
  int32_t index_row_fills;
  for (auto a : _filled_sites) {
    int32_t y = a.first;
    index_row_fills = find_index_in_row_fills(row_fills, y);
    if (index_row_fills == -1) {
      std::vector<std::pair<int32_t, int32_t>> start_end;
      start_end.push_back(a.second);
      row_fills.push_back(make_pair(y, start_end));
    } else {
      row_fills[index_row_fills].second.push_back(a.second);
    }
  }

  _tapcell_index = _tapcell_start_index;
  idb::IdbCellMaster* endcap = idb_cellmaster_list->find_cell_master(endcap_master);
  if (endcap == nullptr) {
    printf("%s is not valid endcap master \n", endcap_master.c_str());
    return;
  }
  int32_t endcap_width = endcap->get_width();

  int32_t bottom_row_index = 0;
  int32_t top_row_index = idb_rows->get_row_list().size() - 1;

  for (int cur_row = bottom_row_index; cur_row <= top_row_index; ++cur_row) {
    std::vector<std::pair<int32_t, int32_t>> row_fill_check;
    idb::IdbRow* row = idb_rows->get_row_list()[cur_row];
    row->set_bounding_box();
    int32_t row_y = row->get_bounding_box()->get_low_y();

    // if row_y has filled sites
    if (obtain_pair_index(row_fills, row_y) != -1) {
      int32_t index = obtain_pair_index(row_fills, row_y);
      row_fill_check = row_fills[index].second;
    }

    row->set_bounding_box();
    idb::IdbOrient orient = row->get_site()->get_orient();
    int32_t llx = row->get_bounding_box()->get_low_x();
    int32_t lly = row->get_bounding_box()->get_low_y();
    int32_t urx = row->get_bounding_box()->get_high_x();
    int32_t ury = row->get_bounding_box()->get_high_y();

    idb::IdbCellMaster* endcap_master_l = endcap;
    idb::IdbCellMaster* endcap_master_r = endcap_master_l;

    if (static_cast<ssize_t>(endcap_master_l->get_width()) > static_cast<ssize_t>(urx - llx)) {
      continue;
    }
    bool filled = check_site_if_filled(llx, endcap_width, orient, row_fill_check);
    if (!filled) {
      build_tapcell_instance(endcap_master_l, orient, llx, lly, "ENDCAP_");
    }

    int32_t endcap_master_r_width = endcap_master_r->get_width();
    int32_t endcap_master_r_height = endcap_master_r->get_height();

    int32_t row_remaining_space_x = urx - endcap_master_r_width;
    int32_t row_remaining_space_y = ury - endcap_master_r_height;
    if (llx == row_remaining_space_x && lly == row_remaining_space_y) {
      printf("%s have enough space for only one endcap!\n", row->get_name().c_str());
      continue;
    }

    idb::IdbOrient orient_r = orient;
    filled = check_site_if_filled(row_remaining_space_x, endcap_width, orient_r, row_fill_check);
    if (!filled) {
      build_tapcell_instance(endcap_master_r, orient_r, row_remaining_space_x, row_remaining_space_y, "ENDCAP_");
    }
  }
  int32_t endcap_count = _tapcell_index - _tapcell_start_index;
  printf("Endcaps inserted: %d \n", endcap_count);
}

/**
 * @brief  creating tapcell instances and place them in design
 * @param  distance
 * @param  tapcell_master_name
 */
void TapCellPlacer::insertTapCells(double distance, std::string tapcell_master_name)
{
  auto idb_layout = dmInst->get_idb_layout();
  auto idb_cellmaster_list = idb_layout->get_cell_master_list();
  auto idb_rows = idb_layout->get_rows();

  if (!_init_macro) {
    initMacroLocation();
    _init_macro = true;
  }

  idb::IdbCellMaster* tapcell_master = idb_cellmaster_list->find_cell_master(tapcell_master_name);
  _tapcell_index = _tapcell_start_index;
  if (tapcell_master == nullptr) {
    printf("Master %s not found! \n", tapcell_master_name.c_str());
  }
  int32_t dbu = idb_layout->get_units()->get_micron_dbu();
  int32_t tapcell_width = tapcell_master->get_width();
  int32_t distance_min = distance * dbu;
  int32_t distance_max = 2 * distance_min;

  // Get the positions of all filled sites in the row
  std::vector<std::pair<int32_t, std::vector<std::pair<int32_t, int32_t>>>> row_fills;
  int32_t index_row_fills;
  for (auto a : _filled_sites) {
    int32_t y = a.first;
    index_row_fills = find_index_in_row_fills(row_fills, y);
    if (index_row_fills == -1) {
      std::vector<std::pair<int32_t, int32_t>> start_end;
      start_end.push_back(a.second);
      row_fills.push_back(make_pair(y, start_end));
    } else {
      row_fills[index_row_fills].second.push_back(a.second);
    }
  }

  std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>> rows_with_macros_result = obtain_macro_outlines(idb_rows);

  std::vector<std::pair<int32_t, int32_t>> rows_with_macros;

  std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>>::iterator iter;
  for (iter = rows_with_macros_result.begin(); iter != rows_with_macros_result.end(); ++iter) {
    for (size_t i = 0; i != iter->second.size() - 1; ++i) {
      int32_t bottom_row_index;
      int32_t top_row_index;

      bottom_row_index = iter->second[i] - 1;
      top_row_index = iter->second[i + 1];

      if (bottom_row_index >= 0) {
        rows_with_macros.push_back(make_pair(bottom_row_index, 0));
      }
      if (static_cast<size_t>(top_row_index) <= idb_rows->get_row_list().size()) {
        rows_with_macros.push_back(make_pair(top_row_index, 0));
      }
    }
  }

  if (rows_with_macros.size() <= 0) {
    return;
  }

  std::vector<int> rows_with_macros_key_list;
  for (size_t j = 0; j != rows_with_macros.size(); ++j) {
    rows_with_macros_key_list.push_back(rows_with_macros[j].second);
  }
  sort(rows_with_macros_key_list.begin(), rows_with_macros_key_list.end());

  for (size_t row_idx = 0; row_idx != idb_rows->get_row_list().size(); ++row_idx) {
    std::vector<std::pair<int32_t, int32_t>> row_fill_check;

    std::vector<idb::IdbRow*> rows = idb_rows->get_row_list();
    idb::IdbRow* row = rows[row_idx];
    IdbSite* site = row->get_site();
    row->set_bounding_box();
    int32_t row_y = row->get_bounding_box()->get_low_y();

    // if row_y has filled sites
    if (obtain_pair_index(row_fills, row_y) != -1) {
      int32_t index = obtain_pair_index(row_fills, row_y);
      row_fill_check = row_fills[index].second;
    }

    int gaps_above_below = 0;
    size_t row_idx_cmp = rows_with_macros_key_list[0];
    if (!rows_with_macros_key_list.empty() && row_idx_cmp == row_idx) {
      gaps_above_below = 1;
      std::vector<int32_t> tmp;
      for (size_t r = 1; r != rows_with_macros_key_list.size(); ++r) {
        tmp.push_back(rows_with_macros_key_list[r]);
      }
      rows_with_macros_key_list = tmp;
    }

    int32_t site_width = site->get_width();
    row->set_bounding_box();
    idb::IdbRect* row_box = row->get_bounding_box();
    int32_t llx = row_box->get_low_x();
    int32_t lly = row_box->get_low_y();
    int32_t urx = row_box->get_high_x();
    idb::IdbOrient orient = site->get_orient();
    if (!check_master_symmetry(tapcell_master, orient)) {
      printf("sys\n");
      continue;
    }
    int32_t offset = 0;
    int32_t pitch = -1;

    if (row_idx % 2 == 0) {
      offset = distance_min;
    } else {
      offset = distance_max;
    }

    if (row_idx == 0 || row_idx == (rows.size() - 1) || gaps_above_below) {
      pitch = distance_min;
      offset = distance_min;
    } else {
      pitch = distance_max;
    }

    // tapcell_master =
    // _cell_master_list->find_cell_master(tapcell_master_name);

    for (int32_t x = llx + offset; x < urx; x = x + pitch) {
      tapcell_master = idb_cellmaster_list->find_cell_master(tapcell_master_name);
      x = make_site_location(x, site_width, -1, llx);

      bool filled = check_site_if_filled(x, tapcell_width, orient, row_fill_check);
      if (!filled) {
        build_tapcell_instance(tapcell_master, orient, x, lly, "PHY_");
      }
    }
  }
  int32_t tap_count = _tapcell_index - _tapcell_start_index;
  printf("Tapcells Inserted: %d\n ", tap_count);
}

/**
 * @brief  build tapcell instance in design
 * @param  tapcell_master
 * @param  orient
 * @param  x
 * @param  lly
 * @param  prefix
 * @return idb::IdbInstance*
 */
idb::IdbInstance* TapCellPlacer::build_tapcell_instance(idb::IdbCellMaster* tapcell_master, idb::IdbOrient orient, int32_t x, int32_t lly,
                                                        string prefix)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_inst_list = idb_design->get_instance_list();

  string tapcell_instance_name;
  idb::IdbInstance* tapcell = new idb::IdbInstance();
  tapcell->set_cell_master(tapcell_master);
  tapcell->set_orient(orient);
  tapcell->set_coodinate(x, lly);
  tapcell->set_bounding_box();
  tapcell->set_type("DIST");
  tapcell->set_status(IdbPlacementStatus::kFixed);
  tapcell_instance_name = prefix + to_string(_tapcell_index);
  tapcell->set_name(tapcell_instance_name);
  int32_t llx = x;
  int32_t urx = tapcell->get_bounding_box()->get_high_x();
  _filled_sites.push_back(make_pair(lly, make_pair(llx, urx)));
  idb_inst_list->add_instance(tapcell);
  ++_tapcell_index;

  return tapcell;
}

/**
 * @brief 检查单元的镜像方向
 *
 * @param master
 * @param orient
 * @return true
 * @return false
 */
bool check_master_symmetry(idb::IdbCellMaster* master, idb::IdbOrient orient)
{
  bool symmetry_x = master->is_symmetry_x();
  bool symmetry_y = master->is_symmetry_y();

  switch (orient) {
    case idb::IdbOrient::kN_R0:
      return true;
      break;
    case idb::IdbOrient::kFS_MX:
      return symmetry_x;
      break;
    case idb::IdbOrient::kFN_MY:
      return symmetry_y;
      break;
    case idb::IdbOrient::kS_R180:
      return (symmetry_x && symmetry_y);
      break;
    default:
      return false;
      break;
  }
  return false;
}

/**
 * @brief 确定site的位置
 *
 * @param x
 * @param site_width
 * @param direct
 * @param offset
 * @return int32_t
 */
int32_t TapCellPlacer::make_site_location(int32_t x, int32_t site_width, int32_t direct, int32_t offset)
{
  int32_t site_int;

  if (direct == 1) {
    site_int = ceil(double(x - offset) / site_width);
  } else {
    site_int = floor(double(x - offset) / site_width);
  }
  int32_t location = int32_t(site_int * site_width + offset);

  return location;
}

/**
 * @brief  describe the outlines of the macro by rows
 * @param  rows
 * @return std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>>
 */
std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>> TapCellPlacer::obtain_macro_outlines(idb::IdbRows* rows)
{
  std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>> macro_outlines;
  std::pair<int32_t, int32_t> min_max = obtain_rows_min_max_x(rows);

  int32_t min_x = min_max.first;
  int32_t max_x = min_max.second;

  int rows_size = rows->get_row_list().size();

  for (int cur_row = 0; cur_row != rows_size; ++cur_row) {
    idb::IdbRow* row = rows->get_row_list()[cur_row];  //

    int32_t macro0 = -1;
    int32_t macro1 = -1;

    row->set_bounding_box();
    int32_t llx = row->get_bounding_box()->get_low_x();
    int32_t urx = row->get_bounding_box()->get_high_x();

    macro1 = llx;
    if (min_x == llx) {
      macro1 = -1;
    }
    if (macro0 != macro1) {
      if (macro_outlines.count(make_pair(macro0, macro1)) == 0) {
        std::vector<int32_t> cur_row_index;
        cur_row_index.push_back(cur_row);
        macro_outlines.insert(make_pair(make_pair(macro0, macro1), cur_row_index));
      } else {
        macro_outlines[make_pair(macro0, macro1)].push_back(cur_row);
      }
    }
    macro0 = urx;
    if (macro0 != max_x) {
      if (macro_outlines.count(make_pair(macro0, macro1)) == 0) {
        std::vector<int32_t> cur_row_index;
        cur_row_index.push_back(cur_row);
        macro_outlines.insert(make_pair(make_pair(macro0, -1), cur_row_index));
      } else {
        macro_outlines[make_pair(macro0, -1)].push_back(cur_row);
      }
    }
  }

  std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>> macro_outlines_array;

  std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>>::iterator iter;
  for (iter = macro_outlines.begin(); iter != macro_outlines.end(); ++iter) {
    std::vector<int32_t> rows = iter->second;
    std::vector<int32_t> new_rows;

    new_rows.push_back(rows[0]);
    for (size_t i = 1; i != rows.size(); ++i) {
      if ((rows[i] + 1) == rows[i + 1]) {
        continue;
      }
      new_rows.push_back(rows[i]);
      new_rows.push_back(rows[i + 1]);
    }

    new_rows.push_back(rows[rows.size() - 1]);
    int32_t x_min = iter->first.first;
    int32_t x_max = iter->first.second;
    macro_outlines_array.insert(make_pair(make_pair(x_min, x_max), new_rows));
  }
  return macro_outlines_array;
}

/**
 * @brief  find min x coordinate and max x coordinate in bouding boxes of rows
 * @param  rows
 * @return std::pair<int32_t, int32_t>
 */
std::pair<int32_t, int32_t> TapCellPlacer::obtain_rows_min_max_x(idb::IdbRows* rows)
{
  int32_t min_x = -1;
  int32_t max_x = -1;
  for (idb::IdbRow* row : rows->get_row_list()) {
    row->set_bounding_box();
    int32_t llx = row->get_bounding_box()->get_low_x();
    int32_t urx = row->get_bounding_box()->get_high_x();

    if (min_x == -1) {
      min_x = llx;
    } else if (min_x > llx) {
      min_x = llx;
    }

    if (max_x == -1) {
      max_x = urx;
    } else if (max_x < urx) {
      max_x = urx;
    }
  }
  std::pair<int32_t, int32_t> result = std::pair<int32_t, int32_t>(min_x, max_x);
  return result;
}

int obtain_pair_index(std::vector<std::pair<int32_t, std::vector<std::pair<int32_t, int32_t>>>> row_fills, int32_t y)
{
  if (row_fills.empty()) {
    return -1;
  }
  int size = row_fills.size();
  for (int i = 0; i != size; ++i) {
    if (row_fills[i].first == y) {
      return i;
    }
  }
  return -1;
}

bool check_site_if_filled(int32_t x, int32_t width, idb::IdbOrient orient, std::vector<std::pair<int32_t, int32_t>> row_instances)
{
  int32_t x_end = (orient == idb::IdbOrient::kFN_MY || orient == idb::IdbOrient::kS_R180) ? x : x + width;

  for (auto row : row_instances) {
    if (x_end > row.first && x_end < row.second) {
      return true;
    }
  }
  return false;
}
}  // namespace ifp
