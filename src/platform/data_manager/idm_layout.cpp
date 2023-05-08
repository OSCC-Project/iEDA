/**
 * @File Name: dm_layout.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "idm.h"

namespace idm {

void DataManager::initDie(int ll_x, int ll_y, int ur_x, int ur_y)
{
  IdbDie* die = _layout->get_die();
  die->reset();
  die->add_point(ll_x, ll_y);
  die->add_point(ur_x, ur_y);
}

uint64_t DataManager::dieArea()
{
  IdbDie* die = _layout->get_die();

  return die->get_area();
}

float DataManager::dieUtilization()
{
  uint64_t inst_area = netlistInstArea() + timingInstArea();

  float utilization = ((double) inst_area) / dieArea();

  return utilization;
}

uint64_t DataManager::coreArea()
{
  IdbCore* core = _layout->get_core();

  return core->get_bounding_box()->get_area();
}

float DataManager::coreUtilization()
{
  uint64_t inst_area = netlistInstArea() + timingInstArea();

  float utilization = ((double) inst_area) / coreArea();

  return utilization;
}

IdbRow* DataManager::createRow(string row_name, string site_name, int32_t orig_x, int32_t orig_y, IdbOrient site_orient, int32_t num_x,
                               int32_t num_y, int32_t step_x, int32_t step_y)
{
  IdbSites* site_list = _layout->get_sites();
  if (site_list == nullptr) {
    return nullptr;
  }
  IdbSite* site = site_list->find_site(site_name);
  if (site == nullptr) {
    return nullptr;
  }

  IdbRows* row_list_ptr = _layout->get_rows();
  if (row_list_ptr == nullptr) {
    return nullptr;
  }

  return row_list_ptr->createRow(row_name, site, orig_x, orig_y, site_orient, num_x, num_y, step_x, step_y);
}

IdbOrient DataManager::getDefaultOrient(int32_t coord_x, int32_t coord_y)
{
  IdbOrient orient = IdbOrient::kNone;

  IdbRows* row_list_ptr = _layout->get_rows();
  if (row_list_ptr == nullptr) {
    return orient;
  }

  /// find row that contains coordinate x y
  for (auto row : row_list_ptr->get_row_list()) {
    if (row->is_horizontal() && row->get_original_coordinate()->get_y() == coord_y) {
      orient = row->get_orient();
      break;
    } else {
      if (row->get_original_coordinate()->get_x() == coord_x) {
        orient = row->get_orient();
        break;
      }
    }
  }

  return orient;
}

}  // namespace idm
