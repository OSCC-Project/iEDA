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
 * @File Name: dm_init.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "checker/check_connection.h"
#include "idm.h"
#include "omp.h"

namespace idm {

bool DataManager::isOnDieBoundary(IdbInstance* io_cell)
{
  IdbOrient io_cell_orient = io_cell->get_orient();
  int io_cell_llx = io_cell->get_bounding_box()->get_low_x();
  int io_cell_urx = io_cell->get_bounding_box()->get_high_x();
  int io_cell_lly = io_cell->get_bounding_box()->get_low_y();
  int io_cell_ury = io_cell->get_bounding_box()->get_high_y();

  auto idb_die = _layout->get_die();
  auto bounding_box = idb_die->get_bounding_box();
  int die_llx = bounding_box->get_low_x();
  int die_urx = bounding_box->get_high_x();
  int die_lly = bounding_box->get_low_y();
  int die_ury = bounding_box->get_high_y();

  switch (io_cell_orient) {
    case IdbOrient::kS_R180:
      if (io_cell_ury == die_ury) {
        return true;
      }
      break;
    case IdbOrient::kN_R0:
      if (io_cell_lly == die_lly) {
        return true;
      }
      break;

    case IdbOrient::kE_R270:
      if (io_cell_llx == die_llx) {
        return true;
      }
      break;
    case IdbOrient::kW_R90:
      if (io_cell_urx == die_urx) {
        return true;
      }
      break;
    default:
      break;
  }
  return false;
}

bool DataManager::isOnDieBoundary(int32_t io_cell_llx, int32_t io_cell_lly, int32_t io_cell_urx, int32_t io_cell_ury,
                                  IdbOrient io_cell_orient)
{
  auto idb_die = _layout->get_die();
  auto bounding_box = idb_die->get_bounding_box();
  int die_llx = bounding_box->get_low_x();
  int die_urx = bounding_box->get_high_x();
  int die_lly = bounding_box->get_low_y();
  int die_ury = bounding_box->get_high_y();

  switch (io_cell_orient) {
    case IdbOrient::kS_R180:
      if (io_cell_ury <= die_ury) {
        return true;
      }
      break;
    case IdbOrient::kN_R0:
      if (io_cell_lly >= die_lly) {
        return true;
      }
      break;

    case IdbOrient::kE_R270:
      if (io_cell_llx >= die_llx) {
        return true;
      }
      break;
    case IdbOrient::kW_R90:
      if (io_cell_urx <= die_urx) {
        return true;
      }
      break;
    default:
      break;
  }
  return false;
}

bool DataManager::isOnIOSite(int32_t llx, int32_t lly, int32_t urx, int32_t ury, IdbOrient orient)
{
  //   auto corner_site = _layout->get_sites()->find_site("corner");
  //   auto iocell_site = _layout->get_sites()->find_site("pad");
  auto corner_site = _layout->get_sites()->get_corner_site();
  auto iocell_site = _layout->get_sites()->get_io_site();

  if (corner_site == nullptr || iocell_site == nullptr) {
    return false;
  }

  int32_t x_offset = corner_site->get_width();
  int32_t y_offset = corner_site->get_height();
  int32_t site_width = iocell_site->get_width();
  // int32_t site_height = iocell_site->get_height();

  auto idb_die = _layout->get_die();
  auto bounding_box = idb_die->get_bounding_box();

  int32_t die_llx = bounding_box->get_low_x();
  int32_t die_lly = bounding_box->get_low_y();
  int32_t x_start = die_llx + x_offset;
  int32_t y_start = die_lly + y_offset;

  if (orient == IdbOrient::kE_R270 || orient == IdbOrient::kW_R90) {
    int32_t y_to_bottom = lly - y_start;
    if (y_to_bottom % site_width != 0) {
      printf("do not match IOSite\n");
      return false;
    }
  } else if (orient == IdbOrient::kN_R0 || orient == IdbOrient::kS_R180) {
    int32_t x_to_left = llx - x_start;
    if (x_to_left % site_width != 0) {
      printf("do not match IOSite\n");
      return false;
    }
  }

  return true;
}

bool DataManager::checkInstPlacer(int32_t llx, int32_t lly, int32_t urx, int32_t ury, IdbOrient orient)
{
  return isOnDieBoundary(llx, lly, urx, ury, orient) && isOnIOSite(llx, lly, urx, ury, orient);
}
/**
 * @brief check net connectivity
 *
 * @param net
 * @return true
 * @return false
 */
bool DataManager::isNetConnected(std::string net_name)
{
  if (_design == nullptr) {
    return false;
  }

  auto net = _design->get_net_list()->find_net(net_name);

  return net == nullptr ? false : isNetConnected(net);
}

bool DataManager::isNetConnected(IdbNet* net)
{
  CheckNet check_net(net);

  return CheckInfo::kConnected == check_net.checkNetConnection() ? true : false;
}

std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int> DataManager::isAllNetConnected()
{
  std::vector<std::string> disconnected_net_list;
  std::vector<std::string> one_pin_net_list;

  omp_lock_t lck;
  omp_init_lock(&lck);

#pragma omp parallel for schedule(dynamic)

  for (auto net : _design->get_net_list()->get_net_list()) {
    if (net->get_pin_number() < 2) {
      omp_set_lock(&lck);

      one_pin_net_list.push_back(net->get_net_name());

      omp_unset_lock(&lck);

      continue;
    }

    CheckNet check_net(net);
    CheckInfo check_result = check_net.checkNetConnection();

    if (CheckInfo::kDisconnected == check_result) {
      omp_set_lock(&lck);

      disconnected_net_list.push_back(net->get_net_name());

      omp_unset_lock(&lck);
    }
  }

  omp_destroy_lock(&lck);

  for (auto disconnected_net : disconnected_net_list) {
    std::cout << "[Error] Disconneted net [pin number >= 2] : " << disconnected_net << std::endl;
  }

  if (one_pin_net_list.size() > 0) {
    std::cout << "[Error] Disconneted nets [pin number < 2] : " << one_pin_net_list.size() << " / "
              << _design->get_net_list()->get_net_list().size() << std::endl;
  }

  if (disconnected_net_list.size() > 0) {
    std::cout << "[Error] Disconneted nets [pin number >= 2] : " << disconnected_net_list.size() << " / "
              << _design->get_net_list()->get_net_list().size() << std::endl;
  }

  bool b_result = disconnected_net_list.size() > 0 ? false : true;

  int num_net = _design->get_net_list()->get_net_list().size();

  return std::make_tuple(b_result, disconnected_net_list, one_pin_net_list, num_net);
}

}  // namespace idm
