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
 * @File Name: contest_evaluation.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "contest_dm.h"
#include "contest_preprocess.h"
#include "contest_util.h"

namespace ieda_contest {

int dx[] = {0, 0, 1, -1};  // 右、左、下、上
int dy[] = {1, -1, 0, 0};

// 定义一个结构体来包含instance和dis
struct InstanceDistance
{
  idb::IdbInstance* instance;
  double dis;

  bool operator>(const InstanceDistance& other) const { return dis > other.dis; }
  bool operator<(const InstanceDistance& other) const { return dis < other.dis; }
};

struct Element
{
  int x, y;     // 坐标
  double area;  // 面积

  bool operator>(const Element& other) const { return area > other.area; }
  bool operator<(const Element& other) const { return area < other.area; }
};

void ContestPreprocess::place()
{
  // std::vector<ContestInstance>& contest_instance_list = _data_manager->get_database()->get_instance_list();

  // double inst_area = 0;
  // int inst_pin = 0;
  // for (idb::IdbInstance* idb_instance : _data_manager->get_idb_design()->get_instance_list()->get_instance_list()) {
  //   inst_area += idb_instance->get_bounding_box()->get_area();
  // inst_pin += idb_instance->get_pin_list()->get_pin_num();
  // }

  double die_utilization = _data_manager->get_idb_layout()->get_die()->get_utilization();
  double max_area_ratio = 1.00;
  // double max_pin_ratio = 0.9;

  // calculate gcell number, max_area
  double gcellWidth = _data_manager->get_database()->get_single_gcell_x_span();
  double gcellHeight = _data_manager->get_database()->get_single_gcell_y_span();
  double gcellArea = gcellWidth * gcellHeight;
  double max_Gcell_area = std::max(max_area_ratio, die_utilization) * gcellArea;
  int num_gcell_x = static_cast<int>(_data_manager->get_idb_layout()->get_die()->get_width() / gcellWidth + 1);
  int num_gcell_y = static_cast<int>(_data_manager->get_idb_layout()->get_die()->get_height() / gcellHeight + 1);
  std::cout << " gcellWidth = " << gcellWidth << " gcellHeight = " << gcellHeight << " max_Gcell_area = " << max_Gcell_area
            << " num_gcell_x = " << num_gcell_x << " num_gcell_y = " << num_gcell_y << std::endl;

  // int max_Gcell_pin = static_cast<int>(inst_pin / die_utilization * max_pin_ratio / (num_gcell_x * num_gcell_y));

  int pattern = 1;  // 1: satisfied max_area, 2: satisfied max_pin

  std::vector<std::vector<double>> gcell_Area(num_gcell_x, std::vector<double>(num_gcell_y, 0.0));
  std::priority_queue<Element> gcell_Area_queues;

  std::vector<std::vector<std::priority_queue<InstanceDistance>>> gcell_queues(
      num_gcell_x, std::vector<std::priority_queue<InstanceDistance>>(num_gcell_y));
  if (pattern == 1) {
    for (idb::IdbInstance* idb_instance : _data_manager->get_idb_design()->get_instance_list()->get_instance_list()) {
      // 看中点在哪个gcell，并计算距离这个gcell中点的距离
      int inst2gcell_x = static_cast<int>(idb_instance->get_bounding_box()->get_middle_point_x() / gcellWidth);
      int inst2gcell_y = static_cast<int>(idb_instance->get_bounding_box()->get_middle_point_y() / gcellHeight);
      double dis = abs(idb_instance->get_bounding_box()->get_middle_point_x() - gcellWidth * inst2gcell_x - gcellWidth / 2)
                   + abs(idb_instance->get_bounding_box()->get_middle_point_y() - gcellHeight * inst2gcell_y - gcellHeight / 2);

      // 按照dis从大到小把inst加入对应的gcell的优先队列gcell_queues中，并更新gcell_Area来计算每个gcell的当前总面积
      InstanceDistance instance_distance;
      instance_distance.instance = idb_instance;
      instance_distance.dis = dis;
      gcell_Area[inst2gcell_x][inst2gcell_y] += idb_instance->get_bounding_box()->get_area();
      gcell_queues[inst2gcell_x][inst2gcell_y].push(instance_distance);
    }

    // for (int ii = 0; ii < num_gcell_x;ii++) {
    //   for (int jj = 0; jj < num_gcell_y;jj++) {
    //     int size1 = gcell_queues[ii][jj].size();
    //     for (int kk = 0; kk < size1; kk++) {
    //       std::cout<<" top = "<<gcell_queues[ii][jj].top().dis<<std::endl;
    //       gcell_queues[ii][jj].pop();
    //     }
    //     std::cout<<"-----------------------------------------------"<<std::endl;
    //   }
    // }
    // exit(0);
    double ori_total = 0, ori_mean = 0, ori_max = 0, ori_min = 1e+09;
    for (int i = 0; i < num_gcell_x; i++) {
      for (int j = 0; j < num_gcell_y; j++) {
        ori_total += gcell_Area[i][j];
        if (gcell_Area[i][j] > ori_max)
          ori_max = gcell_Area[i][j];
        if (gcell_Area[i][j] < ori_min)
          ori_min = gcell_Area[i][j];
      }
    }
    ori_mean = ori_total / (num_gcell_x * num_gcell_y);
    std::cout << " ori_total = " << ori_total << " ori_mean = " << ori_mean << " ori_max = " << ori_max << " ori_min = " << ori_min
              << std::endl;

    // 构建优先队列gcell_Area_queues，将每个gcell按gcell的面积从大到小进行排列
    for (int i = 0; i < num_gcell_x; i++) {
      for (int j = 0; j < num_gcell_y; j++) {
        Element gcell_element;
        gcell_element.x = i;
        gcell_element.y = j;
        gcell_element.area = gcell_Area[i][j];
        gcell_Area_queues.push(gcell_element);
      }
    }
    // int size2 = gcell_Area_queues.size();
    // for (int kk = 0; kk < size2; kk++) {
    //   std::cout<<" top = "<<gcell_Area_queues.top().area<<std::endl;
    //   gcell_Area_queues.pop();
    // }
    // exit(0);
    bool isOverflow = true;
    while (isOverflow) {
      Element top_gcell = gcell_Area_queues.top();
      gcell_Area_queues.pop();
      Element tar_gcell;
      // 如果top_gcell的面积大于可容许的最大面积，则说明有overflow
      if (top_gcell.area > max_Gcell_area) {
        // std::cout<<" max_gcell_Area = "<<max_Gcell_area<<" Cur_max_gcell_area = "<<top_gcell.area<<std::endl;
        // std::cout<<__LINE__<<std::endl;
        // 找到面积最大的这个gcell中距离中心点最远的单元，记为moveInst，从gcell_queues和gcell_Area移除这个单元
        InstanceDistance moveInst = gcell_queues[top_gcell.x][top_gcell.y].top();
        gcell_queues[top_gcell.x][top_gcell.y].pop();
        gcell_Area[top_gcell.x][top_gcell.y] -= moveInst.instance->get_bounding_box()->get_area();
        top_gcell.area -= moveInst.instance->get_bounding_box()->get_area();

        // 建立temp队列，用于以当前gcell为中心，向外BFS
        std::queue<Element> temp;
        temp.push(top_gcell);
        // top_gcell面积最大，所以其他一定比它小
        tar_gcell = top_gcell;
        std::set<std::pair<int, int>> visited;
        visited.insert(std::make_pair(top_gcell.x, top_gcell.y));
        while (!temp.empty()) {
          int size_temp = temp.size();
          // std::cout<<__LINE__<<std::endl;
          // 往上下左右做BFS，每层BFS时，选择其中面积最小的gcell作为tar_gcell
          for (int i = 0; i < size_temp; i++) {
            Element temp1 = temp.front();
            temp.pop();

            for (int dir = 0; dir < 4; dir++) {
              int newX = temp1.x + dx[dir];
              int newY = temp1.y + dy[dir];

              if (newX >= 0 && newX < num_gcell_x && newY >= 0 && newY < num_gcell_y) {
                if (visited.find(std::make_pair(newX, newY)) != visited.end())
                  continue;
                Element temp2;
                temp2.x = newX;
                temp2.y = newY;
                temp2.area = gcell_Area[newX][newY];
                if (gcell_Area[newX][newY] < tar_gcell.area) {
                  tar_gcell = temp2;
                }
                temp.push(temp2);
                visited.insert(std::make_pair(temp2.x, temp2.y));
              }
            }
          }
          // 如果当前找到面积最小的tar_gcell加入moveInst后满足面积约束，则退出
          if (tar_gcell.area + moveInst.instance->get_bounding_box()->get_area() <= max_Gcell_area) {
            break;
          }
        }
        if (gcell_Area[tar_gcell.x][tar_gcell.y] + moveInst.instance->get_bounding_box()->get_area() > max_Gcell_area) {
          std::cout << " Don't find avaliable gcell" << std::endl;
          exit(0);
        }
        // 将moveInst加入tar_gcell中，由于满足面积约束，所以不会再次取出。更新gcell_Area
        gcell_queues[tar_gcell.x][tar_gcell.y].push(moveInst);
        gcell_Area[tar_gcell.x][tar_gcell.y] += moveInst.instance->get_bounding_box()->get_area();
        gcell_Area_queues.push(top_gcell);
      } else {
        isOverflow = false;
      }
      // std::cout<<" gcell_Area_queues.top() = "<<gcell_Area_queues.top().area<<std::endl;
      // exit(0);
    }
    double cur_total = 0, cur_mean = 0, cur_max = 0, cur_min = 1e+09;
    for (int i = 0; i < num_gcell_x; i++) {
      for (int j = 0; j < num_gcell_y; j++) {
        cur_total += gcell_Area[i][j];
        if (gcell_Area[i][j] > cur_max)
          cur_max = gcell_Area[i][j];
        if (gcell_Area[i][j] < cur_min)
          cur_min = gcell_Area[i][j];
      }
    }
    cur_mean = cur_total / (num_gcell_x * num_gcell_y);
    std::cout << " cur_total = " << cur_total << " cur_mean = " << cur_mean << " cur_max = " << cur_max << " cur_min = " << cur_min
              << std::endl;
    for (int i = 0; i < num_gcell_x; i++) {
      for (int j = 0; j < num_gcell_y; j++) {
        if (gcell_Area[i][j] > max_Gcell_area) {
          std::cout << " ====================Overflow ==================" << std::endl;
          exit(0);
        }
      }
    }
  } else if (pattern == 2) {
  } else {
    std::cout << "Warning: XXXXXXXXXXXXXXXXXX" << std::endl;
  }

  // 根据gcell中存的inst，将inst坐标更新为gcell中点
  for (int i = 0; i < num_gcell_x; i++) {
    for (int j = 0; j < num_gcell_y; j++) {
      int size = gcell_queues[i][j].size();
      for (int k = 0; k < size; k++) {
        InstanceDistance temp = gcell_queues[i][j].top();
        gcell_queues[i][j].pop();
        double center_coord_x = (i + 0.5) * gcellWidth;
        double center_coord_y = (j + 0.5) * gcellHeight;
        idb::IdbCoordinate<int32_t> coord(center_coord_x, center_coord_y);
        temp.instance->set_coodinate(coord, true);
      }
    }
  }
}

}  // namespace ieda_contest
