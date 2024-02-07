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
/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-01 21:14:21
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-06 15:56:10
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPRow.hh
 * @Description: 
 * 
 * 
 */
/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-01 21:14:21
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-05 12:05:04
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPRow.hh
 * @Description: Row and Site of detail placement
 * 
 * 
 */
#ifndef IPL_DPROW_H
#define IPL_DPROW_H

#include <string>
#include <vector>
#include <deque>

#include "data/Orient.hh"
#include "DPInterval.hh"

namespace ipl {

class DPSegment;
class DPNode;
class DPBin;

class DPSite
{
public:
    DPSite() = delete;
    explicit DPSite(std::string name);
    DPSite(const DPSite&) = delete;
    DPSite(DPSite&&) = delete;
    ~DPSite();

    DPSite& operator=(const DPSite&) = delete;
    DPSite& operator=(DPSite&&) = delete;

    // getter
    std::string get_name() const { return _name;}
    int32_t get_width() const { return _width;}
    int32_t get_height() const { return _height;}

    // setter
    void set_width(int32_t width) { _width = width;}
    void set_height(int32_t height) { _height = height;}

private:
    std::string _name;
    int32_t _width;
    int32_t _height;
};


class DPRow
{
public:
    DPRow() = delete;
    DPRow(std::string row_name, DPSite* site, int32_t site_num);
    DPRow(const DPRow&) = delete;
    DPRow(DPRow&&) = delete;
    ~DPRow();

    DPRow& operator=(const DPRow&) = delete;
    DPRow& operator=(DPRow&&) = delete;

    // getter
    std::string get_name() const { return _name;}
    DPSite* get_site() const { return _site;}
    int32_t get_site_num() const { return _site_num;}
    const Point<int32_t>& get_coordinate() const { return _coordinate;}
    const Orient& get_row_orient() const { return _orient;}
    int get_id() const {return _id;}
    int get_num_bins() const {return _num_bins;}
    std::vector<DPSegment>& get_segment_list(){ return _segment_list;}
    DPSegment* getFrontSegment() { return &_segment_list.front(); }
    const std::deque<Rectangle<int64_t>>& allBlockages() const {return _blockage_list;}
    bool hasBlockages() const { return !_blockage_list.empty(); }
    const Rectangle<int64_t>& get_bound() const { return _bound; }

    // setter
    void set_id(const int id) {_id = id;}
    void set_bin_length_x(int64_t length_x) {_bin_length_x = length_x; }
    void set_bin_length_y(int64_t length_y) {_bin_length_y = length_y; }
    void add_blockage(const Rectangle<int64_t>& rect) {_blockage_list.push_back(rect); }
    void set_coordinate(int32_t lx, int32_t ly) {_coordinate = std::move(Point<int32_t>(lx,ly));}
    void set_orient(Orient orient) { _orient = std::move(orient);}
    void set_bound(const Rectangle<int64_t>& rect) { _bound = rect; }
    void add_interval(DPInterval* interval) {_interval_list.push_back(interval);}

    void initBlockages();
    void initSegments();
    DPSegment * getNearestSegment(const int64_t pos_x, const int64_t pos_y, int64_t width);
    void mergeNeighborBlockages();
	int64_t computeDisplacementY(const int64_t pos_y);
    bool insertNode(DPNode* inst);
    void placeCell(DPInstance* inst, int64_t pos_x, int64_t pos_y);

private:
    std::string _name;
    Orient _orient;
    DPSite* _site;
    int32_t _site_num;
    Point<int32_t> _coordinate;
    std::vector<DPInterval*> _interval_list;

    int _id = -1;
    int64_t _bin_length_x = -1;
    int64_t _bin_length_y = -1;
    std::deque<Rectangle<int64_t>> _blockage_list;
    std::vector<DPSegment> _segment_list;
    Rectangle<int64_t> _bound;
    int _num_bins = 0;
};
}
#endif