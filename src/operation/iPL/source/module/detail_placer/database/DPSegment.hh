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

#ifndef C8A3E2D7_39F3_4BE7_9A6F_97E902363F3F
#define C8A3E2D7_39F3_4BE7_9A6F_97E902363F3F

#include <set>

#include "DPBin.hh"
#include "DPNode.hh"

namespace ipl{

class DPRow;

class DPSegment{

public:
	DPSegment() = default;
	virtual ~DPSegment() = default;

	void set_id(const int id) { _id = id;}
    void set_bound(const Rectangle<int64_t>& rect) { _bound = rect; }
    void set_bin_length_x(int64_t length_x) {_bin_length_x = length_x; }
    void set_bin_length_y(int64_t length_y) {_bin_length_y = length_y; }
    void set_row(DPRow* row) {_row = row;}
    void set_left(DPSegment * left);
    void set_right(DPSegment * right);
    void add_lower(DPSegment * lower);
	void add_upper(DPSegment * upper);

    int get_id() const {return _id;} 
    const Rectangle<int64_t>& get_bound() const { return _bound; }
    int64_t get_width() const {return _bound.get_width() ; }
    std::vector<DPBin>& get_bin_list() {return _bin_list;}
    int get_num_bins() const {return _bin_list.size(); }
    DPBin* get_front_bin() { return &_bin_list.front() ; }
    DPBin* get_back_bin() {return &_bin_list.back();}
    DPSegment* get_left_segment() { return _left;}
    DPSegment* get_right_segment() {return _right;}

    int64_t getCenterX() const {return _bound.get_center().get_x();}
    int64_t getCenterY() const {return _bound.get_center().get_y();}
    std::string getFullId(const std::string & separator = ":") const;
    int getBinIndex(const int64_t pos_x, const bool round_up = false);
	DPBin * getBinByIndex(const int index);
    DPBin * getBinByPosition(const int64_t pos_x);

	int64_t computeDisplacement(const int64_t pos_x, const int64_t pos_y);

    void initBins();
    void insertNode(DPNode * inst);
    void removeNode(DPNode * inst);

private:
    int _id = -1;
    Rectangle<int64_t> _bound;
    int64_t _bin_length_x;
    int64_t _bin_length_y;
    int64_t _node_usage = 0 ;
    DPRow* _row = nullptr;
    std::vector<DPBin> _bin_list;
    std::set<std::string> _segment_list;
    DPSegment * _left = nullptr;
	DPSegment * _right = nullptr;
    std::vector <DPSegment*> _upper_list;
	std::vector <DPSegment*> _lower_list;
    std::vector <DPSegment*> _neighbor_list;
    std::vector <DPSegment*> _vertical_neighbor_list;

};

}


#endif /* C8A3E2D7_39F3_4BE7_9A6F_97E902363F3F */
