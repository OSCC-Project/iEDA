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

#include <cmath>

#include "DPSegment.hh"
#include "DPBin.hh"
#include "DPRow.hh"

namespace ipl{

void DPSegment::initBins()
{
	const int64_t bin_width = _bin_length_x;
	const int num_bins = std::ceil(get_width()/static_cast<float>(bin_width));
	_bin_list.reserve(num_bins);
	Rectangle<int64_t> bound = get_bound();
	const int64_t right_pos = bound.get_ur_x();
	DPBin * left = nullptr;
	do {
		bound.set_upper_right(std::min(bound.get_ll_x() + bin_width, right_pos), bound.get_ur_y()) ;
		// Merging remain space to the current bin.
		if (right_pos - bound.get_ur_x() < bin_width)
            bound.set_upper_right(right_pos, bound.get_ur_y());
		const int id = _bin_list.size();
		_bin_list.push_back(DPBin());
		DPBin & bin = _bin_list.back();
		bin.set_id(id);
		bin.set_bound(bound);
		bin.set_segment(this);
		bound.set_lower_left(bound.get_ur_x(), bound.get_ll_y());
		if (left) {
			left->set_right(&bin);
			bin.set_left(left);
		}  
		left = &bin;
		// connecting horizontal bins 
	} while (bound.get_ur_x() + bin_width <= right_pos);    
}

void DPSegment::set_left(DPSegment * left) 
{
	const std::string & id = left->getFullId();
	if (_segment_list.find(id) == _segment_list.end()) {
		_left = left;
		_neighbor_list.push_back(left);
		_segment_list.insert(id);
	}  
} 

void DPSegment::set_right(DPSegment * right) 
{
	const std::string & id = right->getFullId();
	if (_segment_list.find(id) == _segment_list.end()) {
		_right = right;
		_neighbor_list.push_back(right);
		_segment_list.insert(id);
	}  
} 

std::string DPSegment::getFullId(const std::string & separator) const 
{
	int row_id = _row->get_id();
	return std::to_string(row_id) + separator + std::to_string(get_id());
}  

void DPSegment::add_lower(DPSegment * lower) 
{
	const std::string & id = lower->getFullId();
	if (_segment_list.find(id) == _segment_list.end()) {
		_lower_list.push_back(lower);
		_neighbor_list.push_back(lower);
		_segment_list.insert(id);
		if (lower->get_bound().get_ur_y() == _bound.get_ll_y()) {
			_vertical_neighbor_list.push_back(lower);
		}  
	}  
} 

void DPSegment::add_upper(DPSegment * upper) 
{
	const std::string & id = upper->getFullId();
	if (_segment_list.find(id) == _segment_list.end()) {
		_upper_list.push_back(upper);
		_neighbor_list.push_back(upper);
		_segment_list.insert(id);
		if (upper->get_bound().get_ll_y() == _bound.get_ur_y()) {
			_vertical_neighbor_list.push_back(upper);
		}  
	}  
} 

void DPSegment::insertNode(DPNode * inst) 
{
	int left = getBinIndex(inst->getPositionX(), false);
	int right = getBinIndex(inst->getPositionX() + inst->getWidth(), true);
	for (int index = left; index < right; ++index) {
		DPBin * bin = getBinByIndex(index);
		bin->insertNode(inst);
	}  
	_node_usage += inst->getWidth();
}

void DPSegment::removeNode(DPNode * inst) 
{
	int left = getBinIndex(inst->getPositionX(), false);
	int right = getBinIndex(inst->getPositionX() + inst->getWidth(), true);
	for (int index = left; index < right; ++index) {
		DPBin * bin = getBinByIndex(index);
		bin->removeNode(inst);
	}  
	_node_usage -= inst->getWidth();
} 

int DPSegment::getBinIndex(const int64_t pos_x, const bool round_up) 
{
	const int64_t delta = pos_x - _bound.get_ll_x();
	const int64_t length = _bin_length_x;
	int num_bins = round_up ? get_num_bins() : get_num_bins() - 1;
	int index = static_cast<int> (delta / length);
	if (round_up) {
		index = delta % length ? index + 1 : index;
	}  
	return std::min(index, num_bins);
}  

DPBin * DPSegment::getBinByIndex(const int index) 
{
	if (index >= 0 && index < get_num_bins())
		return &_bin_list[index];
	return nullptr;
}  

DPBin * DPSegment::getBinByPosition(const int64_t pos_x)
{
	const int id = getBinIndex(pos_x);
	return getBinByIndex(id);
}


int64_t DPSegment::computeDisplacement(const int64_t pos_x, const int64_t pos_y)
{
	const int64_t disp_y = std::abs(pos_y - _bound.get_ll_y());
	if (pos_x >= _bound.get_ll_x() && pos_x <= _bound.get_ur_x()){
		return disp_y;
	}
	const int64_t disp_x = std::min(std::abs(_bound.get_ll_x() - pos_x), std::abs(_bound.get_ur_x() - pos_x));
	return disp_x + disp_y;
}

}