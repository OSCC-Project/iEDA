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
#include <algorithm>
#include "DPRow.hh"
#include "DPSegment.hh"
#include "DPBin.hh"


namespace ipl{

DPSite::DPSite(std::string name): _name(name), _width(INT32_MIN),_height(INT32_MIN){}

DPSite::~DPSite(){

}

DPRow::DPRow(std::string row_name, DPSite* site, int32_t site_num): _name(row_name), _site(site), _site_num(site_num)
{

}

DPRow::~DPRow()
{

}

void DPRow::initBlockages()
{
    std::sort(_blockage_list.begin(), _blockage_list.end(),
    [](const Rectangle<int64_t> & block0, const Rectangle<int64_t> & block1) {
        return block0.get_ll_x() < block1.get_ll_x();
    });
	mergeNeighborBlockages();
}

void DPRow::mergeNeighborBlockages()
{
    if (_blockage_list.size() < 2) {
		return;
	}  
	std::deque<Rectangle<int64_t>> merged;
	int num_blocks = _blockage_list.size();
	Rectangle<int64_t> prev_bound = _blockage_list[0];
	for (int index = 1; index < num_blocks; ++index) {
		Rectangle<int64_t> cur_bound = _blockage_list[index];
		if (prev_bound.get_ur_x() >= cur_bound.get_ll_x()) {
			prev_bound.set_upper_right(cur_bound.get_ur_x(), cur_bound.get_ur_y());
		} else {
			Rectangle<int64_t> merged_block(prev_bound.get_ll_x(),prev_bound.get_ll_y(),prev_bound.get_ur_x(),prev_bound.get_ur_y());
			merged.push_back(merged_block);
			prev_bound = cur_bound;
		} 
	}  
	// adding last blockage
	Rectangle<int64_t> merged_block(prev_bound.get_ll_x(), prev_bound.get_ll_y(),prev_bound.get_ur_x(),prev_bound.get_ur_y());
	merged.push_back(merged_block);
	_blockage_list = merged;
}


void DPRow::initSegments()
{
	if (_blockage_list.empty()) {
		const int id = _segment_list.size();
		_segment_list.push_back(DPSegment());
		DPSegment& segment = _segment_list.back();
		segment.set_bound(get_bound());
		segment.set_id(id);
        segment.set_bin_length_x(_bin_length_x);
        segment.set_bin_length_y(_bin_length_y);
		segment.set_row(this);
	} else {
		int64_t left = _bound.get_ll_x();
		int64_t right;
		_segment_list.reserve(_blockage_list.size());
		for (Rectangle<int64_t> & block : _blockage_list) {
			right = block.get_ll_x();
			Rectangle<int64_t> seg_bound = get_bound();
            seg_bound.set_lower_left(left, seg_bound.get_ll_y());
            seg_bound.set_upper_right(right, seg_bound.get_ur_y());
			if (right <= left) {
				left = block.get_ur_x();
				continue;
			} 
			const int id = _segment_list.size();
			_segment_list.push_back(DPSegment());
			DPSegment & segment = _segment_list.back();
			segment.set_bound(seg_bound);
			segment.set_bin_length_x(_bin_length_x);
			segment.set_bin_length_y(_bin_length_y);
			segment.set_row(this);
			segment.set_id(id);
			left = block.get_ur_x();
		} 
		right = _bound.get_ur_x();
		if (left < right) {
			Rectangle<int64_t> seg_bound = get_bound();
            seg_bound.set_lower_left(left, seg_bound.get_ll_y());
            seg_bound.set_upper_right(right, seg_bound.get_ur_y());
			const int id = _segment_list.size();
			_segment_list.push_back(DPSegment());
			DPSegment & segment = _segment_list.back();
			segment.set_bound(seg_bound);
			segment.set_bin_length_x(_bin_length_x);
			segment.set_bin_length_y(_bin_length_y);
			segment.set_row(this);
			segment.set_id(id);
		} 
	} 

	// initialize bins and connect horizontal bin through blockages
	DPBin * left_bin = nullptr;
	DPSegment * left_segment = nullptr;
	for (DPSegment & segment : _segment_list) {
		segment.initBins();
		_num_bins += segment.get_num_bins();

		// connecting horizontal bins through blockages
		DPBin * right_bin = segment.get_front_bin();
		if (left_bin) {
			right_bin->set_left(left_bin);
			left_bin->set_right(right_bin);
		}
		left_bin = segment.get_back_bin();

		// Connecting horizontal segments
		if (left_segment) {
			left_segment->set_right(&segment);
			segment.set_left(left_segment);
		} 
		left_segment = &segment;
	} 
}

bool DPRow::insertNode(DPNode* inst)
{
	int64_t pos_x = inst->getPositionX();
	int64_t pos_y = inst->getPositionY();
	int64_t pos_ux = inst->getPositionX() + inst->getWidth();
	// Initialize the minimum distance to a very large value.
	double min_dist = std::numeric_limits<double>::max();
	int segment_index = -1;
	// Iterate over all segments to find the nearest one.
	for (std::size_t i = 0; i < _segment_list.size(); i++) {
		DPSegment & dp_segment = _segment_list[i];
		int64_t segment_center_x = dp_segment.getCenterX();
		int64_t segment_center_y = dp_segment.getCenterY();
		double dist = std::pow(segment_center_x - pos_x, 2) + std::pow(segment_center_y - pos_y, 2);
		if (dist < min_dist) {
			min_dist = dist;
			segment_index = i;
		}
	}
	// If no segment was found, return false.
	if (segment_index == -1) {
		return false;
	}
	DPSegment & segment = _segment_list[segment_index];
	if(segment.get_width() < inst->getWidth()){
		return false;
	}
	if (pos_x <= segment.get_bound().get_ur_x() && pos_ux >= segment.get_bound().get_ll_x()){
		if (pos_ux >  segment.get_bound().get_ur_x()){
			pos_x = segment.get_bound().get_ur_x() - inst->getWidth();
			placeCell(inst->get_inst(), pos_x , pos_y);
		}
		if (pos_x < segment.get_bound().get_ll_x()){
			pos_x = segment.get_bound().get_ll_x();
			placeCell(inst->get_inst(), pos_x , pos_y);
		}
		segment.insertNode(inst);
		return true;
	}
	return false;
}

void DPRow::placeCell(DPInstance* inst, int64_t pos_x, int64_t pos_y)
{
	inst->updateCoordi(pos_x, pos_y);
}


DPSegment * DPRow::getNearestSegment(const int64_t pos_x, const int64_t pos_y, int64_t width)
{
	// Initialize the minimum distance to a very large value.
	double min_dist = std::numeric_limits<double>::max();
	std::size_t segment_index = -1;
	// Iterate over all segments to find the nearest one.
	for (std::size_t i = 0; i < _segment_list.size(); i++) {
		DPSegment & dp_segment = _segment_list[i];
		int64_t segment_center_x = dp_segment.getCenterX();
		int64_t segment_center_y = dp_segment.getCenterY();
		double dist = std::pow(segment_center_x - pos_x, 2) + std::pow(segment_center_y - pos_y, 2);
		if (dist < min_dist) {
			min_dist = dist;
			segment_index = i;
		}
	}
	DPSegment* segment = &_segment_list[segment_index];
	if(segment->get_width() >= width){
		return segment;
	}
	int64_t disp = std::numeric_limits<int64_t>::max();
	DPSegment* best = nullptr;
	DPSegment* left = segment->get_left_segment();
	DPSegment* right = segment->get_right_segment();
	while(left || right){
		if (left){
			int64_t seg_disp = left->computeDisplacement(pos_x, pos_y);
			if (left->get_width() >= width && seg_disp < disp){
				best = left;
				disp = seg_disp;
				left = nullptr;
			}else{
				left = left->get_left_segment();
			}
		}
		if (right){
			int64_t seg_disp = right ->computeDisplacement(pos_x, pos_y);
			if (right->get_width() >= width && seg_disp < disp){
				best = right;
				disp = seg_disp;
				right = nullptr;
			}else{
				right = right ->get_right_segment();
			}
		}
	}
	return best;
}

int64_t DPRow::computeDisplacementY(const int64_t pos_y)
{
	return std::abs(_bound.get_ll_y() - pos_y);
}

}
