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

#include "NFSpread.hh"
#include "utility/StreamStateSaver.hh"

#include <queue>
#include <iomanip>

namespace ipl {

NFSpread::NFSpread(DPConfig* config, DPDatabase* database, DPOperator* dp_operator)
{
    _config = config;
    _database = database;
    _operator = dp_operator;
}

NFSpread::~NFSpread()
{
}

void NFSpread::runNFSpread() 
{
	init();
	computeAbu();
	cellSpreading();
	computeAbu();
	
}

void NFSpread::init()
{
	computeBinWidth();
	initRows();
	initBlockages();
	connectVerticalSegments();
	connectVerticalBins();
	connectVerticalSegmentsBinsThroughBlockages();
	initNodes();
	removeBlockageOverlap();
	updateInitLegalPos();
}

void NFSpread::computeAbu()
{
    auto* grid_manager = _operator->get_grid_manager();
    _operator->updateGridManager();

    std::vector<double> ratio_usage;
    int64_t grid_cnt = grid_manager->get_grid_cnt_x() * grid_manager->get_grid_cnt_y();
    ratio_usage.reserve(grid_cnt);

	const double area_threashold = grid_manager->get_grid_size_x() * grid_manager->get_grid_size_y() * 0.2;

    float available_ratio = grid_manager->get_available_ratio();

    double abu_1 = 0.0;
	double abu_2 = 0.0;
	double abu_5 = 0.0;
	double abu_10 = 0.0;
	double abu_20 = 0.0;
	double abu_overfilled = 0.0;
	double abu_penalty = 0.0;
	// double abu_npa = 0.0;
	double abu_max_npa = 0.0;

	// int num_abu_bins = 0;
	int num_abu1_bins = 0;
	int num_abu2_bins = 0;
	int num_abu5_bins = 0;
	int num_abu10_bins = 0;
	int num_abu20_bins = 0;
	int num_abu100_bins = 0;
	int num_abu_npa_bins = 0;

	int64_t total_npa = 0;
	int64_t total_bin_area = 0;

    auto& grid_2d_list = grid_manager->get_grid_2d_list();
    for (auto& grid_row : grid_2d_list) {
        for (size_t i = 0; i < grid_row.size(); i++) {
            auto* grid = &grid_row[i];
			double bin_area = grid->grid_area;
			if (bin_area > area_threashold){
				double free_area = grid->placeable_area - grid->fixed_area;
				if (free_area > 0.2 * bin_area){
					double ratio = grid->occupied_area / free_area;
					ratio_usage.push_back(ratio);
					if (ratio > available_ratio) {
						num_abu100_bins++;
						abu_overfilled += ratio;
					} 	
				}else if (grid->occupied_area > 0){
					num_abu_npa_bins++;
					int64_t area_movable = grid->occupied_area;
					int64_t bin_area = grid->grid_area;
					total_npa += area_movable;
					double area_ratio = area_movable / static_cast<double>(bin_area);
					abu_max_npa = std::max(abu_max_npa, area_ratio);
					total_bin_area += bin_area;
				}
			}
        }
    }

    int64_t num_bins = ratio_usage.size();
	std::sort(ratio_usage.begin(), ratio_usage.end(),
		[](const double ratio1, const double ratio2) {
			return ratio1 > ratio2;
		}); 

	// num_abu_bins = num_bins;

	const int index1 = static_cast<int> (0.01 * num_bins);
	const int index2 = static_cast<int> (0.02 * num_bins);
	const int index5 = static_cast<int> (0.05 * num_bins);
	const int index10 = static_cast<int> (0.10 * num_bins);
	const int index20 = static_cast<int> (0.20 * num_bins);

	for (int j = 0; j < index1; ++j) {
		abu_1 += ratio_usage[j];
		if (ratio_usage[j] > available_ratio) {
			num_abu1_bins++;
		}  
	} 
	
	abu_2 = abu_1;
	num_abu2_bins = num_abu1_bins;
	for (int j = index1; j < index2; ++j) {
		abu_2 += ratio_usage[j];
		if (ratio_usage[j] > available_ratio) {
			num_abu2_bins++;
		}  
	} 

	abu_5 = abu_2;
	num_abu5_bins = num_abu2_bins;
	for (int j = index2; j < index5; ++j) {
		abu_5 += ratio_usage[j];
		if (ratio_usage[j] > available_ratio) {
			num_abu5_bins++;
		}  
	} 

	abu_10 = abu_5;
	num_abu10_bins = num_abu5_bins;
	for (int j = index5; j < index10; ++j) {
		abu_10 += ratio_usage[j];
		if (ratio_usage[j] > available_ratio) {
			num_abu10_bins++;
		}  
	} 

	abu_20 = abu_10;
	num_abu20_bins = num_abu10_bins;
	for (int j = index10; j < index20; ++j) {
		abu_20 += ratio_usage[j];
		if (ratio_usage[j] > available_ratio) {
			num_abu20_bins++;
		} 
	} 

    abu_1 = (index1) ? abu_1 / index1 : 0.0;
	abu_2 = (index2) ? abu_2 / index2 : 0.0;
	abu_5 = (index5) ? abu_5 / index5 : 0.0;
	abu_10 = (index10) ? abu_10 / index10 : 0.0;
	abu_20 = (index20) ? abu_20 / index20 : 0.0;
	abu_overfilled = num_abu100_bins ? abu_overfilled / num_abu100_bins  :0.0;
	// abu_npa = num_abu_npa_bins ? total_npa / static_cast<double>(total_bin_area) : 0.0;

	const double abu2_weight = 10.0;
	const double abu5_weight = 4.0;
	const double abu10_weight = 2.0;
	const double abu20_weight = 1.0;

	double penalty_abu2 = (abu_2 / available_ratio) - 1.0;
	penalty_abu2 = std::max(0.0, penalty_abu2);

	double penalty_abu5 = (abu_5 / available_ratio) - 1.0;
	penalty_abu5 = std::max(0.0, penalty_abu5);

	double penalty_abu10 = (abu_10 / available_ratio) - 1.0;
	penalty_abu10 = std::max(0.0, penalty_abu10);

	double penalty_abu20 = (abu_20 / available_ratio) - 1.0;
	penalty_abu20 = std::max(0.0, penalty_abu20);

	double wpenalty_abu2 = abu2_weight * penalty_abu2;
	double wpenalty_abu5 = abu5_weight * penalty_abu5;
	double wpenalty_abu10 = abu10_weight * penalty_abu10;
	double wpenalty_abu20 = abu20_weight * penalty_abu20;

	abu_penalty = (wpenalty_abu2 + wpenalty_abu5 + wpenalty_abu10 + wpenalty_abu20) /
		(abu2_weight + abu5_weight + abu10_weight + abu20_weight);

	std::cout << "\ttarget util     : " << available_ratio << "\n";
	std::cout << "\tABU {2,5,10,20} : " << abu_2
		<< ", " << abu_5 << ", " << abu_10
		<< ", " << abu_20 << "\n";
	std::cout << "\tABU penalty     : " << abu_penalty << "\n";
}

void NFSpread::cellSpreading(const int max_iteration)
{
    _max_iterations = max_iteration;
	_enable_max_displacement = false;
	performNetworkFlow();
	if (_iteration >= _max_iterations) {
		std::cout << "WARNING: stopped cell spreading algorithm. "
			<< " The maximum number of iterations is reached."
			<< " #Iterarions " << _iteration << " MaxNumberIterations: " << _max_iterations
			<< "\n";
	}  
}


void NFSpread::computeBinWidth()
{
	int64_t total_width = 0;
	std::vector<DPInstance*> inst_list = _database->get_design()->get_inst_list();

	for (DPInstance* inst : inst_list){
		if (inst->get_state() == DPINSTANCE_STATE::kFixed){
			continue;
		}
		total_width += inst->get_shape().get_width();
		_num_movable_cells++;
	}
	_bin_length_x = (20 * total_width / static_cast<double>(_num_movable_cells));
	_bin_length_y = _database->get_layout()->get_row_height();
}

void NFSpread::initRows()
{
	auto layout = _database->get_layout();
	for (int32_t i = 0; i < layout->get_row_num(); i++) {
		for (auto* row : layout->get_row_2d_list().at(i)) {
			row->set_id(i);
			row->set_bin_length_x(_bin_length_x);
			row->set_bin_length_y(_bin_length_y);
			_row_lower_pos = std::min(_row_lower_pos, row->get_coordinate().get_y());
		}
	}
}

void NFSpread::initBlockages()
{
	Rectangle<int64_t> core_shape(0,0,_database->get_layout()->get_max_x(),_database->get_layout()->get_max_y());
	std::vector<DPInstance*> inst_list = _database->get_design()->get_inst_list();
	for (DPInstance* inst : inst_list){
		if (inst->get_master() && inst->get_master()->get_type() == DPCELL_TYPE::kMacro){
			int64_t lx = inst->get_shape().get_ll_x();
			int64_t ly = inst->get_shape().get_ll_y();
			int64_t ux = inst->get_shape().get_ur_x();
			int64_t uy = inst->get_shape().get_ur_y();
			Rectangle<int64_t> inst_shape(lx,ly,ux,uy);
			Rectangle<int64_t> overlap = inst_shape.get_intersetion(core_shape);
			if(overlap.get_width() ==0 || overlap.get_height() == 0){
				continue;
			}
			addBlockage(overlap);
		}
	}

	int num_bins = 0;
	auto layout = _database->get_layout();
	for (int32_t i = 0; i < layout->get_row_num(); i++) {
		for (auto* row : layout->get_row_2d_list().at(i)) {
			row->initBlockages();
			row->initSegments();
			num_bins += row->get_num_bins();
		}
	}
	_bin_list.reserve(num_bins);

	for (int32_t i = 0; i < layout->get_row_num(); i++) {
		for (auto* row : layout->get_row_2d_list().at(i)) {
			for (DPSegment& segment : row->get_segment_list()){
				for (DPBin& bin : segment.get_bin_list()){
					_bin_list.push_back(&bin);
				}
			}
		}
	}
}

void NFSpread::addBlockage(const Rectangle<int64_t>& block)
{
	const int low = getRowIndex(block.get_ll_y());
	const int up = getRowIndex(block.get_ur_y());
	for (int index = low; index < up; ++index) {
		for (auto* row : _database->get_layout()->get_row_2d_list().at(index) )
			row->add_blockage(block);
	} 
}

int32_t NFSpread::getRowIndex(const int64_t pos)
{
	const int64_t delta = pos - _row_lower_pos;
	const int64_t row_height = _database->get_layout()->get_row_height();
	return static_cast<int> (delta / row_height);
}

void NFSpread::connectVerticalSegments()
{
	auto layout = _database->get_layout();
	DPRow * lower = layout->get_row_2d_list().at(0)[0];
	DPSegment * lower_segment = lower->getFrontSegment();
	for (int index = 1; index < layout->get_row_num() ; ++index) {
		DPRow * row = layout->get_row_2d_list().at(index)[0];
		DPSegment * first = row->getFrontSegment();
		connectVerticalSegments(lower_segment, first);
		lower_segment = first;
	} 
}

void NFSpread::connectVerticalSegments(DPSegment * lower_first, DPSegment * upper_first) 
{
	DPSegment * lower = lower_first;
	DPSegment * upper = upper_first;

	while (lower && upper) {
		const int64_t lower_left = lower->get_bound().get_ll_x();
		const int64_t lower_right = lower->get_bound().get_ur_x();
		const int64_t upper_left = upper->get_bound().get_ll_x();
		const int64_t upper_right = upper->get_bound().get_ur_x();

		const bool condition1 = lower_left >= upper_left && lower_left < upper_right;
		const bool condition2 = upper_left >= lower_left && upper_left < lower_right;

		if (condition1 || condition2) 
		{
			lower->add_upper(upper);
			upper->add_lower(lower);
		}  

		if (lower_right <= upper_right) 
		{
			lower = lower->get_right_segment();
		}  

		if (upper_right <= lower_right) 
		{
			upper = upper->get_right_segment();
		}  
	}  	
}

void NFSpread::connectVerticalBins()
{
	auto layout = _database->get_layout();
	DPRow * lower = layout->get_row_2d_list().at(0)[0];
	DPSegment * lower_segment = lower->getFrontSegment();
	DPBin * lower_bin = nullptr;
	if (lower_segment)
		lower_bin = lower_segment->get_front_bin();
	for (int index = 1; index < layout->get_row_num(); ++index) {
		DPRow * row = layout->get_row_2d_list().at(index)[0]; 
		DPSegment * first = row->getFrontSegment();
		DPBin * first_bin = nullptr;
		if (first)
			first_bin = first->get_front_bin();
		connectVerticalBins(lower_bin, first_bin);
		lower_bin = first_bin;
	}  
}

void NFSpread::connectVerticalBins(DPBin * lower_first, DPBin * upper_first) 
{
	DPBin * lower = lower_first;
	DPBin * upper = upper_first;

	while (lower && upper) {
		const int64_t lower_left = lower->get_bound().get_ll_x();
		const int64_t lower_right = lower->get_bound().get_ur_x();
		const int64_t upper_left = upper->get_bound().get_ll_x();
		const int64_t upper_right = upper->get_bound().get_ur_x();

		const bool condition1 = lower_left >= upper_left && lower_left < upper_right;
		const bool condition2 = upper_left >= lower_left && upper_left < lower_right;

		if (condition1 || condition2) 
		{
			lower->add_upper(upper);
			upper->add_lower(lower);
		}  

		if (lower_right <= upper_right) 
		{
			lower = lower->get_right();
		}  

		if (upper_right <= lower_right) 
		{
			upper = upper->get_right();
		}  
	}  
}


void NFSpread::connectVerticalSegmentsBinsThroughBlockages()
{
	auto layout = _database->get_layout();
	for (int32_t id = 1; id < layout->get_row_num(); ++id) {
		DPRow* lower_row = layout->get_row_2d_list().at(id - 1)[0];
		DPRow* upper_row = layout->get_row_2d_list().at(id)[0];
		if (!upper_row->hasBlockages()) {
			continue;
		}  

		DPSegment * lower = lower_row->getFrontSegment();
		const std::deque<Rectangle<int64_t>> & blocks = upper_row->allBlockages();
		int block_id = 0;
		while (lower && block_id < static_cast<int>(blocks.size())) {
			const Rectangle<int64_t> & lower_bds = lower->get_bound();
			const Rectangle<int64_t> & block_bds = blocks[block_id];
			const bool left_condition = block_bds.get_ll_x() > lower_bds.get_ll_x() && block_bds.get_ll_x() < lower_bds.get_ur_x();
			const bool right_condition = block_bds.get_ur_x() > lower_bds.get_ll_x() && block_bds.get_ur_x() < lower_bds.get_ur_x();

			if (left_condition || right_condition) {
				connectVerticalSegmentsBinsThroughBlockages(lower, upper_row, block_bds);
			}  

			if (block_bds.get_ur_x() <= lower_bds.get_ur_x()) {
				++block_id;
			}  

			if (lower_bds.get_ur_x() < block_bds.get_ur_x()) {
				lower = lower->get_right_segment();
			}  
		}  
	}  
}


void NFSpread::connectVerticalSegmentsBinsThroughBlockages(DPSegment * lower, DPRow * upper, const Rectangle<int64_t> & block) 
{
	int row_id = upper->get_id() + 1;
	int64_t left_x = std::max(block.get_ll_x(), lower->get_bound().get_ll_x());
	auto layout = _database->get_layout();
	int32_t row_num = layout->get_row_num();
	while (row_id < row_num && left_x < block.get_ur_x()){
		DPRow* row = getRowByIndex(row_id);
		int64_t pos_x = row->get_bound().get_ll_x();
		int64_t pos_y = row->get_bound().get_ll_y();
		pos_x = left_x;
		DPSegment* segment = row->getNearestSegment(pos_x, pos_y, 0.1*_bin_length_x);
		if (segment){
			const Rectangle<int64_t>& seg_bds = segment->get_bound();
			if (pos_x > seg_bds.get_ll_x() && pos_x < seg_bds.get_ur_x()){
				segment->add_lower(lower);
				lower->add_upper(segment);
				connectVerticalBinsThroughBlockages(lower, segment, left_x, block.get_ur_x());
				left_x = seg_bds.get_ur_x();
			}
		}
		++row_id;
	}
}

DPRow * NFSpread::getRowByIndex(const int index) 
{
	auto layout = _database->get_layout();
	if (index >= 0 && index < layout->get_row_num())
		return layout->get_row_2d_list().at(index)[0];
	return nullptr;
}

void NFSpread::connectVerticalBinsThroughBlockages(DPSegment* lower, DPSegment* upper, const int64_t left_x, const int64_t right_x)
{
	DPBin* lower_bin = lower->getBinByPosition(left_x);
	DPBin* upper_bin = upper->getBinByPosition(left_x);

	while(lower_bin && upper_bin && lower_bin->getPositionX() < right_x
		&& upper_bin->getPositionX() < right_x){
			lower_bin->add_upper(upper_bin);
			upper_bin->add_lower(lower_bin);
			int64_t lower_right_x = lower->get_bound().get_ur_x();
			int64_t upper_right_x = upper->get_bound().get_ur_x();
			if (lower_right_x <= upper_right_x){
				lower_bin = lower_bin->get_right();
			}
			if (upper_right_x <= lower_right_x){
				upper_bin = upper_bin->get_right();
			}
		}
}

void NFSpread::initNodes()
{
	_node_list.reserve(_num_movable_cells);
	std::vector<DPInstance*> inst_list = _database->get_design()->get_inst_list();
	for (DPInstance* inst : inst_list){
		if (inst->get_state() == DPINSTANCE_STATE::kFixed){
			continue;
		}
		alignCellToRow(inst);
		const int id = _node_list.size();
		_node_list.push_back(DPNode());
		DPNode & leg_inst = _node_list.back();
		leg_inst.set_id(id);
		leg_inst.set_inst(inst);
		if(!insertNode(&leg_inst)){
			_dirty_node_list.push_back(&leg_inst);
		}
		_inst_to_node_map[inst] = id;
	}
}

void NFSpread::alignCellToRow(DPInstance* inst)
{
	DPRow* upper_row = _database->get_layout()->get_row_2d_list().back()[0];
	const int64_t upper_row_y = upper_row->get_coordinate().get_y();
	int64_t pos_x = inst->get_coordi().get_x();
	int64_t pos_y = inst->get_coordi().get_y();
	DPRow* row = getRow(pos_y);
	int64_t posY = row->get_coordinate().get_y();
	if (posY != pos_y){
		int64_t row_center_y = row->get_coordinate().get_y() +  row->get_site()->get_height() * 0.5;
		if (pos_y >= row_center_y && posY < upper_row_y){
			pos_y = row->get_coordinate().get_y() + row->get_site()->get_height();
		}else{
			pos_y = posY;
		}
		placeCell(inst, pos_x, pos_y);
	}
}

DPRow* NFSpread::getRow(const int64_t pos, const bool nearest) 
{
	int index = getRowIndex(pos);
	if (index >= _database->get_layout()->get_row_num() || index < 0) {
		if (!nearest)
			return nullptr;
		if (index < 0)
			return _database->get_layout()->get_row_2d_list().front()[0];
		return _database->get_layout()->get_row_2d_list().back()[0];
	} 
	return _database->get_layout()->get_row_2d_list()[index][0];
} 

void NFSpread::placeCell(DPInstance* inst, int64_t pos_x, int64_t pos_y)
{
	inst->updateCoordi(pos_x, pos_y);
}


bool NFSpread::insertNode(DPNode* inst)
{
	const int64_t pos_y = inst->getPositionY();
	DPRow* row = getRow(pos_y);
	return row->insertNode(inst);
}


void NFSpread::removeBlockageOverlap()
{
	for (DPNode* inst: _dirty_node_list){
		DPSegment* segment = getNearestSegment(inst);
		if (segment) {
			int64_t pos_x = inst->getPositionX();
			int64_t pos_y = inst->getPositionY();
			int64_t seg_pos_x = segment->get_bound().get_ll_x();
			int64_t seg_pos_y = segment->get_bound().get_ll_y();
			int64_t seg_pos_ux = segment->get_bound().get_ur_x();

			pos_y = seg_pos_y;
			if (pos_x < seg_pos_x)
				pos_x = seg_pos_x;
			if (pos_x > seg_pos_ux - inst->getWidth())
				pos_x = seg_pos_ux - inst->getWidth();
			placeCell(inst->get_inst(), pos_x, pos_y);
			segment->insertNode(inst);
		} else {
			std::cout << "ERROR: Node: " << inst->get_id()
				<< " was not removed overlap to blockage."
				<< "\n";
		} 
	}  
	_dirty_node_list.clear();	
}

DPSegment* NFSpread::getNearestSegment(DPNode* inst)
{
	const int64_t init_pos_Y = inst->getPositionY();
	const int64_t init_pos_X = inst->getPositionX();
	int init_row_id = getRowIndex(init_pos_Y);
	DPRow* row = getRowByIndex(init_row_id);
	DPSegment* best_segment = row->getNearestSegment(init_pos_X, init_pos_Y, inst->getWidth());
	int num_rows = _database->get_layout()->get_row_num();

	int64_t displacement = std::numeric_limits<int64_t>::max();
	if (best_segment){
		displacement = best_segment->computeDisplacement(init_pos_X, init_pos_Y);
	}
	std::queue<int> row_ids;
	if (init_row_id > 0 ){
		row_ids.push(init_row_id - 1);
	}
	if (init_row_id + 1 < num_rows){
		row_ids.push(init_row_id + 1);
	}
	while (!row_ids.empty()){
		int id = row_ids.front();
		row_ids.pop();
		DPRow* row = getRowByIndex(id);
		int64_t row_disp = row->computeDisplacementY(init_pos_Y);
		if (row_disp > displacement){
			continue;
		}
		if (id >0 && id < init_row_id){
			row_ids.push(id -1 );
		}
		if (id +1 < num_rows && id > init_row_id){
			row_ids.push(id + 1);
		}
		DPSegment* segment = row->getNearestSegment(init_pos_X, init_pos_Y, inst->getWidth());
		if (segment){
			int64_t disp = segment->computeDisplacement(init_pos_X, init_pos_Y);
			if (disp < displacement){
				displacement = disp;
				best_segment = segment;
			}
		}
	}
	return best_segment;
}


void NFSpread::updateInitLegalPos()
{
	std::vector<DPInstance*> inst_list = _database->get_design()->get_inst_list();
	for (DPInstance* inst : inst_list){
		if (inst->get_state() == DPINSTANCE_STATE::kFixed){
			continue;
		}
		inst->set_origin_shape(inst->get_shape());
	}
}


void NFSpread::performNetworkFlow()
{
	_iteration = 0;
	_total_overflow = updateOverflowedBins();
	report(std::cout, true);

	while (_overflowed_bin_list.size() > 0 && _iteration <= _max_iterations) {
		if(_max_overfilled_area_ratio < 1.0) {
			break;
		} 

		updateMaxDisplacement();

		if (_iteration < 10 || (_iteration + 1) % 50 == 0 || _iteration > 800) {
			report(std::cout);
		} 

		for (DPBin * bin : _overflowed_bin_list) {
			if (bin->getSupply() == 0){
				continue;
			}
			int64_t flow = computeInitialFlow(bin);

			TNode root;
			root._bin = bin;
			root._flow = flow;
			std::deque<TNode*> paths;
			TNode * sink = pathAugmentationBranchBound(&root, paths);

			if (sink) {
				moveCells(sink);
			}  
		}  
		_iteration++;
		_total_overflow = updateOverflowedBins();
	}
	report();
}

int64_t NFSpread::updateOverflowedBins()
{
    int64_t total_overflow = 0;
    _max_overfilled_area_ratio = 0.0;
	_avg_overfilled_area_ratio = 0.0;
    _overflowed_bin_list.clear();

	for (DPBin* bin : _bin_list){
		if (bin->getSupply() > 0){
			total_overflow += bin->getSupply();
			_overflowed_bin_list.push_back(bin);
			double ratio = bin->getSupply() / static_cast<double> (bin->getPlaceableSpace());
			_avg_overfilled_area_ratio += ratio;
			_max_overfilled_area_ratio = std::max(_max_overfilled_area_ratio, ratio);
		}
	}
	if (!_overflowed_bin_list.empty()){
		_avg_overfilled_area_ratio /= _overflowed_bin_list.size();
	}
	return total_overflow;
}

void NFSpread::report(std::ostream & out, const bool initial) 
{
	const int N = 15;
	StreamStateSaver stream_out(out);

	out << std::left;
	if (initial) {
		out << "\n";
		out << std::setw(N) << "Iteration";
		out << std::setw(N) << "#OF Bins";
		out << std::setw(N) << "Total OF";
		out << std::setw(N) << "Max Disp (DBU)";
		out << std::setw(N) << "Avg OFAR"; 
		out << std::setw(N) << "Max OFAR";
	} else {
		out << std::setw(N) << (_iteration + 1);
		out << std::setw(N) << _overflowed_bin_list.size();
		out << std::setw(N) << _total_overflow;
		out << std::setw(N) << _max_displacement;
		out << std::setw(N) << _avg_overfilled_area_ratio;
		out << std::setw(N) << _max_overfilled_area_ratio;
	}
	out << "\n";
	stream_out.restore();
} 

void NFSpread::updateMaxDisplacement()
{
	double displ = _alpha * _bin_length_x + _betha * _iteration * _bin_length_x;
	_max_displacement = std::max(static_cast<int64_t> (displ), _bin_length_x);
}

int64_t NFSpread::computeInitialFlow(DPBin* bin)
{
	int64_t avg = std::ceil(static_cast<int64_t>(bin->get_usage()) / bin->get_num_nodes());
	int64_t supply = bin->getSupply();
	int64_t flow = std::min(avg, supply);

	if (bin->get_width() >= _bin_length_x){
		return flow;
	}

	// If the bin is surrounded by blockages in the horizontal in the row
	for (DPNode * inst : bin->get_node_list()) {
		int64_t cell_disp = inst->computeDisplacement();
		if (cell_disp > _max_displacement && _enable_max_displacement)
			continue;
		if (inst->getWidth() > flow)
			continue;
		flow = inst->getWidth();
	}  
	return flow;
}


TNode* NFSpread::pathAugmentationBranchBound(TNode * root, std::deque<TNode*> &paths)
{
    std::set<DPBin*> visited_bins; 

	// The top TNode* is the node with the lowest cost.
	// source: https://en.cppreference.com/w/cpp/container/priority_queue
	auto cmp = [](const TNode* node0, const TNode * node1) {
		return node0->_cost > node1->_cost;
	};
	std::priority_queue< TNode*, std::vector<TNode*>, decltype(cmp)> nodes(cmp);
	nodes.push(root);
	TNode *best_path = nullptr;

	do {
		TNode * parent = nodes.top();
		nodes.pop();
		if (best_path && best_path->_cost < parent->_cost) {
			continue;
		}  

		DPBin * src = parent->_bin;
		visited_bins.insert(src);
		std::deque<TNode> & children = parent->_children;

		for (DPBin * sink : src->get_neighbor_list()) {
			if (visited_bins.find(sink) != visited_bins.end()) { 
				continue; 
			}

			int64_t required_flow = std::max(parent->_flow - src->getDemand(), (int64_t)0);
			double cost = 0;
			int64_t out_supply = computeFlow(src, sink, required_flow, cost);  

			if (out_supply <= 0) {
				continue;
			}  

			children.push_back(TNode());
			TNode & sink_node = children.back();
			sink_node._bin = sink;
			sink_node._cost = cost + parent->_cost;
			sink_node._flow = out_supply;
			sink_node._parent = parent;
		}  

		for (size_t i = 0; i < children.size(); ++i) {
			TNode * nd = &children[i];
			if (nd->_flow <= nd->_bin->getDemand()) {
				paths.push_back(nd);
				if (best_path && (best_path->_cost > nd->_cost)) {
					best_path = nd;
				}  
				if (!best_path) {
					best_path = nd;
				}  
			} else {
				nodes.push(nd);
			} 
		}  

	} while (!nodes.empty());

	return best_path;
}

int64_t NFSpread::computeFlow(DPBin* src, DPBin* sink, const int64_t flow, double& cost)
{
	if (flow == 0) {
		return 0;
	}

	bool isHorNeighbor = isHorizontalNeighbor(src, sink);
	int64_t supply_flow = computeNodeFlow(src, sink, flow, isHorNeighbor, cost);

	if (isHorNeighbor) {
		supply_flow = std::min(supply_flow, flow);
	} 

	return supply_flow;

}

bool NFSpread::isHorizontalNeighbor(DPBin* src, DPBin* sink)
{
	if (src->get_bound().get_ll_y() != sink->get_bound().get_ll_y()){
		return false;
	}
	if (src->get_bound().get_ur_x() == sink->get_bound().get_ll_x()){
		return true;
	}
	if (src->get_bound().get_ll_x() == sink->get_bound().get_ur_x()){
		return true;
	}
	return false;
}

int64_t NFSpread::computeNodeFlow(DPBin* src, DPBin* sink, const int64_t flow, const bool isHorNeighbor, double& cost)
{
	const int64_t bin_disp = computeBinDisplacemnet(src, sink);

	if (bin_disp > _max_displacement) {
		return 0;
	} 

	int64_t supply_flow = 0;
	std::vector<NodeFlow> instances;
	instances.reserve(src->get_num_nodes());

	selectNodes(src, sink, isHorNeighbor, instances);

	int64_t sink_flow = 0;
	for (NodeFlow & inst : instances) {
		supply_flow += inst._src_overlap;
		cost += inst._cost;

		if (!isHorNeighbor) {
			sink_flow += inst._node->getWidth() - inst._sink_overlap;
		} 
		if (supply_flow >= flow) {
			return isHorNeighbor ? supply_flow : sink_flow;
		} 
	} 
	return 0;
}

int64_t NFSpread::computeBinDisplacemnet(DPBin* src, DPBin* sink)
{
	if (isNeighbor(src, sink)) {
		return 0;
	} 

	const int64_t llx = std::max(src->get_bound().get_ll_x(), sink->get_bound().get_ll_x());
	const int64_t urx = std::min(src->get_bound().get_ur_x(), sink->get_bound().get_ur_x());
	const int64_t lly = std::max(src->get_bound().get_ll_y(), sink->get_bound().get_ll_y());
	const int64_t ury = std::min(src->get_bound().get_ur_y(), sink->get_bound().get_ur_y());

	return std::abs((urx - llx) + (ury - lly));

}

bool NFSpread::isNeighbor(DPBin* src, DPBin* sink)
{
	if (src->get_bound().get_ll_y() == sink->get_bound().get_ur_y()){
		return true;
	}
	if (src->get_bound().get_ur_y() == sink->get_bound().get_ll_y()){
		return true;
	}
	if (src->get_bound().get_ur_x() == sink->get_bound().get_ll_x()){
		return true;
	}
	if (src->get_bound().get_ll_x() == sink->get_bound().get_ur_x()){
		return true;
	}
	return false;
}


void NFSpread::selectNodes(DPBin* src, DPBin* sink, const bool isHorNeighbor, std::vector<NodeFlow> &instances)
{
	const Rectangle<int64_t>& src_bound = src->get_bound();
	const Rectangle<int64_t>& sink_bound = sink->get_bound();
	const int64_t sink_placeable = sink->getPlaceableSpace();

	for (DPNode* inst : src->get_node_list())
	{
		const int64_t width = inst->getWidth();

		if (width > sink_placeable){
			continue;
		}

		const Rectangle<int64_t>& inst_bound = inst->getBound();
		Rectangle<int64_t> src_overlap_rect = src_bound.get_intersetion(inst_bound);
		Rectangle<int64_t> sink_overlap_rect = sink_bound.get_intersetion(inst_bound);

		if (isHorNeighbor){
			int64_t total_overlap = sink_overlap_rect.get_width() + src_overlap_rect.get_width();
			if (total_overlap < width){
				continue;
			}
		}
		double cost = 0.0;
		int64_t disp = computeDisplacement(sink, inst, cost);
		bool next = _enable_max_displacement || (!_enable_max_displacement && !isHorNeighbor);
		if (next && disp > _max_displacement) {
			continue;
		} 
		instances.push_back(NodeFlow());
		NodeFlow & instFlow = instances.back();
		instFlow._displacement = disp;
		instFlow._node = inst;
		instFlow._cost = cost;
		instFlow._src_overlap = src_overlap_rect.get_width();
		instFlow._sink_overlap = sink_overlap_rect.get_width();
	} 
	sortNodes(instances);
}

int64_t NFSpread::computeDisplacement(const DPBin* sink, const DPNode* inst, double& cost)
{
	int64_t target_pos_x = inst->getPositionX();
	int64_t target_pos_y = sink->getPositionY();

	const int64_t length = inst->getWidth();
	const int64_t upper_sink = sink->get_bound().get_ur_x() - length;
	if (inst->getPositionX() > upper_sink || inst->getInitialPositionX() > upper_sink){
		target_pos_x = upper_sink;
	}
	if (inst->getPositionX() < sink->get_bound().get_ll_x() || inst->getInitialPositionX() < sink->get_bound().get_ll_x()){
		target_pos_x = sink->get_bound().get_ll_x();
	}

	int64_t pos_x = inst->getPositionX();
	int64_t pos_y = inst->getPositionY();

	int64_t init_pos_x = inst->getInitialPositionX();
	int64_t init_pos_y = inst->getInitialPositionY();

	int64_t current_disp_x = std::abs(pos_x - init_pos_x);
	int64_t current_disp_y = std::abs(pos_y - init_pos_y);

	int64_t target_disp_x = std::abs(target_pos_x - init_pos_x);
	int64_t target_disp_y = std::abs(target_pos_y - init_pos_y);

	cost = (target_disp_x - current_disp_x) + (target_disp_y - current_disp_y);
	return cost;
}

void NFSpread::sortNodes(std::vector<NodeFlow> & instances)
{
	std::sort(instances.begin(), instances.end(), [](NodeFlow & inst1, NodeFlow & inst2) {
		return inst1._cost < inst2._cost;
	}); 
}


bool NFSpread::moveCells(TNode* leaf)
{
	TNode * node_sink = leaf;
	TNode * node_src = leaf->_parent;
	// int64_t last_flow = node_sink->_flow;
	while (node_src) {
		DPBin * sink = node_sink->_bin;
		DPBin * src = node_src->_bin;
		int64_t flow = node_sink->_flow;
		const bool isNeighbor = isHorizontalNeighbor(src, sink);

		if (isNeighbor) { // horizontal and partial moves 
			// last_flow = moveHorizontalNeighborFlow(src, sink, flow);
			moveHorizontalNeighborFlow(src, sink, flow);
		} else { // vertical moves
			// last_flow = moveFullCellFlow(src, sink, flow);
			moveFullCellFlow(src, sink, flow);
		}

		node_sink = node_src;
		node_src = node_src->_parent;
	}
	return true;
}

int64_t NFSpread::moveHorizontalNeighborFlow(DPBin* src, DPBin* sink, const int64_t flow)
{
	std::vector<NodeFlow> nodes;
	selectNodes(src, sink, true, nodes);

	int64_t moved_flow = 0;
	for (NodeFlow & node : nodes) {
		DPNode * inst = node._node;
		const int64_t required_move_flow = std::min(node._src_overlap, flow - moved_flow);
		const Rectangle<int64_t>& inst_shape = inst->getBound();

		int64_t target_pos_x, target_pos_y;
		computeHorizontalPosition(sink, inst_shape, required_move_flow, target_pos_x, target_pos_y);
		moveNode(inst, src->get_segment(), sink->get_segment(), target_pos_x, target_pos_y);
		moved_flow += required_move_flow;
		if (moved_flow >= flow) {
			break;
		}  
	}
	return moved_flow;
}

void NFSpread::computeHorizontalPosition(DPBin* sink, const Rectangle<int64_t>& inst, const int64_t flow, int64_t& pos_x, int64_t& pos_y)
{
	pos_x = inst.get_ll_x();
	pos_y = sink->get_bound().get_ll_y();

	const Rectangle<int64_t> sink_bound = sink->get_bound();
	Rectangle<int64_t> sink_overlap_rect = sink_bound.get_intersetion(inst);
	int64_t width_overlap = sink_overlap_rect.get_width();
	
	const int64_t length = inst.get_width();
	int64_t candidate_flow = length - width_overlap;
	candidate_flow = std::min(flow, candidate_flow);

	// Move cell to left
	if (inst.get_ll_x() > sink->get_bound().get_ll_x()){
		pos_x = sink->get_bound().get_ur_x() - (width_overlap + candidate_flow);
	}else{ // move cell to right
		pos_x = sink->get_bound().get_ll_x() - (length - width_overlap);
		pos_x += candidate_flow;
	}
}

void NFSpread::moveNode(DPNode* node, DPSegment* src, DPSegment* sink ,int64_t target_pos_x, int64_t target_pos_y)
{
	removeNode(src, node);
	node->get_inst()->updateCoordi(target_pos_x, target_pos_y);
	insertNode(sink, node);
}
    
int64_t NFSpread::moveFullCellFlow(DPBin* src, DPBin* sink, const int64_t flow)
{
	std::vector<NodeFlow> nodes;
	bool isHorNeighbor = isHorizontalNeighbor(src, sink);
	selectNodes(src, sink, isHorNeighbor, nodes);

	int64_t moved_flow = 0;
	for (NodeFlow & node : nodes) {
		DPNode * inst = node._node;
		const int64_t width = inst->getWidth() - node._sink_overlap;
		if (moved_flow + width > flow) {
			break;
		}  

		int64_t target_pos_x = inst->getPositionX();
		int64_t target_pos_y = sink->getPositionY();

		const int64_t length = inst->getWidth();
		const int64_t upper_sink = sink->get_bound().get_ur_x() - length;
		if (inst->getPositionX() > upper_sink || inst->getInitialPositionX() > upper_sink){
			target_pos_x = upper_sink;
		}
		if (inst->getPositionX() < sink->get_bound().get_ll_x() || inst->getInitialPositionX() < sink->get_bound().get_ll_x()){
			target_pos_x = sink->get_bound().get_ll_x();
		}

		moveNode(inst, src->get_segment(), sink->get_segment(), target_pos_x, target_pos_y);
		moved_flow += width;
	}
	return moved_flow;
}

}