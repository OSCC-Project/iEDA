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

#ifndef B1E589EF_D8A6_4167_AA2D_6641DA2FB08F
#define B1E589EF_D8A6_4167_AA2D_6641DA2FB08F

#include <vector>
#include <deque>
#include <set>
#include <iostream>

#include "config/DetailPlacerConfig.hh"
#include "database/DPDatabase.hh"
#include "DPOperator.hh"
#include "database/DPNode.hh"
#include "database/DPBin.hh"
#include "database/DPSegment.hh"

namespace ipl {

struct TNode {
	int64_t _flow = 0;
	double _cost = 0.0;
	DPBin * _bin = nullptr;
	TNode * _parent = nullptr;
	std::deque<TNode> _children;
};

struct NodeFlow {
	DPNode * _node;
	int64_t _src_overlap = 0;
	int64_t _sink_overlap = 0;
	int64_t _displacement = 0;
	double _cost = 0.0;
}; 

class NFSpread
{
public:
    NFSpread() = default;
    NFSpread(DPConfig* config, DPDatabase* database, DPOperator* dp_operator);
    NFSpread(const NFSpread&) = delete;
    NFSpread(NFSpread&&) = delete;
    ~NFSpread();

    NFSpread& operator=(const NFSpread&) = delete;
    NFSpread& operator=(NFSpread&&) = delete;

    void runNFSpread();

private:
    DPConfig* _config;
    DPDatabase* _database;
    DPOperator* _operator;

    int64_t _num_movable_cells = 0;
    int64_t _bin_length_x = -1;
    int64_t _bin_length_y = -1;
    int32_t _row_lower_pos = std::numeric_limits<int32_t>::max();
	int _iteration = 0;
    int _max_iterations = 1000;
	bool _enable_max_displacement = false;
    int64_t _total_overflow = 0;
	double _max_overfilled_area_ratio = 0.0;
	double _avg_overfilled_area_ratio = 0.0;
	double _alpha = 0.6;
	double _betha = 0.05;
    int64_t _max_displacement = 0;
    
    std::vector<DPBin*> _bin_list;
    std::deque<DPBin*> _overflowed_bin_list;
    std::vector<DPNode> _node_list; 
    std::deque<DPNode* > _dirty_node_list;
    std::map<DPInstance*, int> _inst_to_node_map;

    void init();
    void computeAbu();
    void cellSpreading(const int max_iteration = 1000);
    
    void computeBinWidth();
    void initRows();
    void initBlockages();
    void connectVerticalSegments();
    void connectVerticalSegments(DPSegment * lower_first, DPSegment * upper_first);
	void connectVerticalBins();
    void connectVerticalBins(DPBin * lower_first, DPBin * upper_first);
	void connectVerticalSegmentsBinsThroughBlockages();
    void connectVerticalSegmentsBinsThroughBlockages(DPSegment * lower, DPRow * upper, const Rectangle<int64_t> & block);
    void connectVerticalBinsThroughBlockages(DPSegment* lower, DPSegment* upper, const int64_t left_x, const int64_t right_x);
    int64_t getNumMovableCells() const {return _num_movable_cells;}
    void addBlockage(const Rectangle<int64_t>& block);
    int32_t getRowIndex(const int64_t pos);
    DPRow * getRowByIndex(const int index);
    void initNodes();
    void alignCellToRow(DPInstance* inst);
    bool insertNode(DPNode* inst);
	DPRow * getRow(const int64_t pos, const bool nearest = false);
    void placeCell(DPInstance* inst, int64_t pos_x, int64_t pos_y);
    void removeBlockageOverlap();
    DPSegment* getNearestSegment(DPNode* inst);
    void updateInitLegalPos();

	void performNetworkFlow();
    int64_t updateOverflowedBins();
    void report(std::ostream & out = std::cout, const bool initial = false);
    void updateMaxDisplacement();
    int64_t computeInitialFlow(DPBin* grid);
	TNode * pathAugmentationBranchBound(TNode * root, std::deque<TNode*> &paths);
    bool moveCells(TNode* leaf);

    int64_t computeFlow(DPBin* src, DPBin* sink, const int64_t flow, double& cost);
    bool isHorizontalNeighbor(DPBin* src, DPBin* sink);
    int64_t computeNodeFlow(DPBin* src, DPBin* sink, const int64_t flow, const bool isHorNeighbor, double& cost);
    int64_t computeBinDisplacemnet(DPBin* src, DPBin* sink);
    bool isNeighbor(DPBin* src, DPBin* sink);
    void selectNodes(DPBin* src, DPBin* sink, const bool isHorNeighbor, std::vector<NodeFlow> &instances);
    int64_t computeDisplacement(const DPBin* sink, const DPNode* inst, double& cost);
    void sortNodes(std::vector<NodeFlow> & instances);
    int64_t moveHorizontalNeighborFlow(DPBin* src, DPBin* sink, const int64_t flow);
    int64_t moveFullCellFlow(DPBin* src, DPBin* sink, const int64_t flow);
    void computeHorizontalPosition(DPBin* sink, const Rectangle<int64_t>& inst, const int64_t flow, int64_t& pos_x, int64_t& pos_y);
    void moveNode(DPNode* node, DPSegment* src, DPSegment* sink ,int64_t target_pos_x, int64_t target_pos_y);
	void insertNode(DPSegment* segment, DPNode* inst){ segment->insertNode(inst);}
    void removeNode(DPSegment* segment, DPNode* inst){ segment->removeNode(inst);}
};

}

#endif /* B1E589EF_D8A6_4167_AA2D_6641DA2FB08F */
