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

#ifndef D9EE2B35_D4DA_4138_B27C_56C8E711CD4D
#define D9EE2B35_D4DA_4138_B27C_56C8E711CD4D

#include <set>

#include "DPNode.hh"
#include "data/Rectangle.hh"

namespace ipl{

class DPSegment;

class DPBin{

public:
	DPBin() = default;
	virtual ~DPBin() = default;

	void set_id(const int id) { _id = id;}
    void set_bound(const Rectangle<int64_t>& rect) { _bound = rect; }
    void set_segment(DPSegment* segment) {_segment = segment;}
    void set_cache_node(DPNode * node);
    void set_right(DPBin* right) { _right = right; _neighbor_list.push_back(right); }
    void set_left(DPBin* left) { _left = left; _neighbor_list.push_back(left); }
    void add_lower(DPBin * bin) {_lower_list.push_back(bin); _neighbor_list.push_back(bin);}  
	void add_upper(DPBin * bin) { _upper_list.push_back(bin); _neighbor_list.push_back(bin);}  

    const Rectangle<int64_t> & get_bound() const {return _bound;} 
    int64_t get_width() const {return _bound.get_width();}  
	int64_t getPlaceableSpace() const {return get_width();}  
    int64_t get_usage() const {return _usage;}  
	int64_t getSupply() const {	return std::max((int64_t) 0, get_usage() - getPlaceableSpace());}  
	int64_t getDemand() const {	return std::max((int64_t) 0, getPlaceableSpace() - get_usage());}  
    DPSegment* get_segment(){return _segment;}
	DPBin * get_right() const { return _right;} 
    DPBin * get_left() const { return _left;} 
    int get_num_nodes(){ return _node_list.size();}
    int64_t getPositionX() const{ return _bound.get_ll_x();}
    int64_t getPositionY() const{ return _bound.get_ll_y();}
	std::vector<DPBin*> &get_neighbor_list() {return _neighbor_list;} 

    struct cmp {
        bool operator()(const DPNode* inst0, const DPNode* inst1) const{
            const int64_t pos0 = inst0->getPositionX();
            const int64_t pos1 = inst1->getPositionX();
            if (pos0 == pos1)
                return inst0->get_id() < inst1->get_id();
            return pos0 < pos1;
        } 
    }; 
    const std::set<DPNode*, DPBin::cmp> & get_node_list() const{ return _node_list;}

    void insertNode(DPNode * node);
    void removeNode(DPNode * inst);
    void removeCacheNode(DPNode * node);
    void resertCacheNode() {_cache_node = nullptr;}  

private:
    int _id = -1;
    Rectangle<int64_t> _bound;
    DPSegment* _segment;
    DPBin * _left = nullptr;
	DPBin * _right = nullptr;
    std::vector <DPBin*> _neighbor_list;
    std::vector <DPBin*> _upper_list;
	std::vector <DPBin*> _lower_list;
    int64_t _usage= 0;
    std::set<DPNode*, cmp>_node_list;
    DPNode* _cache_node = nullptr;

};

}

#endif /* D9EE2B35_D4DA_4138_B27C_56C8E711CD4D */
