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

#include "DPBin.hh"
#include "DPSegment.hh"

namespace ipl{

void DPBin::insertNode(DPNode * node)
{
	int64_t inst_x = node->getPositionX();
	int64_t inst_ux = node->getPositionX() + node->getWidth();
	int64_t bin_x = _bound.get_ll_x();
	int64_t bin_ux = _bound.get_ur_x();
	int64_t lx = std::max(bin_x, inst_x);
	int64_t ux = std::min(bin_ux, inst_ux);
	_usage += (ux - lx) ;
	_node_list.insert(node);
	set_cache_node(node);
}

void DPBin::set_cache_node(DPNode * node) 
{
	if (!node) {
		return;
	} 
	int64_t center = (_bound.get_ll_x() + _bound.get_ur_x()) * 0.5;
	if (center >= node->getPositionX() && center <= (node->getPositionX() + node->getWidth())) {
		_cache_node = node;
	} 
}

void DPBin::removeNode(DPNode * node) 
{
	int64_t inst_x = node->getPositionX();
	int64_t inst_ux = node->getPositionX() + node->getWidth();
	int64_t bin_x = _bound.get_ll_x();
	int64_t bin_ux = _bound.get_ur_x();
	int64_t lx = std::max(bin_x, inst_x);
	int64_t ux = std::min(bin_ux, inst_ux);
	_usage -= (ux - lx) ;
	_node_list.erase(node);
	removeCacheNode(node);
}

void DPBin::removeCacheNode(DPNode * node) 
{
	if (!node || !_cache_node) {
		return;
	} 
	DPNode * cache = _cache_node;
	if (cache->get_id() == node->get_id())
		resertCacheNode();
}

}