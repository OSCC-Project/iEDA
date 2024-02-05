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

#ifndef A5B2ED52_23D4_4BA4_9A6C_D5D5C0623D52
#define A5B2ED52_23D4_4BA4_9A6C_D5D5C0623D52


#include <iostream>

// Inpired by Boost. 

// Helper class to store output stream configuration (flags, precision, width)
// and restore them. Automatically restore the configuration and this class
// is destructred.

class StreamStateSaver {
private:
	std::ostream &out;
	std::ios_base::fmtflags	flags;
	std::streamsize precision;
	std::streamsize width;
	
public:
	
	StreamStateSaver(std::ostream &out) : out(out) {
		flags = out.flags();
		precision = out.precision();
		width = out.width();
	} 
	
	~StreamStateSaver() {
		restore();
	} 
	
	void restore() {
		out.flags(flags);
		out.precision(precision);
		out.width(width);		
	} 
	
}; 

#endif /* A5B2ED52_23D4_4BA4_9A6C_D5D5C0623D52 */
