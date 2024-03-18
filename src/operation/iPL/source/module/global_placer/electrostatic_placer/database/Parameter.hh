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
 * @Author: S.J Chen
 * @Date: 2022-03-06 21:26:05
 * @LastEditTime: 2022-10-27 19:41:20
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/global_placer/nesterov_place/database/NesterovDatabase.hh
 * Contact : https://github.com/sjchanson
 */
#ifndef IPL_OPERATOR_GP_NESTEROV_DATABASE_PARAMETER_H
#define IPL_OPERATOR_GP_NESTEROV_DATABASE_PARAMETER_H

inline double netWiringDistributionMapWeight(int num_pins, int aspect_ratio, double l_ness) 
{ 
    switch (aspect_ratio) 
    { 
        case 1: 
            switch(num_pins) 
            { 
                case 1: case 2: case 3: 
                    return 1.00;        
                    break; 
                case 4: 
                    if (l_ness <= 0.25) return 1.330; 
                    if (l_ness <= 0.50) return 1.115; 
                    if (l_ness <= 0.75) return 1.050; 
                    if (l_ness <= 1.00) return 1.020; 
                    break; 
                case 5: 
                    if (l_ness <= 0.25) return 1.330; 
                    if (l_ness <= 0.50) return 1.150; 
                    if (l_ness <= 0.75) return 1.080; 
                    if (l_ness <= 1.00) return 1.040; 
                    break; 
                case 6: 
                    if (l_ness <= 0.25) return 1.355; 
                    if (l_ness <= 0.50) return 1.185; 
                    if (l_ness <= 0.75) return 1.110; 
                    if (l_ness <= 1.00) return 1.050; 
                    break; 
                case 7: 
                    if (l_ness <= 0.25) return 1.390; 
                    if (l_ness <= 0.50) return 1.220; 
                    if (l_ness <= 0.75) return 1.135; 
                    if (l_ness <= 1.00) return 1.065; 
                    break; 
                case 8: 
                    if (l_ness <= 0.25) return 1.415; 
                    if (l_ness <= 0.50) return 1.250; 
                    if (l_ness <= 0.75) return 1.160; 
                    if (l_ness <= 1.00) return 1.080; 
                    break; 
                case 9: 
                    if (l_ness <= 0.25) return 1.450; 
                    if (l_ness <= 0.50) return 1.285; 
                    if (l_ness <= 0.75) return 1.185; 
                    if (l_ness <= 1.00) return 1.090; 
                    break; 
                case 10: 
                    if (l_ness <= 0.25) return 1.490; 
                    if (l_ness <= 0.50) return 1.325; 
                    if (l_ness <= 0.75) return 1.210; 
                    if (l_ness <= 1.00) return 1.105; 
                    break; 
                case 11: 
                    if (l_ness <= 0.25) return 1.515; 
                    if (l_ness <= 0.50) return 1.355; 
                    if (l_ness <= 0.75) return 1.235; 
                    if (l_ness <= 1.00) return 1.120; 
                    break; 
                case 12: 
                    if (l_ness <= 0.25) return 1.555; 
                    if (l_ness <= 0.50) return 1.385; 
                    if (l_ness <= 0.75) return 1.260; 
                    if (l_ness <= 1.00) return 1.135; 
                    break; 
                case 13: 
                    if (l_ness <= 0.25) return 1.590; 
                    if (l_ness <= 0.50) return 1.420; 
                    if (l_ness <= 0.75) return 1.280; 
                    if (l_ness <= 1.00) return 1.145; 
                    break; 
                case 14: 
                    if (l_ness <= 0.25) return 1.620; 
                    if (l_ness <= 0.50) return 1.450; 
                    if (l_ness <= 0.75) return 1.310; 
                    if (l_ness <= 1.00) return 1.160; 
                    break; 
                case 15: 
                    if (l_ness <= 0.25) return 1.660; 
                    if (l_ness <= 0.50) return 1.485; 
                    if (l_ness <= 0.75) return 1.330; 
                    if (l_ness <= 1.00) return 1.175; 
                    break; 
                default: 
                    if (l_ness <= 0.25) return 1.660; 
                    if (l_ness <= 0.50) return 1.485; 
                    if (l_ness <= 0.75) return 1.330; 
                    if (l_ness <= 1.00) return 1.175; 
                    break; 
            } 
            break; 
        case 2: case 3:
            switch(num_pins)
            {
                case 1: case 2: case 3:
                    return 1.00;
                    break;
                case 4:
                    if (l_ness <= 0.25) return 1.240; 
                    if (l_ness <= 0.50) return 1.094; 
                    if (l_ness <= 0.75) return 1.047; 
                    if (l_ness <= 1.00) return 1.023; 
                    break; 
                case 5: 
                    if (l_ness <= 0.25) return 1.240; 
                    if (l_ness <= 0.50) return 1.127; 
                    if (l_ness <= 0.75) return 1.070; 
                    if (l_ness <= 1.00) return 1.037; 
                    break; 
                case 6: 
                    if (l_ness <= 0.25) return 1.273; 
                    if (l_ness <= 0.50) return 1.155; 
                    if (l_ness <= 0.75) return 1.103; 
                    if (l_ness <= 1.00) return 1.051; 
                    break; 
                case 7: 
                    if (l_ness <= 0.25) return 1.306; 
                    if (l_ness <= 0.50) return 1.193; 
                    if (l_ness <= 0.75) return 1.127; 
                    if (l_ness <= 1.00) return 1.065; 
                    break; 
                case 8: 
                    if (l_ness <= 0.25) return 1.348; 
                    if (l_ness <= 0.50) return 1.221; 
                    if (l_ness <= 0.75) return 1.150; 
                    if (l_ness <= 1.00) return 1.080; 
                    break; 
                case 9: 
                    if (l_ness <= 0.25) return 1.377; 
                    if (l_ness <= 0.50) return 1.259; 
                    if (l_ness <= 0.75) return 1.174; 
                    if (l_ness <= 1.00) return 1.094; 
                    break; 
                case 10:
                    if (l_ness <= 0.25) return 1.409; 
                    if (l_ness <= 0.50) return 1.287; 
                    if (l_ness <= 0.75) return 1.197; 
                    if (l_ness <= 1.00) return 1.103; 
                    break; 
                case 11:
                    if (l_ness <= 0.25) return 1.447; 
                    if (l_ness <= 0.50) return 1.315; 
                    if (l_ness <= 0.75) return 1.221; 
                    if (l_ness <= 1.00) return 1.117; 
                    break; 
                case 12:
                    if (l_ness <= 0.25) return 1.485; 
                    if (l_ness <= 0.50) return 1.344; 
                    if (l_ness <= 0.75) return 1.240; 
                    if (l_ness <= 1.00) return 1.131; 
                    break; 
                case 13: 
                    if (l_ness <= 0.25) return 1.513; 
                    if (l_ness <= 0.50) return 1.377; 
                    if (l_ness <= 0.75) return 1.263; 
                    if (l_ness <= 1.00) return 1.141; 
                    break; 
                case 14: 
                    if (l_ness <= 0.25) return 1.546; 
                    if (l_ness <= 0.50) return 1.405; 
                    if (l_ness <= 0.75) return 1.287; 
                    if (l_ness <= 1.00) return 1.150; 
                    break; 
                case 15: 
                    if (l_ness <= 0.25) return 1.579; 
                    if (l_ness <= 0.50) return 1.433; 
                    if (l_ness <= 0.75) return 1.306; 
                    if (l_ness <= 1.00) return 1.169; 
                    break; 
                default: 
                    if (l_ness <= 0.25) return 1.579; 
                    if (l_ness <= 0.50) return 1.433; 
                    if (l_ness <= 0.75) return 1.306; 
                    if (l_ness <= 1.00) return 1.169; 
                    break; 
            }
            break; 
        case 4: 
            switch(num_pins) 
            {
                case 1: case 2: case 3:
                    return 1.00;
                    break;
                case 4:
                    if (l_ness <= 0.25) return 1.144; 
                    if (l_ness <= 0.50) return 1.064; 
                    if (l_ness <= 0.75) return 1.032; 
                    if (l_ness <= 1.00) return 1.016; 
                    break; 
                case 5:
                    if (l_ness <= 0.25) return 1.144; 
                    if (l_ness <= 0.50) return 1.084; 
                    if (l_ness <= 0.75) return 1.052; 
                    if (l_ness <= 1.00) return 1.032; 
                    break; 
                case 6:
                    if (l_ness <= 0.25) return 1.172; 
                    if (l_ness <= 0.50) return 1.108; 
                    if (l_ness <= 0.75) return 1.076; 
                    if (l_ness <= 1.00) return 1.044; 
                    break; 
                case 7:
                    if (l_ness <= 0.25) return 1.200; 
                    if (l_ness <= 0.50) return 1.128; 
                    if (l_ness <= 0.75) return 1.092; 
                    if (l_ness <= 1.00) return 1.056; 
                    break; 
                case 8:
                    if (l_ness <= 0.25) return 1.224; 
                    if (l_ness <= 0.50) return 1.156; 
                    if (l_ness <= 0.75) return 1.116; 
                    if (l_ness <= 1.00) return 1.068; 
                    break; 
                case 9:
                    if (l_ness <= 0.25) return 1.252; 
                    if (l_ness <= 0.50) return 1.180; 
                    if (l_ness <= 0.75) return 1.132; 
                    if (l_ness <= 1.00) return 1.076; 
                    break; 
                case 10:
                    if (l_ness <= 0.25) return 1.276; 
                    if (l_ness <= 0.50) return 1.204; 
                    if (l_ness <= 0.75) return 1.152; 
                    if (l_ness <= 1.00) return 1.088; 
                    break; 
                case 11:
                    if (l_ness <= 0.25) return 1.308; 
                    if (l_ness <= 0.50) return 1.228; 
                    if (l_ness <= 0.75) return 1.164; 
                    if (l_ness <= 1.00) return 1.100; 
                    break; 
                case 12:
                    if (l_ness <= 0.25) return 1.332; 
                    if (l_ness <= 0.50) return 1.252; 
                    if (l_ness <=0.75) return 1.188; 
                    if (l_ness <= 1.00) return 1.108; 
                    break; 
                case 13:
                    if (l_ness <= 0.25) return 1.360; 
                    if (l_ness <= 0.50) return 1.276; 
                    if (l_ness <= 0.75) return 1.208; 
                    if (l_ness <= 1.00) return 1.120; 
                    break; 
                case 14:
                    if (l_ness <= 0.25) return 1.388; 
                    if (l_ness <= 0.50) return 1.300; 
                    if (l_ness <= 0.75) return 1.220; 
                    if (l_ness <= 1.00) return 1.132; 
                    break; 
                case 15:
                    if (l_ness <= 0.25) return 1.416; 
                    if (l_ness <= 0.50) return 1.316; 
                    if (l_ness <= 0.75) return 1.240; 
                    if (l_ness <= 1.00) return 1.144; 
                    break; 
                default: 
                    if (l_ness <= 0.25) return 1.416; 
                    if (l_ness <= 0.50) return 1.316; 
                    if (l_ness <= 0.75) return 1.240; 
                    if (l_ness <= 1.00) return 1.144; 
                    break; 
            } 
            break; 
        default: 
            switch(num_pins) 
            {
                case 1: case 2: case 3:
                    return 1.00;
                    break;
                case 4:
                    if (l_ness <= 0.25) return 1.144; 
                    if (l_ness <= 0.50) return 1.064; 
                    if (l_ness <= 0.75) return 1.032; 
                    if (l_ness <= 1.00) return 1.016; 
                    break; 
                case 5:
                    if (l_ness <= 0.25) return 1.144; 
                    if (l_ness <= 0.50) return 1.084; 
                    if (l_ness <= 0.75) return 1.052; 
                    if (l_ness <= 1.00) return 1.032; 
                    break; 
                case 6:
                    if (l_ness <= 0.25) return 1.172; 
                    if (l_ness <= 0.50) return 1.108; 
                    if (l_ness <= 0.75) return 1.076; 
                    if (l_ness <= 1.00) return 1.044; 
                    break; 
                case 7:
                    if (l_ness <= 0.25) return 1.200; 
                    if (l_ness <= 0.50) return 1.128; 
                    if (l_ness <= 0.75) return 1.092; 
                    if (l_ness <= 1.00) return 1.056; 
                    break; 
                case 8:
                    if (l_ness <= 0.25) return 1.224; 
                    if (l_ness <= 0.50) return 1.156; 
                    if (l_ness <= 0.75) return 1.116; 
                    if (l_ness <= 1.00) return 1.068; 
                    break; 
                case 9:
                    if (l_ness <= 0.25) return 1.252; 
                    if (l_ness <= 0.50) return 1.180; 
                    if (l_ness <= 0.75) return 1.132; 
                    if (l_ness <= 1.00) return 1.076; 
                    break; 
                case 10:
                    if (l_ness <= 0.25) return 1.276; 
                    if (l_ness <= 0.50) return 1.204; 
                    if (l_ness <= 0.75) return 1.152; 
                    if (l_ness <= 1.00) return 1.088; 
                    break; 
                case 11:
                    if (l_ness <= 0.25) return 1.308; 
                    if (l_ness <= 0.50) return 1.228; 
                    if (l_ness <= 0.75) return 1.164; 
                    if (l_ness <= 1.00) return 1.100; 
                    break; 
                case 12:
                    if (l_ness <= 0.25) return 1.332; 
                    if (l_ness <= 0.50) return 1.252; 
                    if (l_ness <=0.75) return 1.188; 
                    if (l_ness <= 1.00) return 1.108; 
                    break; 
                case 13:
                    if (l_ness <= 0.25) return 1.360; 
                    if (l_ness <= 0.50) return 1.276; 
                    if (l_ness <= 0.75) return 1.208; 
                    if (l_ness <= 1.00) return 1.120; 
                    break; 
                case 14:
                    if (l_ness <= 0.25) return 1.388; 
                    if (l_ness <= 0.50) return 1.300; 
                    if (l_ness <= 0.75) return 1.220; 
                    if (l_ness <= 1.00) return 1.132; 
                    break; 
                case 15:
                    if (l_ness <= 0.25) return 1.416; 
                    if (l_ness <= 0.50) return 1.316; 
                    if (l_ness <= 0.75) return 1.240; 
                    if (l_ness <= 1.00) return 1.144; 
                    break; 
                default: 
                    if (l_ness <= 0.25) return 1.416; 
                    if (l_ness <= 0.50) return 1.316; 
                    if (l_ness <= 0.75) return 1.240; 
                    if (l_ness <= 1.00) return 1.144; 
                    break; 
            } 
            break; 
    } 
    return 0.0;
}



#endif /* IPL_OPERATOR_GP_NESTEROV_DATABASE_PARAMETER_H */