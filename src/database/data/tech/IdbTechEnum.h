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
#ifndef IDB_TECH_ENUM
#define IDB_TECH_ENUM

namespace idb {

  enum class LayerTypeEnum { kUNKNOWN, kCUT, kROUTING, kMASTERSLICE };
  enum class LayerDirEnum { kUNKNOWN, kHORITIONAL, kVERTICAL };
  enum class ViaMetalLayerDirection {
    kUNKNOWN,
    kBottomVerticalTopHoritional,
    kBottomHoritionalTopVertical,
    kBottomVerticalTopVertical,
    kBottomHoritionalTopHoritional
  };
  enum class CutArrayTypeEnum { kUNKNOWN, kNORTH, kSOUTH, kWEST, kEAST, kLOWLEFT, kUPLEFT, kLOWRIGHT, kUPRIGHT };
  enum class CheckTypeEnum {
    kNONE,
    kIDBMAXWIDTHCHECK,
    kMINWIDTHCHECK,
    kMINAREACHECK,
    kIDBCORNERSPACINGCHECK,
    kIDBLEF58SPACINGEOLCHECK,
    kIDBLEF58SPACINGEOLWITHINCHECK,
    kIDBLEF58SPACINGEOLWITHINENDTOENDCHECK,
    kIDBLEF58SPACINGEOLWITHINPARALLELEDGECHECK,
    kIDBLEF58SPACINGEOLWITHINMAXMINLENGTHCHECK,
    kIDBSPACINGCHECK,
    kIDBSPACINGSAMENETCHECK,
    kIDBSPACINGEOLCHECK,
    kIDBSPACINGRANGECHECK,
    kIDBSPACINGTABLEPARALLELRUNLENGTHCHECK,
    kIDBSPACINGTABLETWOWIDTHCHECK,
    kIDBLEF58SPACINGTABLECHECK,
    kIDBLEF58RIGHTWAYONGRIDONLYCHECK,
    kIDBLEF58RECTONLYCHECK,
    kIDBLEF58MINSTEPCHECK,
    kIDBMINSTEPCHECK,
    kIDBMINIMUMCUTCHECK,
    kIDBLEF58CUTSPACINGTABLECHECK,
    kIDBLEF58CUTSPACINGTABLEPRLCHECK,
    kIDBLEF58CUTSPACINGTABLELAYERCHECK,
    kIDBCUTSPACINGCHECK,
    kIDBLEF58CUTSPACINGCHECK,
    kIDBMINENCLOSEDAREACHECK,
    kIDBENCLOSURECHECK,
    kIDBARRAYSPACINGCHECK,
    kIDBANTENNACHECK,
    kIDBDCCURRENTDENSITYCHECK,
    kIDBDENSITYCHECK
  };
  enum class CornerTypeEnum { kUNKNOWN, kCONCAVE, kCONVEX };
  enum class MinstepTypeEnum { kUNKNOWN = -1, kINSIDECORNER = 0, kOUTSIDECORNER = 1, kSTEP = 2 };
  enum class MinimumcutConnectionEnum { kDEFAULT, kFROMABOVE, kFROMBELOW };
  enum class TechShapeType { kUNKNOWN = -1, kRECT = 0, kPOLYGON = 1 };
  enum class EnclosureLayerFrom { kUNKNOWN, kABOVE, kBELOW };
  enum class ViaLayerEnum { kRoutingLayerBottom, kLayerCut, kRoutingLayerTop };
  enum class ViaCutNumEnum { kUNKOWN, k1CUTVIA, k2CUTVIA };
}  // namespace idb

#endif
