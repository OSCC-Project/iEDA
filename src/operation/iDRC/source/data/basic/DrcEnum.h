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
#ifndef IDRC_SRC_DB_DRCTYPE_H_
#define IDRC_SRC_DB_DRCTYPE_H_

namespace idrc {

enum class QueryBoxDir
{
  kNone = 0,
  kUp = 1,
  kDown = 2,
  kLeft = 3,
  kRight = 4,
  kNE = 5,
  kNW = 6,
  kSE = 7,
  kSW = 8
};

enum class EdgeDirection
{
  kNone = 0,
  kNorth = 1,
  kEast = 2,
  kSouth = 3,
  kWest = 4,
};

enum class CornerDirEnum
{
  kNone = 0,
  kNE,
  kSE,
  kSW,
  kNW
};

enum class RuleTypeEnum
{  // check FlexDR.h fixMode
  kShortRule = 0,
  kAreaRule = 1,
  kWidthRule = 2,
  kMaxWidthRule,
  kSpacingRule,
  kMinEnclosedAreaRule,
  kDensityRule,
  kCutSpacingRule,
  kSpacingTablePrlRule,
  kMinimumCutRule,
  kEnclosureRule,

  kLEF58VoltageSpacingRule,
  kLEF58CutSpacingRule,
  kLEF58SpacingTableCutRule,
  kLEF58CutClassRule,
  kLEF58SquaEnclosureRule,
  kLEF58RectEnclosureRule,
  kLEF58EnclosureEdgeRule,
  kLEF58SpacingTableJogRule,
  kLEF58MinStepRule,
  kLEF58MinimumCutRule,
  kLEF58AreaRule,
  kLEF58SpacingEOLRule
};

enum class ViolationType
{
  kNone = 0,
  kRoutingWidth = 1,
  kRoutingArea = 2,
  kRoutingSpacing = 3,
  kEnclosedArea = 4,
  kShort = 5,
  kDensity = 6,
  kEnclosure = 7,
  kCutSpacing = 8,
  kEnd2EndEOLSpacing = 9,
  kEOLSpacing = 10,
  kCutEOLSpacing = 11,
  kCutDiffLayerSpacing = 12,
  kEnclosureEdge = 13,
  kCornerFillingSpacing = 14,
  kJogSpacing = 15,
  kNotchSpacing = 16,
  kMinStep = 17,
  kCutShort,
};

enum class LayerType
{
  kNone = 0,
  kRouting = 1,
  kCut = 2
};

enum class LayerDirection
{
  kNone = 0,
  kHorizontal = 1,
  kVertical = 2
};

enum class ScopeType
{
  kNone = 0,
  kMax,
  kMin,
  EOL,
  CornerFill,
  Common
};

enum class RectOwnerType
{
  kNone = 0,
  kRoutingMetal,
  kSegment,
  kViaMetal,
  kViaCut,
  kPin,
  kObs,
  kBlockage,
  kSpotMark,
  kCommonRegion,
  kEOLRegion,
  kCornerFillRegion,
  kCutCommonRegion
};

enum class DrcDirection
{
  kNone = 0,
  kHorizontal = 1,
  kVertical = 2,
  kOblique = 3
};

enum class MinimumCutDirEnum
{
  Default = 0,
  FromBelow = 1,
  FromAbove = 2
};

enum class EnclosureDirEnum
{
  Default = 0,
  Below = 1,
  Above = 2
};

}  // namespace idrc

#endif