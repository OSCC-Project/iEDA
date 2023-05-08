#pragma once

#include <string>

#include "GdsElement.hpp"
#include "GdsStrans.hpp"

namespace idb {

enum class GdsPresentation
{
  kLeft = 0b00,
  kCenterH = 0b01,
  kRight = 0b10,
  kDefaultH = kLeft,

  kBottom = 0b1000,
  kCenterV = 0b0100,
  kTop = 0b0000,
  kDefaultV = kBottom,

  kDefault = kDefaultV | kDefaultH,

  kCenter = kCenterV | kCenterH,
  kBottomLeft = kBottom | kLeft,
  kBottomRight = kBottom | kRight,
  kTopLeft = kTop | kLeft,
  kTopRight = kTop | kRight,
};

class GdsText : public GdsElemBase
{
 public:
  GdsText()
      : GdsElemBase(GdsElemType::kText),
        layer(0),
        text_type(0),
        presentation(GdsPresentation::kDefault),
        path_type(GdsPathType::kDefault),
        width(0),
        strans(),
        str()
  {
  }

  GdsText& operator=(const GdsText& rhs)
  {
    GdsElemBase::operator=(rhs);
    layer = rhs.layer;
    text_type = rhs.text_type;
    presentation = rhs.presentation;
    path_type = rhs.path_type;
    width = rhs.width;
    strans = rhs.strans;
    str = rhs.str;

    return *this;
  }

  void reset() override
  {
    reset_base();
    layer = 0;
    text_type = 0;
    presentation = GdsPresentation::kDefault;
    path_type = GdsPathType::kDefault;
    width = 0;
    strans.reset();
    str.clear();
  }

  // members
  GdsLayer layer;
  GdsTextType text_type;
  GdsPresentation presentation;
  GdsPathType path_type;
  GdsWidth width;
  GdsStrans strans;
  std::string str;
};

}  // namespace idb
