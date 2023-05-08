/**
 * @file IdbBusBitChars.h
 * @author pengming (435788362@qq.com)
 * @brief
 * @version 0.1
 * @date 2022-09-13
 */
#pragma once

namespace idb {

/**
 * @brief
 * Bus Bit Characters:
 *  BUSBITCHARS "delimiterPair" ;
 * example:
 *  BUSBITCHARS "[]" ;
 * Used to specifies the pair of characters when DEF names are mapped to or from other dbs.
 */
class IdbBusBitChars
{
 public:
  IdbBusBitChars();

  // getter
  char getLeftDelimiter() const { return _left_delimiter; }
  char getRightDelimiter() const { return _right_delimiter; }

  // setter
  void setLeftDelimiter(char left_delimiter) { _left_delimiter = left_delimiter; }
  void setRightDelimter(char right_delimiter) { _right_delimiter = right_delimiter; }

 private:
  char _left_delimiter;
  char _right_delimiter;
};

}  // namespace idb