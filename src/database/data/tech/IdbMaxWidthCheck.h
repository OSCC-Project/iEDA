#ifndef _IDB_MAX_WIDTH_CHECK_H
#define _IDB_MAX_WIDTH_CHECK_H

namespace idb {
  class IdbMaxWidthCheck {
   public:
    IdbMaxWidthCheck() { }
    explicit IdbMaxWidthCheck(int maxWidth) : _max_width(maxWidth) { }
    ~IdbMaxWidthCheck() { }
    // getter
    int get_max_width() { return _max_width; }
    // setter
    void set_max_width(int maxWidth) { _max_width = maxWidth; }

   private:
    int _max_width;
  };
}  // namespace idb

#endif
