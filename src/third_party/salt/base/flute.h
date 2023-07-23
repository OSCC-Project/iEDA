#pragma once

#include "tree.h"

namespace salt {

class FluteBuilder {
public:
    void Run(const Net& net, Tree& saltTree);
};

}  // namespace salt