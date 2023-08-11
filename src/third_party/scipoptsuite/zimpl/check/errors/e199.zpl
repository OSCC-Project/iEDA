# $Id: e199.zpl,v 1.6 2010/06/10 19:42:42 bzfkocht Exp $
set I   := {1 .. 5};
var x[I] binary;
sos c2: type1 priority 100 : sum <i> in I : i * x[i] + 3;
