# $Id: w215.zpl,v 1.4 2010/06/10 19:42:42 bzfkocht Exp $
set I := { 1, 2, 3 };
var x[<i> in I] integer >= 0 <= 10 startval i;
subto c1: sum <i> in I : x[i] <= 5;
subto c2: sum <i> in I : x[i] >= 7;
subto c3: sum <i> in I : x[i] == 8;
