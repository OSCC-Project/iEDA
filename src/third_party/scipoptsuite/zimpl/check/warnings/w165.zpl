# $Id: w165.zpl,v 1.4 2010/06/10 19:42:42 bzfkocht Exp $
set A:={1,2};
set B:={<1,1>,<2,2>};
param x := if A == B then 1 else 2 end;