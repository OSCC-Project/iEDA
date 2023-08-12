# $Id: w200.zpl,v 1.5 2010/06/10 19:42:42 bzfkocht Exp $
var x[{1 .. 2}] binary;
sos c1: type1 priority 100 : 5 * x[1] + 5 * x[2];
