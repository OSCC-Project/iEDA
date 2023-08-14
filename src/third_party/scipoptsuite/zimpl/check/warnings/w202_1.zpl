# $Id: w202_1.zpl,v 1.4 2010/06/10 19:42:42 bzfkocht Exp $
set E := { 1..3 } * { 4..5 };
set X := { <i,j> in E with i > 10: <i + j, 6> };
