# $Id: print.zpl,v 1.8 2011/07/31 15:10:45 bzfkocht Exp $
# 2 "<print.zpl>"
set A := { 1 .. 5 };
set B := { "a", "b", "c" };
param m := 6;
#
var x[<a> in A] >= if a < 3 then -infinity else 6 end <= if a mod 2 == 1 then infinity else 20 end;
var y binary;
var z integer >= -5 <= 7;
var w implicit integer >= 8;
#
do check card(A) > card(B);
do print "Test: ", 13 mod 7 div 2;
do print A*B;
do forall <a,b> in A*B do print <a,b>;
do print x;
do print y;
do print z;
do print w;
do print unknown;
do print m > 7;
do print 12 == 2*m;

