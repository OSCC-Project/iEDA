module simple (
inp1,
inp2,
tau2015_clk,
out
);

input inp1 ;
input inp2 ;
input tau2015_clk ;

output out ;

wire n1 ;
wire n2 ;
wire n3 ;
wire n4 ;
wire inp1 ;
wire inp2 ;
wire tau2015_clk ;
wire out ;

NAND2X1 u1 ( .A(inp1 ), .B(inp2 ), .Y(n1 ) );
DFFNEGX1 f1 ( .D(n2 ), .CLK(tau2015_clk ), .Q(n3 ) );
INVX1 u2 ( .A(n3 ), .Y(n4 ) );
INVX2 u3 ( .A(n4 ), .Y(out ) );
NOR2X1 u4 ( .A(n1 ), .B(n3 ), .Y(n2 ) );

endmodule
