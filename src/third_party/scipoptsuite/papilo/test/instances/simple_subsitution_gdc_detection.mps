ROWS
 N  COST
 E  INF
% G  CON1
% L  CON2
COLUMNS
    X      INF                  3   COST                 1
%    X      CON1                 1   CON2                 1
    Y      INF                  8   COST                 1
%    Y      CON1                 1   CON2                 1

RHS
    RHS1   INF                 37   CON1                 5
    LHS1   CON2                 0
BOUNDS
 UI BND1      X                 5
 LI BND1      X                 0
 LI BND1      Y                 0
 UI BND1      Y                 5
ENDATA