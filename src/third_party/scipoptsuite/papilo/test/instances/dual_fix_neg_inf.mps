NAME          TESTPROB
ROWS
 N  COST
 L  CON1
 L  CON2
 G  CON3
COLUMNS
    XONE      CON1                 1   CON2                 1
    XONE      CON3                -2
    YTWO      COST                 1   CON1                 1
    YTWO      CON3                 2
    ZTHREE    COST                 1   CON2                 1
    ZTHREE    CON3                 1
RHS
    RHS1      CON1                 8   CON2                -4
    LHS1      CON3                 6
BOUNDS
 UI BND1      XONE                 4
 MI BND1      XONE                 1
 LO BND1      YTWO                -5
 UP BND1      YTWO                 5
 LO BND1      ZTHREE              -5
 UP BND1      ZTHREE               5
ENDATA