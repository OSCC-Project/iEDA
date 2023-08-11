NAME          TESTPROB
ROWS
 N  COST
 G  CON1
 L  CON2
 G  CON3
 L  CON4
 G  CON5
 G  CON6
COLUMNS
    XONE      CON1                 1   CON2                -1
    XONE      CON6                 1
    YTWO      CON1                 1   CON3                 2
    ZTHREE    COST                 1   CON2                 1
    ZTHREE    CON3                -1   CON4                 1
    ZTHREE    CON5                -1
    WFOUR     COST                 1   CON4                -1
    WFOUR     CON5                 2   CON6                 1
RHS
    LHS1      CON1                 8   CON3                -4
    LHS1      CON5                 0   CON6                10
    RHS1      CON2                 6   CON4                 5
BOUNDS
 LI BND1      XONE                 0
 LI BND1      YTWO                 9
 LO BND1      ZTHREE              -5
 UP BND1      ZTHREE               5
 LO BND1      WFOUR               -5
 UP BND1      WFOUR                5
ENDATA