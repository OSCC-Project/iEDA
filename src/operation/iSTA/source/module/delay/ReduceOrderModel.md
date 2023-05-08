#<font face="宋体" size=6>Krylov subspace</font>
<font face="Times" size=4>For $n*n$ matrix $A$ and $a$ vector $b$, the Krylov subspace $K_q(A,b)$ is defined as
$$
K_q(A,b)=span\{ b,Ab,A^2b,...,A^qb \},
$$
$q$ is a given positive integer.</font>
#<font face="宋体" size=6>Arnoldi Algorithm</font>
<font face="Times" size=4>The Arnoldi method is the classic method to find a set of orthonormal vectors as a basis for a given Krylov subspace. For Krylov subspace $K_q(A,b)$,the Arnoldi method using the modified Gram-Schmidt orthogonalization is as follows, for more details,you can refer to : Tan, Sheldon, and LeiHe.  Advanced Model Order Reduction Techniques in VLSI Design.2007.</font>
$
intputs:\\
\quad \quad \quad \quad A:n*n\quad matrix \\
\quad \quad \quad \quad {b: initial\quad vector} \\
output:\\
\quad \quad \quad \quad v:orthogonal\quad basis\\
\quad \quad \quad \quad  H:Hessenberg\quad matrix\\
$

| $ARNOLDI\quad ALGORITHM(A,b)$                                                                                                                                                                                                                                                                                                                                                                       |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 $\quad Compute$ $v_1=\frac {b}{\|b\|_2}$ <br>2$\quad$ $For\quad $$j=$1,...$q\quad Do$<br>3 $\quad$ $Compute\quad$$w_j=A*v_j$<br>4$\quad$ $For\quad $$i=$1,...$j\quad$$ Do$<br>5$\quad h_{ij}=v_i^T*w_j$<br>6$\quad$$w_j=w_j-h_{ij}*v_i$<br>7$\quad$$EndDo$<br>8$\quad If\quad w_i=0,break,return$<br>9$\quad$ $else\quad h_{j+1,j}=\|w_j\|_2,$$v_{j+1}=\frac {w_j}{h_{j+1,j}}$<br>10$\quad EndDo$ |



