//
// The following FFT library came from
// http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
//
//
/// 1D FFT ////////////////////////////////////////////////////////////////
void cdft(int n, int isgn, float* a, int* ip, float* w);
void ddct(int n, int isgn, float* a, int* ip, float* w);
void ddst(int n, int isgn, float* a, int* ip, float* w);

/// 2D FFT ////////////////////////////////////////////////////////////////
void cdft2d(int, int, int, float**, float*, int*, float*);
void rdft2d(int, int, int, float**, float*, int*, float*);
void rdft2dsort(int n1, int n2, int isgn, float** a);
void ddct2d(int, int, int, float**, float*, int*, float*);
void ddst2d(int, int, int, float**, float*, int*, float*);
void ddsct2d(int n1, int n2, int isgn, float** a, float* t, int* ip, float* w);
void ddcst2d(int n1, int n2, int isgn, float** a, float* t, int* ip, float* w);

/// 3D FFT ////////////////////////////////////////////////////////////////
void cdft3d(int, int, int, int, float***, float*, int*, float*);
void rdft3d(int, int, int, int, float***, float*, int*, float*);
void ddct3d(int, int, int, int, float***, float*, int*, float*);
void ddst3d(int, int, int, int, float***, float*, int*, float*);
void ddscct3d(int, int, int, int isgn, float***, float*, int*, float*);
void ddcsct3d(int, int, int, int isgn, float***, float*, int*, float*);
void ddccst3d(int, int, int, int isgn, float***, float*, int*, float*);