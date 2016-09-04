
#ifndef SRC_AttRecursiveGatedNN_attention1_H_
#define SRC_AttRecursiveGatedNN_attention1_H_

#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class AttRecursiveGatedNN_attention1 {
public:
  BiLayer<xpu> _reset1;
  BiLayer<xpu> _reset2;
  BiLayer<xpu> _reset3;
  BiLayer<xpu> _update1;
  BiLayer<xpu> _update2;
  BiLayer<xpu> _update3;
  BiLayer<xpu> _update_tilde;
  TriLayer<xpu> _recursive_tilde;


  Tensor<xpu, 2, dtype> nx1;
  Tensor<xpu, 2, dtype> nx2;
  Tensor<xpu, 2, dtype> nx3;
  Tensor<xpu, 2, dtype> sum;

  Tensor<xpu, 2, dtype> px1;
  Tensor<xpu, 2, dtype> px2;
  Tensor<xpu, 2, dtype> px3;
  Tensor<xpu, 2, dtype> pmy;


  Tensor<xpu, 2, dtype> lrx1;
  Tensor<xpu, 2, dtype> lrx2;
  Tensor<xpu, 2, dtype> lrx3;
  Tensor<xpu, 2, dtype> lmy;
  Tensor<xpu, 2, dtype> lux1;
  Tensor<xpu, 2, dtype> lux2;
  Tensor<xpu, 2, dtype> lux3;
  Tensor<xpu, 2, dtype> lumy;

  Tensor<xpu, 2, dtype> lnx1;
  Tensor<xpu, 2, dtype> lnx2;
  Tensor<xpu, 2, dtype> lnx3;
  Tensor<xpu, 2, dtype> lsum;

  Tensor<xpu, 2, dtype> lpx1;
  Tensor<xpu, 2, dtype> lpx2;
  Tensor<xpu, 2, dtype> lpx3;
  Tensor<xpu, 2, dtype> lpmy;


public:
  AttRecursiveGatedNN_attention1() {
  }

  inline void initial(int dimension, int attDim, int seed = 0) {
    _reset1.initial(dimension, dimension, attDim, false, seed, 1);
    _reset2.initial(dimension, dimension, attDim, false, seed + 10, 1);
    _reset3.initial(dimension, dimension, attDim, false, seed + 20, 1);
    _update1.initial(dimension, dimension, attDim, false, seed + 30, 3);
    _update2.initial(dimension, dimension, attDim, false, seed + 40, 3);
    _update3.initial(dimension, dimension, attDim, false, seed + 50, 3);
    _update_tilde.initial(dimension, dimension, attDim, false, seed + 60, 3);
    _recursive_tilde.initial(dimension, dimension, dimension, dimension, false, seed + 70, 0);

    nx1 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    nx2 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    nx3 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    sum = NewTensor<xpu>(Shape2(1, dimension), d_zero);

    px1 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    px2 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    px3 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    pmy = NewTensor<xpu>(Shape2(1, dimension), d_zero);


    lrx1 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lrx2 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lrx3 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lmy = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lux1 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lux2 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lux3 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lumy = NewTensor<xpu>(Shape2(1, dimension), d_zero);

    lnx1 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lnx2 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lnx3 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lsum = NewTensor<xpu>(Shape2(1, dimension), d_zero);

    lpx1 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lpx2 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lpx3 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lpmy = NewTensor<xpu>(Shape2(1, dimension), d_zero);
  }


  inline void release() {
    _reset1.release();
    _reset2.release();
    _reset3.release();

    _update1.release();
    _update2.release();
    _update3.release();
    _update_tilde.release();

    _recursive_tilde.release();

    FreeSpace(&nx1);
    FreeSpace(&nx2);
    FreeSpace(&nx3);
    FreeSpace(&sum);
    FreeSpace(&px1);
    FreeSpace(&px2);
    FreeSpace(&px3);
    FreeSpace(&pmy);
    FreeSpace(&lnx1);
    FreeSpace(&lnx2);
    FreeSpace(&lnx3);
    FreeSpace(&lsum);
    FreeSpace(&lpx1);
    FreeSpace(&lpx2);
    FreeSpace(&lpx3);
    FreeSpace(&lpmy);
    FreeSpace(&lrx1);
    FreeSpace(&lrx2);
    FreeSpace(&lrx3);
    FreeSpace(&lmy);
    FreeSpace(&lux1);
    FreeSpace(&lux2);
    FreeSpace(&lux3);
    FreeSpace(&lumy);
  }

  virtual ~AttRecursiveGatedNN_attention1() {
    // TODO Auto-generated destructor stub
  }



public:

  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2, Tensor<xpu, 2, dtype> x3,
		  Tensor<xpu, 2, dtype> a1,
      Tensor<xpu, 2, dtype> rx1, Tensor<xpu, 2, dtype> rx2, Tensor<xpu, 2, dtype> rx3, Tensor<xpu, 2, dtype> my,
      Tensor<xpu, 2, dtype> ux1, Tensor<xpu, 2, dtype> ux2, Tensor<xpu, 2, dtype> ux3, Tensor<xpu, 2, dtype> umy,
      Tensor<xpu, 2, dtype> y) {

    nx1 = 0.0;
    nx2 = 0.0;
    nx3 = 0.0;
    sum = 0.0;

    px1 = 0.0;
    px2 = 0.0;
    px3 = 0.0;
    pmy = 0.0;

    _reset1.ComputeForwardScore(x1, a1, rx1);
    _reset2.ComputeForwardScore(x2, a1, rx2);
    _reset3.ComputeForwardScore(x3, a1, rx3);


    nx1 = rx1 * x1;
    nx2 = rx2 * x2;
    nx2 = rx3 * x3;

    _recursive_tilde.ComputeForwardScore(nx1, nx2, nx3, my);


    _update1.ComputeForwardScore(x1, a1, ux1);
    _update2.ComputeForwardScore(x2, a1, ux2);
    _update3.ComputeForwardScore(x3, a1, ux3);
    _update_tilde.ComputeForwardScore(my, a1, umy);

    sum = ux1 + ux2 + ux3 + umy;

    px1 = ux1 / sum;
    px2 = ux2 / sum;
    px3 = ux3 / sum;
    pmy = umy / sum;

    y = px1 * x1 + px2 * x2 + px3 * x3 + pmy * my;

  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2, Tensor<xpu, 2, dtype> x3,
		  Tensor<xpu, 2, dtype> a1,
      Tensor<xpu, 2, dtype> rx1, Tensor<xpu, 2, dtype> rx2, Tensor<xpu, 2, dtype> rx3, Tensor<xpu, 2, dtype> my,
      Tensor<xpu, 2, dtype> ux1, Tensor<xpu, 2, dtype> ux2, Tensor<xpu, 2, dtype> ux3, Tensor<xpu, 2, dtype> umy,
      Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly,
      Tensor<xpu, 2, dtype> lx1, Tensor<xpu, 2, dtype> lx2, Tensor<xpu, 2, dtype> lx3,
	  	  Tensor<xpu, 2, dtype> la1,
      bool bclear = false) {
    if (bclear){
      lx1 = 0.0; lx2 = 0.0; lx3 = 0.0; la1 = 0.0;
    }

    nx1 = 0.0;
    nx2 = 0.0;
    nx3 = 0.0;
    sum = 0.0;

    px1 = 0.0;
    px2 = 0.0;
    px3 = 0.0;
    pmy = 0.0;


    lrx1 = 0.0;
    lrx2 = 0.0;
    lrx3 = 0.0;
    lmy = 0.0;
    lux1 = 0.0;
    lux2 = 0.0;
    lux3 = 0.0;
    lumy = 0.0;

    lnx1 = 0.0;
    lnx2 = 0.0;
    lnx3 = 0.0;
    lsum = 0.0;

    lpx1 = 0.0;
    lpx2 = 0.0;
    lpx3 = 0.0;
    lpmy = 0.0;

    nx1 = rx1 * x1;
    nx2 = rx2 * x2;
    nx3 = rx3 * x3;

    sum = ux1 + ux2 + ux3 + umy;

    px1 = ux1 / sum;
    px2 = ux2 / sum;
    px3 = ux3 / sum;
    pmy = umy / sum;


    lpx1 += ly * x1;
    lx1 += ly * px1;

    lpx2 += ly * x2;
    lx2 += ly * px2;

    lpx3 += ly * x3;
    lx3 += ly * px3;

    lpmy += ly * my;
    lmy += ly * pmy;



    lux1 += lpx1 / sum;
    lux2 += lpx2 / sum;
    lux3 += lpx3 / sum;
    lumy += lpmy / sum;

    lsum -= lpx1 * px1 / sum;
    lsum -= lpx2 * px2 / sum;
    lsum -= lpx3 * px3 / sum;
    lsum -= lpmy * pmy / sum;


    lux1 += lsum;
    lux2 += lsum;
    lux3 += lsum;
    lumy += lsum;

    _update1.ComputeBackwardLoss(x1, a1, ux1, lux1, lx1, la1);
    _update2.ComputeBackwardLoss(x2, a1, ux2, lux2, lx2, la1);
    _update3.ComputeBackwardLoss(x3, a1, ux3, lux3, lx3, la1);
    _update_tilde.ComputeBackwardLoss(my, a1, umy, lumy, lmy, la1);

    _recursive_tilde.ComputeBackwardLoss(nx1, nx2, nx3, my, lmy, lnx1, lnx2, lnx3);

    lrx1 += lnx1 * x1;
    lx1 += lnx1 * rx1;

    lrx2 += lnx2 * x2;
    lx2 += lnx2 * rx2;

    lrx3 += lnx3 * x3;
    lx3 += lnx3 * rx3;

    _reset1.ComputeBackwardLoss(x1, a1, rx1, lrx1, lx1, la1);
    _reset2.ComputeBackwardLoss(x2, a1, rx2, lrx2, lx2, la1);
    _reset3.ComputeBackwardLoss(x3, a1, rx3, lrx3, lx3, la1);
  }


  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _reset1.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _reset2.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _reset3.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

    _update1.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _update2.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _update3.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _update_tilde.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

    _recursive_tilde.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }



};



#endif /* SRC_AttRecursiveGatedNN_H_ */
