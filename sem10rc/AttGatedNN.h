/*
 * AttRecursiveGatedNN.h
 *  Gated Recursive Neural network structure with attention technique.
 *  Created on: Nov 5, 2015
 *      Author: mszhang
 */

#ifndef SRC_AttGatedNN_H_
#define SRC_AttGatedNN_H_

#include "N3L.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class AttGatedNN {
public:
  BiLayer<xpu> _reset_left;
  BiLayer<xpu> _reset_right;
  BiLayer<xpu> _recursive_tilde;


  Tensor<xpu, 2, dtype> nxl;
  Tensor<xpu, 2, dtype> nxr;


  Tensor<xpu, 2, dtype> lrxl;
  Tensor<xpu, 2, dtype> lrxr;

  Tensor<xpu, 2, dtype> lnxl;
  Tensor<xpu, 2, dtype> lnxr;



public:
  AttGatedNN() {
  }

  virtual ~AttGatedNN() {
    // TODO Auto-generated destructor stub
  }

  inline void initial(int dimension, int attDim, int seed = 0) {
    _reset_left.initial(dimension, dimension, attDim, false, seed, 1);
    _reset_right.initial(dimension, dimension, attDim, false, seed + 10, 1);
    _recursive_tilde.initial(dimension, dimension, dimension, false, seed + 50, 0);

    nxl = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    nxr = NewTensor<xpu>(Shape2(1, dimension), d_zero);


    lrxl = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lrxr = NewTensor<xpu>(Shape2(1, dimension), d_zero);

    lnxl = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lnxr = NewTensor<xpu>(Shape2(1, dimension), d_zero);
  }


  inline void release() {
    _reset_left.release();
    _reset_right.release();

    _recursive_tilde.release();

    FreeSpace(&nxl);
    FreeSpace(&nxr);

    FreeSpace(&lnxl);
    FreeSpace(&lnxr);

    FreeSpace(&lrxl);
    FreeSpace(&lrxr);
  }



  inline dtype squarenormAll() {
    dtype norm = _reset_left.squarenormAll();
    norm += _reset_right.squarenormAll();
    norm += _recursive_tilde.squarenormAll();

    return norm;
  }

  inline void scaleGrad(dtype scale) {
    _reset_left.scaleGrad(scale);
    _reset_right.scaleGrad(scale);

    _recursive_tilde.scaleGrad(scale);
  }

public:

  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> xl, Tensor<xpu, 2, dtype> xr, Tensor<xpu, 2, dtype> a,
      Tensor<xpu, 2, dtype> rxl, Tensor<xpu, 2, dtype> rxr, Tensor<xpu, 2, dtype> y
      ) {

    nxl = 0.0;
    nxr = 0.0;

    _reset_left.ComputeForwardScore(xl, a, rxl);
    _reset_right.ComputeForwardScore(xr, a, rxr);


    nxl = rxl * xl;
    nxr = rxr * xr;

    _recursive_tilde.ComputeForwardScore(nxl, nxr, y);

  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> xl, Tensor<xpu, 2, dtype> xr, Tensor<xpu, 2, dtype> a,
      Tensor<xpu, 2, dtype> rxl, Tensor<xpu, 2, dtype> rxr, Tensor<xpu, 2, dtype> y,
	  Tensor<xpu, 2, dtype> ly,
      Tensor<xpu, 2, dtype> lxl, Tensor<xpu, 2, dtype> lxr, Tensor<xpu, 2, dtype> la,
      bool bclear = false) {
    if (bclear){
      lxl = 0.0; lxr = 0.0; la = 0.0;
    }

    nxl = 0.0;
    nxr = 0.0;

    lrxl = 0.0;
    lrxr = 0.0;

    lnxl = 0.0;
    lnxr = 0.0;

    nxl = rxl * xl;
    nxr = rxr * xr;

    _recursive_tilde.ComputeBackwardLoss(nxl, nxr, y, ly, lnxl, lnxr);

    lrxl += lnxl * xl;
    lxl += lnxl * rxl;

    lrxr += lnxr * xr;
    lxr += lnxr * rxr;

    _reset_left.ComputeBackwardLoss(xl, a, rxl, lrxl, lxl, la);
    _reset_right.ComputeBackwardLoss(xr, a, rxr, lrxr, lxr, la);

  }


  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _reset_left.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _reset_right.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

    _recursive_tilde.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }

};



#endif /* SRC_AttRecursiveGatedNN_H_ */
