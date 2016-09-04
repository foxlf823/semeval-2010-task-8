/*
 * AttentionPooling.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_AttentionPooling1_H_
#define SRC_AttentionPooling1_H_
#include "tensor.h"

#include "BiLayer.h"
#include "MyLib.h"
#include "Utiltensor.h"
#include "Pooling.h"
#include "UniLayer.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// similar with AttentionPooling but can pooling on specified words
template<typename xpu>
class AttentionPooling1 {

public:
  BiLayer<xpu> _bi_gates;
  UniLayer<xpu> _uni_gates;

public:
  AttentionPooling1() {
  }

  inline void initial(int hiddenSize, int attentionSize, bool bUseB = true, int seed = 0) {
    _bi_gates.initial(hiddenSize, hiddenSize, attentionSize, bUseB, seed);
    _uni_gates.initial(hiddenSize, hiddenSize, false, seed + 10, 3);
  }


  inline void release() {
    _bi_gates.release();
    _uni_gates.release();
  }


public:
  // xExp, xSumIndex, xSum ad xPoolIndex are temporal variables, which reduce computation in back-propagation
  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> xAtt,
      Tensor<xpu, 3, dtype> xMExp, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y,
	  const hash_set<int>& indexes) {
    y = 0.0;

    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "AttentionPooling Forward error: dim invalid" << std::endl;
    }

    static hash_set<int>::iterator it;

    for (it = indexes.begin(); it != indexes.end(); ++it)
    {
    	_bi_gates.ComputeForwardScore(x[*it], xAtt[*it], xMExp[*it]);
    	_uni_gates.ComputeForwardScore(xMExp[*it], xExp[*it]);
    	xSum = xSum + xExp[*it];
    }

    for (it = indexes.begin(); it != indexes.end(); ++it) {
      xPoolIndex[*it] = xExp[*it] / xSum;
    }
    for (it = indexes.begin(); it != indexes.end(); ++it) {
      y += x[*it] * xPoolIndex[*it];
    }

  }

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 2, dtype> xAtt,
      Tensor<xpu, 3, dtype> xMExp, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y,
	  const hash_set<int>& indexes) {
    y = 0.0;

    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "AttentionPooling Forward error: dim invalid" << std::endl;
    }

    static hash_set<int>::iterator it;

    for (it = indexes.begin(); it != indexes.end(); ++it)
    {
    	_bi_gates.ComputeForwardScore(x[*it], xAtt, xMExp[*it]);
    	_uni_gates.ComputeForwardScore(xMExp[*it], xExp[*it]);
    	xSum = xSum + xExp[*it];
    }

    for (it = indexes.begin(); it != indexes.end(); ++it) {
      xPoolIndex[*it] = xExp[*it] / xSum;
    }
    for (it = indexes.begin(); it != indexes.end(); ++it) {
      y += x[*it] * xPoolIndex[*it];
    }

  }

  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> xAtt,
      Tensor<xpu, 3, dtype> xMExp, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y,
	  const hash_set<int>& indexes,
      Tensor<xpu, 2, dtype> ly, Tensor<xpu, 3, dtype> lx, Tensor<xpu, 3, dtype> lxAtt, bool bclear = false) {
    int seq_size = x.size(0);
    if(seq_size == 0) return;
    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if(bclear) lx = 0.0;
    if(bclear) lxAtt = 0.0;

    Tensor<xpu, 3, dtype> xMExpLoss = NewTensor<xpu>(Shape3(seq_size, dim1, dim2), d_zero);
    Tensor<xpu, 3, dtype> xExpLoss = NewTensor<xpu>(Shape3(seq_size, dim1, dim2), d_zero);
    Tensor<xpu, 2, dtype> xSumLoss = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
    Tensor<xpu, 3, dtype> xPoolIndexLoss = NewTensor<xpu>(Shape3(seq_size, dim1, dim2), d_zero);
    static hash_set<int>::iterator it;

    for (it = indexes.begin(); it != indexes.end(); ++it) {
      xPoolIndexLoss[*it] = ly * x[*it];
      lx[*it] += ly * xPoolIndex[*it];
    }

    for (it = indexes.begin(); it != indexes.end(); ++it) {
      xExpLoss[*it] += xPoolIndexLoss[*it] / xSum;
      xSumLoss -= xPoolIndexLoss[*it] * xExp[*it] / xSum / xSum;
    }

    for (it = indexes.begin(); it != indexes.end(); ++it)
    {
    	xExpLoss[*it] += xSumLoss;
        _uni_gates.ComputeBackwardLoss(xMExp[*it], xExp[*it], xExpLoss[*it], xMExpLoss[*it]);
        _bi_gates.ComputeBackwardLoss(x[*it], xAtt[*it], xMExp[*it], xMExpLoss[*it], lx[*it], lxAtt[*it]);

    }

    FreeSpace(&xMExpLoss);
    FreeSpace(&xExpLoss);
    FreeSpace(&xSumLoss);
    FreeSpace(&xPoolIndexLoss);
  }

  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 2, dtype> xAtt,
      Tensor<xpu, 3, dtype> xMExp, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y,
	  const hash_set<int>& indexes,
      Tensor<xpu, 2, dtype> ly, Tensor<xpu, 3, dtype> lx, Tensor<xpu, 2, dtype> lxAtt, bool bclear = false) {
    int seq_size = x.size(0);
    if(seq_size == 0) return;
    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if(bclear) lx = 0.0;
    if(bclear) lxAtt = 0.0;

    Tensor<xpu, 3, dtype> xMExpLoss = NewTensor<xpu>(Shape3(seq_size, dim1, dim2), d_zero);
    Tensor<xpu, 3, dtype> xExpLoss = NewTensor<xpu>(Shape3(seq_size, dim1, dim2), d_zero);
    Tensor<xpu, 2, dtype> xSumLoss = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
    Tensor<xpu, 3, dtype> xPoolIndexLoss = NewTensor<xpu>(Shape3(seq_size, dim1, dim2), d_zero);
    static hash_set<int>::iterator it;

    for (it = indexes.begin(); it != indexes.end(); ++it) {
      xPoolIndexLoss[*it] = ly * x[*it];
      lx[*it] += ly * xPoolIndex[*it];
    }

    for (it = indexes.begin(); it != indexes.end(); ++it) {
      xExpLoss[*it] += xPoolIndexLoss[*it] / xSum;
      xSumLoss -= xPoolIndexLoss[*it] * xExp[*it] / xSum / xSum;
    }

    for (it = indexes.begin(); it != indexes.end(); ++it)
    {
    	xExpLoss[*it] += xSumLoss;
        _uni_gates.ComputeBackwardLoss(xMExp[*it], xExp[*it], xExpLoss[*it], xMExpLoss[*it]);
        _bi_gates.ComputeBackwardLoss(x[*it], xAtt, xMExp[*it], xMExpLoss[*it], lx[*it], lxAtt);

    }

    FreeSpace(&xMExpLoss);
    FreeSpace(&xExpLoss);
    FreeSpace(&xSumLoss);
    FreeSpace(&xPoolIndexLoss);
  }


  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _bi_gates.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _uni_gates.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }


};

#endif /* SRC_AttentionPooling_H_ */
