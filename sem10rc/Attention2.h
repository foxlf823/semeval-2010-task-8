

#ifndef SRC_Attention_H_
#define SRC_Attention_H_

#include "N3L.h"


using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class Attention2 {

public:
  TriLayer<xpu> _tri_gates;
  UniLayer<xpu> _uni_gates;

public:
  Attention2() {
  }

/*  virtual ~Attention() {
    // TODO Auto-generated destructor stub
  }*/

  /*
   * input and output should both be nOSize, and two attention basis should be nAttSize1 and nAttSize2
   */
  inline void initial(int nOSize, int nAttSize1, int nAttSize2, bool bUseB = true, int seed = 0) {

	  _tri_gates.initial(nOSize, nOSize, nAttSize1, nAttSize2, bUseB, seed);
	  _uni_gates.initial(nOSize, nOSize, false, seed + 10, 3);
  }


  inline void release() {
	  _tri_gates.release();
	  _uni_gates.release();
  }



public:


  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> >& x,
		  Tensor<xpu, 2, dtype> xFormer, Tensor<xpu, 2, dtype> xLatter,
      std::vector<Tensor<xpu, 2, dtype> >& xMExp, std::vector<Tensor<xpu, 2, dtype> >& xExp, Tensor<xpu, 2, dtype> xSum,
      std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, Tensor<xpu, 2, dtype> y) {
    y = 0.0;
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y.size(0), odim2 = y.size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "Attention Forward error: dim invalid" << std::endl;
    }

    for (int idx = 0; idx < seq_size; idx++) {
    	_tri_gates.ComputeForwardScore(x[idx], xFormer, xLatter, xMExp[idx]);
    }
    _uni_gates.ComputeForwardScore(xMExp, xExp);

    sumpool_forward(xExp, xSum);
    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndex[idx] = xExp[idx] / xSum;
    }
    for (int idx = 0; idx < seq_size; idx++) {
      y += x[idx] * xPoolIndex[idx];
    }
  }

  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> >& x,
		  Tensor<xpu, 2, dtype> xFormer, Tensor<xpu, 2, dtype> xLatter,
      std::vector<Tensor<xpu, 2, dtype> >& xMExp, std::vector<Tensor<xpu, 2, dtype> >& xExp, Tensor<xpu, 2, dtype> xSum,
      std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, std::vector<Tensor<xpu, 2, dtype> >& y) {

    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y[0].size(0), odim2 = y[0].size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "Attention Forward error: dim invalid" << std::endl;
    }

    for(int idx=0; idx<seq_size; idx++)
    	y[idx] = 0.0;

    for (int idx = 0; idx < seq_size; idx++) {
    	_tri_gates.ComputeForwardScore(x[idx], xFormer, xLatter, xMExp[idx]);
    }
    _uni_gates.ComputeForwardScore(xMExp, xExp);

    sumpool_forward(xExp, xSum);
    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndex[idx] = xExp[idx] / xSum;
    }
    for (int idx = 0; idx < seq_size; idx++) {
      y[idx] += x[idx] * xPoolIndex[idx];
    }
  }

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 2, dtype> xAtt1, Tensor<xpu, 2, dtype> xAtt2,
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
    	_tri_gates.ComputeForwardScore(x[*it], xAtt1, xAtt2, xMExp[*it]);
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


  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> >& x,
		  Tensor<xpu, 2, dtype> xFormer, Tensor<xpu, 2, dtype> xLatter,
      std::vector<Tensor<xpu, 2, dtype> >& xMExp, std::vector<Tensor<xpu, 2, dtype> >& xExp,
      Tensor<xpu, 2, dtype> xSum, std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, Tensor<xpu, 2, dtype> y,
      Tensor<xpu, 2, dtype> ly, std::vector<Tensor<xpu, 2, dtype> >& lx,
	  Tensor<xpu, 2, dtype> lxFormer, Tensor<xpu, 2, dtype> lxLatter,
	  bool bclear = false) {
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y.size(0), odim2 = y.size(1);


    if(bclear){
      for (int idx = 0; idx < seq_size; idx++) {
        lx[idx] = 0.0;
      }
      lxFormer = 0.0;
      lxLatter = 0.0;
    }

    vector<Tensor<xpu, 2, dtype> > xMExpLoss(seq_size), xExpLoss(seq_size), xPoolIndexLoss(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      xMExpLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
      xExpLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
      xPoolIndexLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
    }

    Tensor<xpu, 2, dtype> xSumLoss = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);

    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndexLoss[idx] = ly * x[idx];
      lx[idx] += ly * xPoolIndex[idx];
    }

    for (int idx = 0; idx < seq_size; idx++) {
      xExpLoss[idx] += xPoolIndexLoss[idx] / xSum;
      xSumLoss -= xPoolIndexLoss[idx] * xExp[idx] / xSum / xSum;
    }

    sumpool_backward(xSumLoss, xExpLoss);

    _uni_gates.ComputeBackwardLoss(xMExp, xExp, xExpLoss, xMExpLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      _tri_gates.ComputeBackwardLoss(x[idx], xFormer, xLatter, xMExp[idx],
    		  xMExpLoss[idx],
			  lx[idx], lxFormer, lxLatter);
    }

    FreeSpace(&xSumLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(xExpLoss[idx]));
      FreeSpace(&(xPoolIndexLoss[idx]));
    }
  }

  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> >& x,
		  Tensor<xpu, 2, dtype> xFormer, Tensor<xpu, 2, dtype> xLatter,
      std::vector<Tensor<xpu, 2, dtype> >& xMExp, std::vector<Tensor<xpu, 2, dtype> >& xExp,
      Tensor<xpu, 2, dtype> xSum, std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, std::vector<Tensor<xpu, 2, dtype> >& y,
	  std::vector<Tensor<xpu, 2, dtype> >& ly, std::vector<Tensor<xpu, 2, dtype> >& lx,
	  Tensor<xpu, 2, dtype> lxFormer, Tensor<xpu, 2, dtype> lxLatter,
	  bool bclear = false) {
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y[0].size(0), odim2 = y[0].size(1);


    if(bclear){
      for (int idx = 0; idx < seq_size; idx++) {
        lx[idx] = 0.0;
      }
      lxFormer = 0.0;
      lxLatter = 0.0;
    }

    vector<Tensor<xpu, 2, dtype> > xMExpLoss(seq_size), xExpLoss(seq_size), xPoolIndexLoss(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      xMExpLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
      xExpLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
      xPoolIndexLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
    }

    Tensor<xpu, 2, dtype> xSumLoss = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);

    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndexLoss[idx] = ly[idx] * x[idx];
      lx[idx] += ly[idx] * xPoolIndex[idx];
    }

    for (int idx = 0; idx < seq_size; idx++) {
      xExpLoss[idx] += xPoolIndexLoss[idx] / xSum;
      xSumLoss -= xPoolIndexLoss[idx] * xExp[idx] / xSum / xSum;
    }

    sumpool_backward(xSumLoss, xExpLoss);

    _uni_gates.ComputeBackwardLoss(xMExp, xExp, xExpLoss, xMExpLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      _tri_gates.ComputeBackwardLoss(x[idx], xFormer, xLatter, xMExp[idx],
    		  xMExpLoss[idx],
			  lx[idx], lxFormer, lxLatter);
    }

    FreeSpace(&xSumLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(xExpLoss[idx]));
      FreeSpace(&(xPoolIndexLoss[idx]));
    }
  }

  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 2, dtype> xAtt1, Tensor<xpu, 2, dtype> xAtt2,
      Tensor<xpu, 3, dtype> xMExp, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y,
	  const hash_set<int>& indexes,
      Tensor<xpu, 2, dtype> ly, Tensor<xpu, 3, dtype> lx, Tensor<xpu, 2, dtype> lxAtt1, Tensor<xpu, 2, dtype> lxAtt2, bool bclear = false) {
    int seq_size = x.size(0);
    if(seq_size == 0) return;
    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if(bclear){
    	lx = 0.0;
    	lxAtt1 = 0.0;
    	lxAtt2 = 0.0;
    }

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
        _tri_gates.ComputeBackwardLoss(x[*it], xAtt1, xAtt2, xMExp[*it], xMExpLoss[*it], lx[*it], lxAtt1, lxAtt2);

    }

    FreeSpace(&xMExpLoss);
    FreeSpace(&xExpLoss);
    FreeSpace(&xSumLoss);
    FreeSpace(&xPoolIndexLoss);
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _tri_gates.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _uni_gates.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }


};

#endif /* SRC_AttentionPooling_H_ */
