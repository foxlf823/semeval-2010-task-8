/*
 * LSTM.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_LSTM_CELL_H_
#define SRC_LSTM_CELL_H_
#include "tensor.h"

#include "BiLayer.h"
#include "MyLib.h"
#include "Utiltensor.h"
#include "TriLayerLSTM.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class LSTM_cell {
public:
  TriLayerLSTM<xpu> _lstm_output;
  TriLayerLSTM<xpu> _lstm_input;
  TriLayerLSTM<xpu> _lstm_forget;
  BiLayer<xpu> _lstm_cell;
  bool _left2right;

  Tensor<xpu, 2, dtype> _null1, _null1Loss, _null2, _null2Loss;

public:
  LSTM_cell() {
  }

  inline void initial(int outputsize, int inputsize, bool left2right, int seed = 0) {
    _left2right = left2right;

    _lstm_output.initial(outputsize, outputsize, inputsize, true, seed, 1);
    _lstm_input.initial(outputsize, outputsize, inputsize, true, seed + 10, 1);
    _lstm_forget.initial(outputsize, outputsize, inputsize, true, seed + 20, 1);
    _lstm_cell.initial(outputsize, outputsize, inputsize, true, seed + 30, 0);

    _null1 = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _null1Loss = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _null2 = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _null2Loss = NewTensor<xpu>(Shape2(1, outputsize), d_zero);

  }


  inline void release() {
    _lstm_output.release();
    _lstm_input.release();
    _lstm_forget.release();
    _lstm_cell.release();

    FreeSpace(&_null1);
    FreeSpace(&_null1Loss);
    FreeSpace(&_null2);
    FreeSpace(&_null2Loss);
  }

  virtual ~LSTM_cell() {
    // TODO Auto-generated destructor stub
  }


public:

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> iy, Tensor<xpu, 3, dtype> oy, Tensor<xpu, 3, dtype> fy,
      Tensor<xpu, 3, dtype> mcy, Tensor<xpu, 3, dtype> cy, Tensor<xpu, 3, dtype> my, Tensor<xpu, 3, dtype> y) {
    iy = 0.0;
    oy = 0.0;
    fy = 0.0;
    mcy = 0.0;
    cy = 0.0;
    my = 0.0;
    y = 0.0;
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if (_left2right) {
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0) {
          _lstm_input.ComputeForwardScore(_null1, x[idx], _null2,  iy[idx]);
          _lstm_cell.ComputeForwardScore(_null1, x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx];
          _lstm_output.ComputeForwardScore(_null1, x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        } else {
          _lstm_input.ComputeForwardScore(y[idx - 1], x[idx], cy[idx - 1], iy[idx]);
          _lstm_forget.ComputeForwardScore(y[idx - 1], x[idx], cy[idx - 1], fy[idx]);
          _lstm_cell.ComputeForwardScore(y[idx - 1], x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx] + cy[idx - 1] * fy[idx];
          _lstm_output.ComputeForwardScore(y[idx - 1], x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        }
      }
    } else {
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx == seq_size - 1) {
          _lstm_input.ComputeForwardScore(_null1, x[idx], _null2,  iy[idx]);
          _lstm_cell.ComputeForwardScore(_null1, x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx];
          _lstm_output.ComputeForwardScore(_null1, x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        } else {
          _lstm_input.ComputeForwardScore(y[idx + 1], x[idx], cy[idx + 1], iy[idx]);
          _lstm_forget.ComputeForwardScore(y[idx + 1], x[idx], cy[idx + 1], fy[idx]);
          _lstm_cell.ComputeForwardScore(y[idx + 1], x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx] + cy[idx + 1] * fy[idx];
          _lstm_output.ComputeForwardScore(y[idx + 1], x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        }
      }
    }
  }
  

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> iy, Tensor<xpu, 3, dtype> oy, Tensor<xpu, 3, dtype> fy,
      Tensor<xpu, 3, dtype> mcy, Tensor<xpu, 3, dtype> cy, Tensor<xpu, 3, dtype> my, 
      Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly, Tensor<xpu, 3, dtype> lcy, Tensor<xpu, 3, dtype> lx, bool bclear = false) {
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if (bclear) lx = 0.0;
    	
    //left rnn
    Tensor<xpu, 3, dtype> liy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lfy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> loy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lmcy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    //Tensor<xpu, 3, dtype> lcy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lmy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    
    Tensor<xpu, 3, dtype> lFcy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lFy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);

    if (_left2right) {
      //left rnn
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx < seq_size - 1)
          ly[idx] = ly[idx] + lFy[idx];

        lmy[idx] = ly[idx] * oy[idx];
        loy[idx] = ly[idx] * my[idx];
        if (idx < seq_size - 1) {
          lcy[idx] += lmy[idx] * (1.0 - my[idx] * my[idx]) + lFcy[idx];
        } else {
          lcy[idx] += lmy[idx] * (1.0 - my[idx] * my[idx]);
        }

        if (idx == 0) {
          _lstm_output.ComputeBackwardLoss(_null1, x[idx], cy[idx], oy[idx],
              loy[idx], _null1Loss, lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];

          _lstm_cell.ComputeBackwardLoss(_null1, x[idx], mcy[idx], lmcy[idx], _null1Loss, lx[idx]);

          _lstm_input.ComputeBackwardLoss(_null1, x[idx], _null2, iy[idx],
              liy[idx], _null1Loss, lx[idx], _null2Loss);

        } else {
          _lstm_output.ComputeBackwardLoss(y[idx - 1], x[idx], cy[idx], oy[idx],
              loy[idx], lFy[idx - 1], lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];
          lFcy[idx - 1] = lcy[idx] * fy[idx];
          lfy[idx] = lcy[idx] * cy[idx - 1];

          _lstm_cell.ComputeBackwardLoss(y[idx - 1], x[idx], mcy[idx],
              lmcy[idx], lFy[idx - 1], lx[idx]);

          _lstm_forget.ComputeBackwardLoss(y[idx - 1], x[idx], cy[idx - 1],
              fy[idx], lfy[idx], lFy[idx - 1], lx[idx], lFcy[idx - 1]);

          _lstm_input.ComputeBackwardLoss(y[idx - 1], x[idx], cy[idx - 1],
              iy[idx], liy[idx], lFy[idx - 1], lx[idx], lFcy[idx - 1]);
        }
      }
    } else {
      // right rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx > 0)
          ly[idx] = ly[idx] + lFy[idx];

        lmy[idx] = ly[idx] * oy[idx];
        loy[idx] = ly[idx] * my[idx];
        if (idx > 0) {
          lcy[idx] += lmy[idx] * (1.0 - my[idx] * my[idx]) + lFcy[idx];
        } else {
          lcy[idx] += lmy[idx] * (1.0 - my[idx] * my[idx]);
        }

        if (idx == seq_size - 1) {
          _lstm_output.ComputeBackwardLoss(_null1, x[idx], cy[idx], oy[idx],
              loy[idx], _null1Loss, lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];

          _lstm_cell.ComputeBackwardLoss(_null1, x[idx], mcy[idx], lmcy[idx], _null1Loss, lx[idx]);

          _lstm_input.ComputeBackwardLoss(_null1, x[idx], _null2, iy[idx],
              liy[idx], _null1Loss, lx[idx], _null2Loss);
              
        } else {
          _lstm_output.ComputeBackwardLoss(y[idx + 1], x[idx], cy[idx], oy[idx],
              loy[idx], lFy[idx + 1], lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];
          lFcy[idx + 1] = lcy[idx] * fy[idx];
          lfy[idx] = lcy[idx] * cy[idx + 1];

          _lstm_cell.ComputeBackwardLoss(y[idx + 1], x[idx], mcy[idx],
              lmcy[idx], lFy[idx + 1], lx[idx]);

          _lstm_forget.ComputeBackwardLoss(y[idx + 1], x[idx], cy[idx + 1],
              fy[idx], lfy[idx], lFy[idx + 1], lx[idx], lFcy[idx + 1]);

          _lstm_input.ComputeBackwardLoss(y[idx + 1], x[idx], cy[idx + 1],
              iy[idx], liy[idx], lFy[idx + 1], lx[idx], lFcy[idx + 1]);
        }
      }
    }

    FreeSpace(&liy);
    FreeSpace(&lfy);
    FreeSpace(&loy);
    FreeSpace(&lmcy);
    //FreeSpace(&lcy);
    FreeSpace(&lmy);
    FreeSpace(&lFcy);
    FreeSpace(&lFy);   
  }
  

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _lstm_output.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _lstm_input.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _lstm_forget.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _lstm_cell.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }

};

#endif /* SRC_LSTM_H_ */
