

#ifndef SRC_CLASSIFIER_5_H_
#define SRC_CLASSIFIER_5_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class Classifier_5 {
public:
	Classifier_5() {
    _dropOut = 0.5;
  }
  ~Classifier_5() {

  }

public:
  LookupTable<xpu> _words;

  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;

  int _token_representation_size;
  int _inputsize;
  int _hiddensize;
  int _rnnHiddenSize;

  UniLayer<xpu> _olayer_linear;

  GRNN<xpu> _rnn_left;
  GRNN<xpu> _rnn_right;

  int _poolmanners;
  int _poolfunctions;


  int _poolsize;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3, min, 4, std, 5, pro

  Options options;
  UniLayer<xpu> penulLayer;
  int _penultimate;

public:

  inline void init(const NRMat<dtype>& wordEmb, Options options) {
	  this->options = options;
    _wordcontext = options.wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _labelSize = MAX_RELATION;
    _token_representation_size = _wordDim;
    _poolfunctions = 5;  // (avg, max, min, std, pro)
    _poolmanners = _poolfunctions * 1;
    _inputsize = _wordwindow * _token_representation_size;
    _hiddensize = options.rnnHiddenSize;
    _rnnHiddenSize = options.rnnHiddenSize;
    _penultimate = options.hiddenSize;

    _poolsize = 2*_poolmanners * _hiddensize;

    _words.initial(wordEmb);

    _rnn_left.initial(_rnnHiddenSize, _inputsize, true, 10);
    _rnn_right.initial(_rnnHiddenSize, _inputsize, true, 40);

    //_olayer_linear.initial(_labelSize, _poolsize, false, 70, 2);
    penulLayer.initial(_penultimate, _poolsize, true, 50, 0);
    _olayer_linear.initial(_labelSize, _penultimate, false, 70, 2);

    _remove = 0;

    cout<<"do grnn in from e1 to ancestor and e2 to ancestor"<<endl;
  }

  inline void release() {
    _words.release();
    _olayer_linear.release();
    _rnn_left.release();
    _rnn_right.release();
    penulLayer.release();
  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int leftsize = example.m_before.size();
      int rightsize = example.m_after.size();

      if( leftsize==0 || rightsize==0)
    	  continue;

      Tensor<xpu, 3, dtype> inputLeft, inputLeftLoss;
      Tensor<xpu, 3, dtype> inputRight, inputRightLoss;

      Tensor<xpu, 3, dtype> rnn_hidden_left, rnn_hidden_leftLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current;
      Tensor<xpu, 3, dtype> rnn_hidden_right, rnn_hidden_rightLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current;

      vector<Tensor<xpu, 2, dtype> > poolLeft(_poolmanners), poolLeftLoss(_poolmanners);
      vector<Tensor<xpu, 3, dtype> > poolIndexLeft(_poolmanners);
      vector<Tensor<xpu, 2, dtype> > poolRight(_poolmanners), poolRightLoss(_poolmanners);
      vector<Tensor<xpu, 3, dtype> > poolIndexRight(_poolmanners);

      Tensor<xpu, 2, dtype> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;
      Tensor<xpu, 2, dtype> penul, penulLoss;

      Tensor<xpu, 3, dtype> wordprimeLeft, wordprimeLeftLoss, wordprimeLeftMask;
      Tensor<xpu, 3, dtype> wordrepresentLeft, wordrepresentLeftLoss;
      Tensor<xpu, 3, dtype> wordprimeRight, wordprimeRightLoss, wordprimeRightMask;
      Tensor<xpu, 3, dtype> wordrepresentRight, wordrepresentRightLoss;

      //initialize
      wordprimeLeft = NewTensor<xpu>(Shape3(leftsize, 1, _wordDim), 0.0);
      wordprimeLeftLoss = NewTensor<xpu>(Shape3(leftsize, 1, _wordDim), 0.0);
      wordprimeLeftMask = NewTensor<xpu>(Shape3(leftsize, 1, _wordDim), 1.0);
      wordrepresentLeft = NewTensor<xpu>(Shape3(leftsize, 1, _token_representation_size), 0.0);
      wordrepresentLeftLoss = NewTensor<xpu>(Shape3(leftsize, 1, _token_representation_size), 0.0);

      wordprimeRight = NewTensor<xpu>(Shape3(rightsize, 1, _wordDim), 0.0);
      wordprimeRightLoss = NewTensor<xpu>(Shape3(rightsize, 1, _wordDim), 0.0);
      wordprimeRightMask = NewTensor<xpu>(Shape3(rightsize, 1, _wordDim), 1.0);
      wordrepresentRight = NewTensor<xpu>(Shape3(rightsize, 1, _token_representation_size), 0.0);
      wordrepresentRightLoss = NewTensor<xpu>(Shape3(rightsize, 1, _token_representation_size), 0.0);

      inputLeft = NewTensor<xpu>(Shape3(leftsize, 1, _inputsize), 0.0);
      inputLeftLoss = NewTensor<xpu>(Shape3(leftsize, 1, _inputsize), 0.0);
      inputRight = NewTensor<xpu>(Shape3(rightsize, 1, _inputsize), 0.0);
      inputRightLoss = NewTensor<xpu>(Shape3(rightsize, 1, _inputsize), 0.0);

      rnn_hidden_left_reset = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_update = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_current = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_leftLoss = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);

      rnn_hidden_right_reset = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_update = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_current = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_rightLoss = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);


      for (int idm = 0; idm < _poolmanners; idm++) {
        poolLeft[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        poolLeftLoss[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        poolIndexLeft[idm] = NewTensor<xpu>(Shape3(leftsize, 1, _hiddensize), 0.0);
        poolRight[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        poolRightLoss[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        poolIndexRight[idm] = NewTensor<xpu>(Shape3(rightsize, 1, _hiddensize), 0.0);
      }


      poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      /*      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
            outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);*/
            penul = NewTensor<xpu>(Shape2(1, _penultimate), 0.0);
            penulLoss = NewTensor<xpu>(Shape2(1, _penultimate), 0.0);
            output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
            outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);


      for (int idx = 0; idx < leftsize; idx++) {
        srand(iter * example_num + count * leftsize + idx);

       _words.GetEmb(example.m_before[idx], wordprimeLeft[idx]);

        //dropout
        dropoutcol(wordprimeLeftMask[idx], _dropOut);
        wordprimeLeft[idx] = wordprimeLeft[idx] * wordprimeLeftMask[idx];
      }
      for (int idx = 0; idx < rightsize; idx++) {
        srand(iter * example_num + count * rightsize + idx);

       _words.GetEmb(example.m_after[idx], wordprimeRight[idx]);

        //dropout
        dropoutcol(wordprimeRightMask[idx], _dropOut);
        wordprimeRight[idx] = wordprimeRight[idx] * wordprimeRightMask[idx];
      }


      for (int idx = 0; idx < leftsize; idx++) {
        wordrepresentLeft[idx] += wordprimeLeft[idx];
      }

      for (int idx = 0; idx < rightsize; idx++) {
        wordrepresentRight[idx] += wordprimeRight[idx];
      }

      windowlized(wordrepresentLeft, inputLeft, _wordcontext);
      windowlized(wordrepresentRight, inputRight, _wordcontext);

      _rnn_left.ComputeForwardScore(inputLeft, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
      _rnn_right.ComputeForwardScore(inputRight, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

      offset = 0;
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(rnn_hidden_left, poolLeft[offset], poolIndexLeft[offset]);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(rnn_hidden_left, poolLeft[offset + 1], poolIndexLeft[offset + 1]);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(rnn_hidden_left, poolLeft[offset + 2], poolIndexLeft[offset + 2]);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(rnn_hidden_left, poolLeft[offset + 3], poolIndexLeft[offset + 3]);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(rnn_hidden_left, poolLeft[offset + 4], poolIndexLeft[offset + 4]);
      }

      offset = 0;
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(rnn_hidden_right, poolRight[offset], poolIndexRight[offset]);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(rnn_hidden_right, poolRight[offset + 1], poolIndexRight[offset + 1]);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(rnn_hidden_right, poolRight[offset + 2], poolIndexRight[offset + 2]);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(rnn_hidden_right, poolRight[offset + 3], poolIndexRight[offset + 3]);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(rnn_hidden_right, poolRight[offset + 4], poolIndexRight[offset + 4]);
      }

      vector<Tensor<xpu, 2, dtype> > v_penultimate;
      v_penultimate.push_back(poolLeft[offset]);
      v_penultimate.push_back(poolLeft[offset+1]);
      v_penultimate.push_back(poolLeft[offset+2]);
      v_penultimate.push_back(poolLeft[offset+3]);
      v_penultimate.push_back(poolLeft[offset+4]);
      v_penultimate.push_back(poolRight[offset]);
      v_penultimate.push_back(poolRight[offset+1]);
      v_penultimate.push_back(poolRight[offset+2]);
      v_penultimate.push_back(poolRight[offset+3]);
      v_penultimate.push_back(poolRight[offset+4]);
     // cout<<"pool "<<poolmerge.size(1)<<" "<<poolLeft[offset].size(1)<<endl;
      concat(v_penultimate, poolmerge);

      //_olayer_linear.ComputeForwardScore(poolmerge, output);
      penulLayer.ComputeForwardScore(poolmerge, penul);
      _olayer_linear.ComputeForwardScore(penul, output);

      // get delta for each output
      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      //_olayer_linear.ComputeBackwardLoss(poolmerge, output, outputLoss, poolmergeLoss);

      _olayer_linear.ComputeBackwardLoss(penul, output, outputLoss, penulLoss);
      penulLayer.ComputeBackwardLoss(poolmerge, penul, penulLoss, poolmergeLoss);

      vector<Tensor<xpu, 2, dtype> > v_penultimateLoss;
      v_penultimateLoss.push_back(poolLeftLoss[offset]);
      v_penultimateLoss.push_back(poolLeftLoss[offset+1]);
      v_penultimateLoss.push_back(poolLeftLoss[offset+2]);
      v_penultimateLoss.push_back(poolLeftLoss[offset+3]);
      v_penultimateLoss.push_back(poolLeftLoss[offset+4]);
      v_penultimateLoss.push_back(poolRightLoss[offset]);
      v_penultimateLoss.push_back(poolRightLoss[offset+1]);
      v_penultimateLoss.push_back(poolRightLoss[offset+2]);
      v_penultimateLoss.push_back(poolRightLoss[offset+3]);
      v_penultimateLoss.push_back(poolRightLoss[offset+4]);


      //before
      unconcat(v_penultimateLoss, poolmergeLoss);

      offset = 0;
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolLeftLoss[offset], poolIndexLeft[offset],  rnn_hidden_leftLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolLeftLoss[offset + 1], poolIndexLeft[offset + 1], rnn_hidden_leftLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolLeftLoss[offset + 2], poolIndexLeft[offset + 2], rnn_hidden_leftLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolLeftLoss[offset + 3], poolIndexLeft[offset + 3], rnn_hidden_leftLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolLeftLoss[offset + 4], poolIndexLeft[offset + 4], rnn_hidden_leftLoss);
      }


      offset = 0;
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolRightLoss[offset], poolIndexRight[offset],  rnn_hidden_rightLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolRightLoss[offset + 1], poolIndexRight[offset + 1], rnn_hidden_rightLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolRightLoss[offset + 2], poolIndexRight[offset + 2], rnn_hidden_rightLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolRightLoss[offset + 3], poolIndexRight[offset + 3], rnn_hidden_rightLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolRightLoss[offset + 4], poolIndexRight[offset + 4], rnn_hidden_rightLoss);
      }


      _rnn_left.ComputeBackwardLoss(inputLeft, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left, rnn_hidden_leftLoss, inputLeftLoss);
      _rnn_right.ComputeBackwardLoss(inputRight, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right, rnn_hidden_rightLoss, inputRightLoss);
	  
      // word context
      windowlized_backward(wordrepresentLeftLoss, inputLeftLoss, _wordcontext);
      windowlized_backward(wordrepresentRightLoss, inputRightLoss, _wordcontext);

      for (int idx = 0; idx < leftsize; idx++) {
        wordprimeLeftLoss[idx] += wordrepresentLeftLoss[idx];
      }
      for (int idx = 0; idx < rightsize; idx++) {
        wordprimeRightLoss[idx] += wordrepresentRightLoss[idx];
      }

      if (_words.bEmbFineTune()) {
        for (int idx = 0; idx < leftsize; idx++) {
          wordprimeLeftLoss[idx] = wordprimeLeftLoss[idx] * wordprimeLeftMask[idx];
          _words.EmbLoss(example.m_before[idx], wordprimeLeftLoss[idx]);
        }
        for (int idx = 0; idx < rightsize; idx++) {
          wordprimeRightLoss[idx] = wordprimeRightLoss[idx] * wordprimeRightMask[idx];
          _words.EmbLoss(example.m_after[idx], wordprimeRightLoss[idx]);
        }
      }

      //release
      FreeSpace(&wordprimeLeft);
      FreeSpace(&wordprimeLeftLoss);
      FreeSpace(&wordprimeLeftMask);
      FreeSpace(&wordrepresentLeft);
      FreeSpace(&wordrepresentLeftLoss);

      FreeSpace(&wordprimeRight);
      FreeSpace(&wordprimeRightLoss);
      FreeSpace(&wordprimeRightMask);
      FreeSpace(&wordrepresentRight);
      FreeSpace(&wordrepresentRightLoss);

      FreeSpace(&inputLeft);
      FreeSpace(&inputLeftLoss);
      FreeSpace(&inputRight);
      FreeSpace(&inputRightLoss);

      FreeSpace(&rnn_hidden_left_reset);
      FreeSpace(&rnn_hidden_left_update);
      FreeSpace(&rnn_hidden_left_afterreset);
      FreeSpace(&rnn_hidden_left_current);
      FreeSpace(&rnn_hidden_left);
      FreeSpace(&rnn_hidden_leftLoss);


      FreeSpace(&rnn_hidden_right_reset);
      FreeSpace(&rnn_hidden_right_update);
      FreeSpace(&rnn_hidden_right_afterreset);
      FreeSpace(&rnn_hidden_right_current);
      FreeSpace(&rnn_hidden_right);
      FreeSpace(&rnn_hidden_rightLoss);



      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(poolLeft[idm]));
        FreeSpace(&(poolLeftLoss[idm]));
        FreeSpace(&(poolIndexLeft[idm]));
        FreeSpace(&(poolRight[idm]));
        FreeSpace(&(poolRightLoss[idm]));
        FreeSpace(&(poolIndexRight[idm]));
      }


      FreeSpace(&poolmerge);
      FreeSpace(&poolmergeLoss);
      FreeSpace(&output);
      FreeSpace(&outputLoss);
      FreeSpace(&penul);
      FreeSpace(&penulLoss);
    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  int predict(const Example& example, vector<dtype>& results) {
	const vector<Feature>& features = example.m_features;
    int leftsize = example.m_before.size();
    int rightsize = example.m_after.size();

    if( leftsize==0 || rightsize==0)
  	  return OTHER_LABEL;

    int offset = 0;

    Tensor<xpu, 3, dtype> inputLeft;
    Tensor<xpu, 3, dtype> inputRight;

    Tensor<xpu, 3, dtype> rnn_hidden_left_update;
    Tensor<xpu, 3, dtype> rnn_hidden_left_reset;
    Tensor<xpu, 3, dtype> rnn_hidden_left;
    Tensor<xpu, 3, dtype> rnn_hidden_left_afterreset;
    Tensor<xpu, 3, dtype> rnn_hidden_left_current;

    Tensor<xpu, 3, dtype> rnn_hidden_right_update;
    Tensor<xpu, 3, dtype> rnn_hidden_right_reset;
    Tensor<xpu, 3, dtype> rnn_hidden_right;
    Tensor<xpu, 3, dtype> rnn_hidden_right_afterreset;
    Tensor<xpu, 3, dtype> rnn_hidden_right_current;

    vector<Tensor<xpu, 2, dtype> > poolLeft(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndexLeft(_poolmanners);
    vector<Tensor<xpu, 2, dtype> > poolRight(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndexRight(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> output;
    Tensor<xpu, 2, dtype> penul;

    Tensor<xpu, 3, dtype> wordprimeLeft;
    Tensor<xpu, 3, dtype> wordrepresentLeft;
    Tensor<xpu, 3, dtype> wordprimeRight;
    Tensor<xpu, 3, dtype> wordrepresentRight;


    //initialize
    wordprimeLeft = NewTensor<xpu>(Shape3(leftsize, 1, _wordDim), 0.0);
    wordrepresentLeft = NewTensor<xpu>(Shape3(leftsize, 1, _token_representation_size), 0.0);

    wordprimeRight = NewTensor<xpu>(Shape3(rightsize, 1, _wordDim), 0.0);
    wordrepresentRight = NewTensor<xpu>(Shape3(rightsize, 1, _token_representation_size), 0.0);

    inputLeft = NewTensor<xpu>(Shape3(leftsize, 1, _inputsize), 0.0);
    inputRight = NewTensor<xpu>(Shape3(rightsize, 1, _inputsize), 0.0);

    rnn_hidden_left_reset = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_update = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_current = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left = NewTensor<xpu>(Shape3(leftsize, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_right_reset = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_update = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_current = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right = NewTensor<xpu>(Shape3(rightsize, 1, _rnnHiddenSize), 0.0);


    for (int idm = 0; idm < _poolmanners; idm++) {
        poolLeft[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      poolIndexLeft[idm] = NewTensor<xpu>(Shape3(leftsize, 1, _hiddensize), 0.0);
      poolRight[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      poolIndexRight[idm] = NewTensor<xpu>(Shape3(rightsize, 1, _hiddensize), 0.0);
    }

    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    penul = NewTensor<xpu>(Shape2(1, _penultimate), 0.0);

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < leftsize; idx++) {
     _words.GetEmb(example.m_before[idx], wordprimeLeft[idx]);
    }
    for (int idx = 0; idx < rightsize; idx++) {
     _words.GetEmb(example.m_after[idx], wordprimeRight[idx]);
    }

    for (int idx = 0; idx < leftsize; idx++) {
      wordrepresentLeft[idx] += wordprimeLeft[idx];
    }

    for (int idx = 0; idx < rightsize; idx++) {
      wordrepresentRight[idx] += wordprimeRight[idx];
    }

    windowlized(wordrepresentLeft, inputLeft, _wordcontext);
    windowlized(wordrepresentRight, inputRight, _wordcontext);

    _rnn_left.ComputeForwardScore(inputLeft, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
    _rnn_right.ComputeForwardScore(inputRight, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

    offset = 0;
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(rnn_hidden_left, poolLeft[offset], poolIndexLeft[offset]);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(rnn_hidden_left, poolLeft[offset + 1], poolIndexLeft[offset + 1]);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(rnn_hidden_left, poolLeft[offset + 2], poolIndexLeft[offset + 2]);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(rnn_hidden_left, poolLeft[offset + 3], poolIndexLeft[offset + 3]);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(rnn_hidden_left, poolLeft[offset + 4], poolIndexLeft[offset + 4]);
    }

    offset = 0;
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(rnn_hidden_right, poolRight[offset], poolIndexRight[offset]);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(rnn_hidden_right, poolRight[offset + 1], poolIndexRight[offset + 1]);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(rnn_hidden_right, poolRight[offset + 2], poolIndexRight[offset + 2]);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(rnn_hidden_right, poolRight[offset + 3], poolIndexRight[offset + 3]);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(rnn_hidden_right, poolRight[offset + 4], poolIndexRight[offset + 4]);
    }

    vector<Tensor<xpu, 2, dtype> > v_penultimate;
    v_penultimate.push_back(poolLeft[offset]);
    v_penultimate.push_back(poolLeft[offset+1]);
    v_penultimate.push_back(poolLeft[offset+2]);
    v_penultimate.push_back(poolLeft[offset+3]);
    v_penultimate.push_back(poolLeft[offset+4]);
    v_penultimate.push_back(poolRight[offset]);
    v_penultimate.push_back(poolRight[offset+1]);
    v_penultimate.push_back(poolRight[offset+2]);
    v_penultimate.push_back(poolRight[offset+3]);
    v_penultimate.push_back(poolRight[offset+4]);
   // cout<<"pool "<<poolmerge.size(1)<<" "<<poolLeft[offset].size(1)<<endl;
    concat(v_penultimate, poolmerge);


    //_olayer_linear.ComputeForwardScore(poolmerge, output);
    penulLayer.ComputeForwardScore(poolmerge, penul);
    _olayer_linear.ComputeForwardScore(penul, output);

    // decode algorithm
    int optLabel = softmax_predict(output, results);

    //release
    FreeSpace(&wordprimeLeft);
    FreeSpace(&wordrepresentLeft);

    FreeSpace(&wordprimeRight);
    FreeSpace(&wordrepresentRight);


    FreeSpace(&inputLeft);
    FreeSpace(&inputRight);

    FreeSpace(&rnn_hidden_left_reset);
    FreeSpace(&rnn_hidden_left_update);
    FreeSpace(&rnn_hidden_left_afterreset);
    FreeSpace(&rnn_hidden_left_current);
    FreeSpace(&rnn_hidden_left);

    FreeSpace(&rnn_hidden_right_reset);
    FreeSpace(&rnn_hidden_right_update);
    FreeSpace(&rnn_hidden_right_afterreset);
    FreeSpace(&rnn_hidden_right_current);
    FreeSpace(&rnn_hidden_right);

    for (int idm = 0; idm < _poolmanners; idm++) {
      FreeSpace(&(poolLeft[idm]));
      FreeSpace(&(poolIndexLeft[idm]));
      FreeSpace(&(poolRight[idm]));
      FreeSpace(&(poolIndexRight[idm]));
    }


    FreeSpace(&poolmerge);
    FreeSpace(&output);
    FreeSpace(&penul);

    return optLabel;
  }

  dtype computeScore(const Example& example) {
	return 0;
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {

    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    penulLayer.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {
	
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 3, dtype> Wd, Tensor<xpu, 3, dtype> gradWd, const string& mark, int iter) {

  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
       const hash_set<int>& indexes, bool bRow = true) {
  }

  void checkgrads(const vector<Example>& examples, int iter) {

  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
    _words.setEmbFineTune(b_wordEmb_finetune);
  }

  inline void resetRemove(int remove) {
    _remove = remove;
  }
};

#endif /* SRC_PoolRNNClassifier_H_ */

