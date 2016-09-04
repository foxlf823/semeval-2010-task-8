
#ifndef SRC_CLASSIFIER_3_H_
#define SRC_CLASSIFIER_3_H_

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

//A native neural network classfier using only word embeddings
template<typename xpu>
class Classifier_3 {
public:
	Classifier_3() {
    _dropOut = 0.5;
  }
  ~Classifier_3() {

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
  UniLayer<xpu> _tanh_project;

  LSTM<xpu> _rnn_left;
  LSTM<xpu> _rnn_right;

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
    _poolfunctions = 5;// (avg, max, min, std, pro)
    _poolmanners = _poolfunctions * 1;
    _inputsize = _wordwindow * _token_representation_size;
    _hiddensize = options.wordEmbSize;
    _rnnHiddenSize = options.rnnHiddenSize;
    _penultimate = options.hiddenSize;

    _poolsize = _poolmanners * _hiddensize;

    _words.initial(wordEmb);

    _rnn_left.initial(_rnnHiddenSize, _inputsize, true, 10);
    _rnn_right.initial(_rnnHiddenSize, _inputsize, false, 40);

    _tanh_project.initial(_hiddensize, 2 * _rnnHiddenSize, true, 60, 0);
    //_olayer_linear.initial(_labelSize, _poolsize, false, 70, 2);
    penulLayer.initial(_penultimate, _poolsize, true, 50, 0);
    _olayer_linear.initial(_labelSize, _penultimate, false, 70, 2);

    _remove = 0;

    cout<<"do lstm recurrent in the dependency path of the two entities"<<endl;
  }

  inline void release() {
    _words.release();


    _olayer_linear.release();
    _tanh_project.release();
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

      int seq_size = example.m_features.size();
      if(seq_size==0)
    	  continue;

      Tensor<xpu, 3, dtype> input, inputLoss;
      Tensor<xpu, 3, dtype> project, projectLoss;

      Tensor<xpu, 3, dtype> rnn_hidden_left, rnn_hidden_leftLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_left_iy, rnn_hidden_left_oy, rnn_hidden_left_fy,
	  	  rnn_hidden_left_mcy, rnn_hidden_left_cy, rnn_hidden_left_my;
      Tensor<xpu, 3, dtype> rnn_hidden_right, rnn_hidden_rightLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_right_iy, rnn_hidden_right_oy, rnn_hidden_right_fy,
	  	  rnn_hidden_right_mcy, rnn_hidden_right_cy, rnn_hidden_right_my;

      Tensor<xpu, 3, dtype> rnn_hidden_merge, rnn_hidden_mergeLoss;

      vector<Tensor<xpu, 2, dtype> > pool(_poolmanners), poolLoss(_poolmanners);
      vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

      Tensor<xpu, 2, dtype> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, dtype> penul, penulLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;

      Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, dtype> wordrepresent, wordrepresentLoss;



      static hash_set<int>::iterator it;

      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 1.0);
      wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);
      wordrepresentLoss = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

      input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);
      inputLoss = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);


      rnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_leftLoss = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

      rnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_rightLoss = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

      rnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2 * _rnnHiddenSize), 0.0);
      rnn_hidden_mergeLoss = NewTensor<xpu>(Shape3(seq_size, 1, 2 * _rnnHiddenSize), 0.0);

      project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
      projectLoss = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);

      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        poolLoss[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
      }


      poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      /*      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
            outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);*/
            penul = NewTensor<xpu>(Shape2(1, _penultimate), 0.0);
            penulLoss = NewTensor<xpu>(Shape2(1, _penultimate), 0.0);
            output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
            outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);

        const vector<int>& words = feature.words;

       _words.GetEmb(words[0], wordprime[idx]);

        //dropout
        dropoutcol(wordprimeMask[idx], _dropOut);
        wordprime[idx] = wordprime[idx] * wordprimeMask[idx];
      }

      for (int idx = 0; idx < seq_size; idx++) {
        wordrepresent[idx] += wordprime[idx];
      }

      windowlized(wordrepresent, input, _wordcontext);

      _rnn_left.ComputeForwardScore(input, rnn_hidden_left_iy, rnn_hidden_left_oy,
    		  rnn_hidden_left_fy, rnn_hidden_left_mcy, rnn_hidden_left_cy, rnn_hidden_left_my,
			  rnn_hidden_left);
      _rnn_right.ComputeForwardScore(input, rnn_hidden_right_iy, rnn_hidden_right_oy,
    		  rnn_hidden_right_fy, rnn_hidden_right_mcy, rnn_hidden_right_cy, rnn_hidden_right_my,
			  rnn_hidden_right);


      for (int idx = 0; idx < seq_size; idx++) {
        concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);
      }

      // do we need a convolution? future work, currently needn't
      for (int idx = 0; idx < seq_size; idx++) {
        _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);
      }

      offset = 0;
      //before
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(project, pool[offset], poolIndex[offset]);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1]);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(project, pool[offset + 2], poolIndex[offset + 2]);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3]);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(project, pool[offset + 4], poolIndex[offset + 4]);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], poolmerge);


      //_olayer_linear.ComputeForwardScore(poolmerge, output);
      penulLayer.ComputeForwardScore(poolmerge, penul);
      _olayer_linear.ComputeForwardScore(penul, output);

      // get delta for each output
      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      //_olayer_linear.ComputeBackwardLoss(poolmerge, output, outputLoss, poolmergeLoss);

      _olayer_linear.ComputeBackwardLoss(penul, output, outputLoss, penulLoss);
      penulLayer.ComputeBackwardLoss(poolmerge, penul, penulLoss, poolmergeLoss);


      offset = 0;
      //before
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], poolmergeLoss);

      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolLoss[offset], poolIndex[offset],  projectLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], projectLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], projectLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolLoss[offset + 3], poolIndex[offset + 3], projectLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolLoss[offset + 4], poolIndex[offset + 4], projectLoss);
      }


      for (int idx = 0; idx < seq_size; idx++) {
        _tanh_project.ComputeBackwardLoss(rnn_hidden_merge[idx], project[idx], projectLoss[idx], rnn_hidden_mergeLoss[idx]);
      }

      for (int idx = 0; idx < seq_size; idx++) {
        unconcat(rnn_hidden_leftLoss[idx], rnn_hidden_rightLoss[idx], rnn_hidden_mergeLoss[idx]);
      }

      _rnn_left.ComputeBackwardLoss(input, rnn_hidden_left_iy, rnn_hidden_left_oy,
    		  rnn_hidden_left_fy, rnn_hidden_left_mcy, rnn_hidden_left_cy, rnn_hidden_left_my,
			  rnn_hidden_left, rnn_hidden_leftLoss, inputLoss);
      _rnn_right.ComputeBackwardLoss(input, rnn_hidden_right_iy, rnn_hidden_right_oy,
    		  rnn_hidden_right_fy, rnn_hidden_right_mcy, rnn_hidden_right_cy, rnn_hidden_right_my,
			  rnn_hidden_right, rnn_hidden_rightLoss, inputLoss);

      // word context
      windowlized_backward(wordrepresentLoss, inputLoss, _wordcontext);

      for (int idx = 0; idx < seq_size; idx++) {
        wordprimeLoss[idx] += wordrepresentLoss[idx];
      }

      if (_words.bEmbFineTune()) {
        for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];
          const vector<int>& words = feature.words;
          wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
          _words.EmbLoss(words[0], wordprimeLoss[idx]);
        }
      }

      //release
      FreeSpace(&wordprime);
      FreeSpace(&wordprimeLoss);
      FreeSpace(&wordprimeMask);
      FreeSpace(&wordrepresent);
      FreeSpace(&wordrepresentLoss);

      FreeSpace(&input);
      FreeSpace(&inputLoss);


		FreeSpace(&(rnn_hidden_left_iy));
		FreeSpace(&(rnn_hidden_left_oy));
		FreeSpace(&(rnn_hidden_left_fy));
		FreeSpace(&(rnn_hidden_left_mcy));
		FreeSpace(&(rnn_hidden_left_cy));
		FreeSpace(&(rnn_hidden_left_my));
	      FreeSpace(&rnn_hidden_left);
	      FreeSpace(&rnn_hidden_leftLoss);

		FreeSpace(&(rnn_hidden_right_iy));
		FreeSpace(&(rnn_hidden_right_oy));
		FreeSpace(&(rnn_hidden_right_fy));
		FreeSpace(&(rnn_hidden_right_mcy));
		FreeSpace(&(rnn_hidden_right_cy));
		FreeSpace(&(rnn_hidden_right_my));
	      FreeSpace(&rnn_hidden_right);
	      FreeSpace(&rnn_hidden_rightLoss);


      FreeSpace(&rnn_hidden_merge);
      FreeSpace(&rnn_hidden_mergeLoss);

      FreeSpace(&project);
      FreeSpace(&projectLoss);

      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[idm]));
        FreeSpace(&(poolLoss[idm]));
        FreeSpace(&(poolIndex[idm]));
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
    int seq_size = features.size();
    if(seq_size==0)
    	return OTHER_LABEL;
    int offset = 0;

    Tensor<xpu, 3, dtype> input;
    Tensor<xpu, 3, dtype> project;

    Tensor<xpu, 3, dtype> rnn_hidden_left;
    Tensor<xpu, 3, dtype> rnn_hidden_left_iy, rnn_hidden_left_oy, rnn_hidden_left_fy,
	  	  rnn_hidden_left_mcy, rnn_hidden_left_cy, rnn_hidden_left_my;
    Tensor<xpu, 3, dtype> rnn_hidden_right;
    Tensor<xpu, 3, dtype> rnn_hidden_right_iy, rnn_hidden_right_oy, rnn_hidden_right_fy,
	  	  rnn_hidden_right_mcy, rnn_hidden_right_cy, rnn_hidden_right_my;

    Tensor<xpu, 3, dtype> rnn_hidden_merge;

    vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> output;
    Tensor<xpu, 2, dtype> penul;

    Tensor<xpu, 3, dtype> wordprime, wordrepresent;

    static hash_set<int>::iterator it;

    //initialize
    wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
    wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

    input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

    rnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2 * _rnnHiddenSize), 0.0);

    project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);

    for (int idm = 0; idm < _poolmanners; idm++) {
      pool[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
    }


    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
    penul = NewTensor<xpu>(Shape2(1, _penultimate), 0.0);

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out

      const vector<int>& words = feature.words;

      _words.GetEmb(words[0], wordprime[idx]);

    }

    for (int idx = 0; idx < seq_size; idx++) {
      wordrepresent[idx] += wordprime[idx];
    }

    windowlized(wordrepresent, input, _wordcontext);

    _rnn_left.ComputeForwardScore(input, rnn_hidden_left_iy, rnn_hidden_left_oy,
  		  rnn_hidden_left_fy, rnn_hidden_left_mcy, rnn_hidden_left_cy, rnn_hidden_left_my,
			  rnn_hidden_left);
    _rnn_right.ComputeForwardScore(input, rnn_hidden_right_iy, rnn_hidden_right_oy,
  		  rnn_hidden_right_fy, rnn_hidden_right_mcy, rnn_hidden_right_cy, rnn_hidden_right_my,
			  rnn_hidden_right);

    for (int idx = 0; idx < seq_size; idx++) {
      concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);
    }

    // do we need a convolution? future work, currently needn't
    for (int idx = 0; idx < seq_size; idx++) {
      _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);
    }

    offset = 0;
    //before
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset]);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1]);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2]);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3]);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4]);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], poolmerge);

    //_olayer_linear.ComputeForwardScore(poolmerge, output);
    penulLayer.ComputeForwardScore(poolmerge, penul);
    _olayer_linear.ComputeForwardScore(penul, output);


    // decode algorithm
    int optLabel = softmax_predict(output, results);

    //release
    FreeSpace(&wordprime);
    FreeSpace(&wordrepresent);

    FreeSpace(&input);

	FreeSpace(&(rnn_hidden_left_iy));
	FreeSpace(&(rnn_hidden_left_oy));
	FreeSpace(&(rnn_hidden_left_fy));
	FreeSpace(&(rnn_hidden_left_mcy));
	FreeSpace(&(rnn_hidden_left_cy));
	FreeSpace(&(rnn_hidden_left_my));
	FreeSpace(&(rnn_hidden_left));

	FreeSpace(&(rnn_hidden_right_iy));
	FreeSpace(&(rnn_hidden_right_oy));
	FreeSpace(&(rnn_hidden_right_fy));
	FreeSpace(&(rnn_hidden_right_mcy));
	FreeSpace(&(rnn_hidden_right_cy));
	FreeSpace(&(rnn_hidden_right_my));
	FreeSpace(&(rnn_hidden_right));

    FreeSpace(&rnn_hidden_merge);

    FreeSpace(&project);

    for (int idm = 0; idm < _poolmanners; idm++) {
      FreeSpace(&(pool[idm]));
      FreeSpace(&(poolIndex[idm]));
    }



    FreeSpace(&poolmerge);
    FreeSpace(&output);
    FreeSpace(&penul);

    return optLabel;
  }

  dtype computeScore(const Example& example) {}

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    penulLayer.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {}

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 3, dtype> Wd, Tensor<xpu, 3, dtype> gradWd, const string& mark, int iter) {}

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {}

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

#endif /* SRC_PoolGRNNClassifier_H_ */
