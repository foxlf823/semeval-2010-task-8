
#ifndef SRC_PoolExGRNNClassifier3_H_
#define SRC_PoolExGRNNClassifier3_H_

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
class PoolExGRNNClassifier3 {
public:
	PoolExGRNNClassifier3() {
    _dropOut = 0.5;
  }
  ~PoolExGRNNClassifier3() {

  }

public:
  LookupTable<xpu> _words;
  LookupTable<xpu> _pos;
  LookupTable<xpu> _sst;
  LookupTable<xpu> _ner;


  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;

  int _token_representation_size;
  int _inputsize;
  int _hiddensize;
  int _rnnHiddenSize;

  //gated interaction part
  UniLayer<xpu> _represent_transform[3];
  //overall
  AttRecursiveGatedNN<xpu> _target_attention;

  UniLayer<xpu> _olayer_linear;
  UniLayer<xpu> _tanh_project;

  GRNN<xpu> _rnn_left;
  GRNN<xpu> _rnn_right;

  int _poolmanners;
  int _poolfunctions;
  int _targetdim;


  int _poolsize;
  int _gatedsize;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3, min, 4, std, 5, pro

  int _otherInputSize;
  int _channel;
  int _otherDim;

  Options options;

public:

  inline void init(const NRMat<dtype>& wordEmb, Options options) {
	 this->options = options;

    _wordcontext = options.wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    if(options.lossFunction == 1) {
        _labelSize = options.omitOther ? MAX_RELATION-1:MAX_RELATION;
    } else
    	_labelSize = MAX_RELATION;
    _token_representation_size = _wordDim;
    _poolfunctions = 5;
    _poolmanners = _poolfunctions * 3; //( before, middle , after) * (avg, max, min, std, pro)
    _inputsize = _wordwindow * _token_representation_size;
    // put other emb to the penultimate layer
    _channel = options.channelMode;
    _otherDim = options.otherEmbSize;
    _otherInputSize = 0;
	if((_channel & 2) == 2) {
		_otherInputSize += 2*_otherDim;
	}
	if((_channel & 4) == 4) {
		_otherInputSize += 2*_otherDim;
	}
	if((_channel & 8) == 8) {
		_otherInputSize += 2*_otherDim;
	}
	if((_channel & 16) == 16) {
		_otherInputSize += 2*_otherDim;
	}
	if((_channel & 32) == 32) {
		_otherInputSize += 2*_otherDim;
	}



    _hiddensize = options.wordEmbSize;
    _rnnHiddenSize = options.rnnHiddenSize;

    _targetdim = _hiddensize;

    _poolsize = _poolmanners * _hiddensize;
    _gatedsize = _targetdim;

    _words.initial(wordEmb);

    for (int idx = 0; idx < 3; idx++) {
      _represent_transform[idx].initial(_targetdim, _poolfunctions * _hiddensize, true, (idx + 1) * 100 + 60, 0);
    }

    _target_attention.initial(_targetdim, _targetdim, 100);

    _rnn_left.initial(_rnnHiddenSize, _inputsize, true, 10);
    _rnn_right.initial(_rnnHiddenSize, _inputsize, false, 40);

    _tanh_project.initial(_hiddensize, 2 * _rnnHiddenSize, true, 70, 0);
    _olayer_linear.initial(_labelSize, _poolsize + _gatedsize + _otherInputSize, false, 80, 2);

    _remove = 0;

    cout<<"PoolExGRNNClassifier3 initial"<<endl;
    if(options.lossFunction == 1)
    	cout<< "use rank loss"<<endl;
    else
    	cout<< "use softmax loss" <<endl;
    cout<<"use NER POS SST features of the two nouns in the penultimate layer"<<endl;
    cout<<"do recurrent on the dependency path"<<endl;
  }

  inline void release() {
    _words.release();
    _sst.release();
    _ner.release();
    _pos.release();

    _olayer_linear.release();
    _tanh_project.release();
    _rnn_left.release();
    _rnn_right.release();

    for (int idx = 0; idx < 5; idx++) {
      _represent_transform[idx].release();
    }

    _target_attention.release();

  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_before.size()+example.m_middle.size()+example.m_after.size();
      if(seq_size==0)
    	  continue;
      Tensor<xpu, 3, dtype> input, inputLoss;
      Tensor<xpu, 3, dtype> project, projectLoss;

      Tensor<xpu, 3, dtype> rnn_hidden_left, rnn_hidden_leftLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current;
      Tensor<xpu, 3, dtype> rnn_hidden_right, rnn_hidden_rightLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current;

      Tensor<xpu, 3, dtype> rnn_hidden_merge, rnn_hidden_mergeLoss;

      vector<Tensor<xpu, 2, dtype> > pool(_poolmanners), poolLoss(_poolmanners);
      vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

      Tensor<xpu, 2, dtype> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, dtype> gatedmerge, gatedmergeLoss;
      Tensor<xpu, 2, dtype> allmerge, allmergeLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;

      //gated interaction part
      Tensor<xpu, 2, dtype> input_span[3], input_spanLoss[3];
      Tensor<xpu, 2, dtype> reset_left, reset_right, interact_middle;
      Tensor<xpu, 2, dtype> update_left, update_right, update_interact;
      Tensor<xpu, 2, dtype> interact, interactLoss;

      Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, dtype> wordrepresent, wordrepresentLoss;

      Tensor<xpu, 2, dtype> otherFeature, otherFeatureLoss;
      Tensor<xpu, 2, dtype> nerprimeFormer, nerprimeFormerLoss, nerprimeFormerMask;
      Tensor<xpu, 2, dtype> nerprimeLatter, nerprimeLatterLoss, nerprimeLatterMask;
      Tensor<xpu, 2, dtype> posprimeFormer, posprimeFormerLoss, posprimeFormerMask;
      Tensor<xpu, 2, dtype> posprimeLatter, posprimeLatterLoss, posprimeLatterMask;
      Tensor<xpu, 2, dtype> sstprimeFormer, sstprimeFormerLoss, sstprimeFormerMask;
      Tensor<xpu, 2, dtype> sstprimeLatter, sstprimeLatterLoss, sstprimeLatterMask;

      hash_set<int> beforeIndex, middleIndex, afterIndex;
      Tensor<xpu, 2, dtype> beforerepresent, beforerepresentLoss;
	  Tensor<xpu, 2, dtype> middlerepresent, middlerepresentLoss;
	  Tensor<xpu, 2, dtype> afterrepresent, afterrepresentLoss;

      static hash_set<int>::iterator it;

      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 1.0);
      wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);
      wordrepresentLoss = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);


	  nerprimeFormer = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  nerprimeFormerLoss = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
      nerprimeFormerMask = NewTensor<xpu>(Shape2(1, _otherDim), 1.0);
	  nerprimeLatter = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  nerprimeLatterLoss = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
      nerprimeLatterMask = NewTensor<xpu>(Shape2(1, _otherDim), 1.0);
	  posprimeFormer = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  posprimeFormerLoss = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
      posprimeFormerMask = NewTensor<xpu>(Shape2(1, _otherDim), 1.0);
	  posprimeLatter = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  posprimeLatterLoss = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
      posprimeLatterMask = NewTensor<xpu>(Shape2(1, _otherDim), 1.0);
	  sstprimeFormer = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  sstprimeFormerLoss = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
      sstprimeFormerMask = NewTensor<xpu>(Shape2(1, _otherDim), 1.0);
	  sstprimeLatter = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  sstprimeLatterLoss = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
      sstprimeLatterMask = NewTensor<xpu>(Shape2(1, _otherDim), 1.0);
	  otherFeature = NewTensor<xpu>(Shape2(1, _otherInputSize), 0.0);
	  otherFeatureLoss = NewTensor<xpu>(Shape2(1, _otherInputSize), 0.0);


      input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);
      inputLoss = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);


      rnn_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_leftLoss = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

      rnn_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
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

      beforerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
      beforerepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

      middlerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
      middlerepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

	  afterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
      afterrepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

      for (int idm = 0; idm < 3; idm++) {
        input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
        input_spanLoss[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      }

      reset_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      reset_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      interact_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      update_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      update_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      update_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      interactLoss = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);


      poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      gatedmerge = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
      gatedmergeLoss = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
      allmerge = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize + _otherInputSize), 0.0);
      allmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize + _otherInputSize), 0.0);
      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      //forward propagation
      //input setting, and linear setting
      int idx = 0;
      for (int i = 0; i < example.m_before.size(); i++) {
    	  srand(iter * example_num + count * seq_size + idx);

          beforeIndex.insert(idx);

          _words.GetEmb(example.m_before[i], wordprime[idx]);

           //dropout
           dropoutcol(wordprimeMask[idx], _dropOut);
           wordprime[idx] = wordprime[idx] * wordprimeMask[idx];

    	  idx++;
      }

      for (int i = 0; i < example.m_middle.size(); i++) {
    	  srand(iter * example_num + count * seq_size + idx);

          middleIndex.insert(idx);

          _words.GetEmb(example.m_middle[i], wordprime[idx]);

           //dropout
           dropoutcol(wordprimeMask[idx], _dropOut);
           wordprime[idx] = wordprime[idx] * wordprimeMask[idx];

    	  idx++;
      }

      for (int i = 0; i < example.m_after.size(); i++) {
    	  srand(iter * example_num + count * seq_size + idx);

          afterIndex.insert(idx);

          _words.GetEmb(example.m_after[i], wordprime[idx]);

           //dropout
           dropoutcol(wordprimeMask[idx], _dropOut);
           wordprime[idx] = wordprime[idx] * wordprimeMask[idx];

    	  idx++;
      }


      for (int idx = 0; idx < seq_size; idx++) {
        wordrepresent[idx] += wordprime[idx];
      }

      windowlized(wordrepresent, input, _wordcontext);


      _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
      _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

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
        avgpool_forward(project, pool[offset], poolIndex[offset], beforeIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], beforeIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], beforeIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], beforeIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(project, pool[offset + 4], poolIndex[offset + 4], beforeIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], beforerepresent);

      offset += _poolfunctions;
      //middle
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(project, pool[offset], poolIndex[offset], middleIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], middleIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], middleIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], middleIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(project, pool[offset + 4], poolIndex[offset + 4], middleIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], middlerepresent);

	  offset +=  _poolfunctions;
      //after
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(project, pool[offset], poolIndex[offset], afterIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], afterIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], afterIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], afterIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(project, pool[offset + 4], poolIndex[offset + 4], afterIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], afterrepresent);

      _represent_transform[0].ComputeForwardScore(beforerepresent, input_span[0]);
      _represent_transform[1].ComputeForwardScore(middlerepresent, input_span[1]);
      _represent_transform[2].ComputeForwardScore(afterrepresent, input_span[2]);


      _target_attention.ComputeForwardScore(input_span[0], input_span[2], input_span[1],
          reset_left, reset_right, interact_middle,
          update_left, update_right, update_interact,
          interact);


      concat(beforerepresent, middlerepresent, afterrepresent, poolmerge);
      gatedmerge += interact;


      // put other features into the penultimate layer
      vector<Tensor<xpu, 2, dtype> > v_otherInput;
		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
			  _ner.GetEmb(example.m_features[example.formerTkEnd].ner, nerprimeFormer);
			  _ner.GetEmb(example.m_features[example.latterTkEnd].ner, nerprimeLatter);

		        //dropout
		        dropoutcol(nerprimeFormerMask, _dropOut);
		        nerprimeFormer = nerprimeFormer * nerprimeFormerMask;
		        dropoutcol(nerprimeLatterMask, _dropOut);
		        nerprimeLatter = nerprimeLatter * nerprimeLatterMask;

			  v_otherInput.push_back(nerprimeFormer);
			  v_otherInput.push_back(nerprimeLatter);
		}
		if((_channel & 16) == 16) {
			_pos.GetEmb(example.m_features[example.formerTkEnd].pos, posprimeFormer);
			_pos.GetEmb(example.m_features[example.latterTkEnd].pos, posprimeLatter);

	        //dropout
	        dropoutcol(posprimeFormerMask, _dropOut);
	        posprimeFormer = posprimeFormer * posprimeFormerMask;
	        dropoutcol(posprimeLatterMask, _dropOut);
	        posprimeLatter = posprimeLatter * posprimeLatterMask;

			  v_otherInput.push_back(posprimeFormer);
			  v_otherInput.push_back(posprimeLatter);
		}
		if((_channel & 32) == 32) {
			_sst.GetEmb(example.m_features[example.formerTkEnd].sst, sstprimeFormer);
			_sst.GetEmb(example.m_features[example.latterTkEnd].sst, sstprimeLatter);

	        //dropout
	        dropoutcol(sstprimeFormerMask, _dropOut);
	        sstprimeFormer = sstprimeFormer * sstprimeFormerMask;
	        dropoutcol(sstprimeLatterMask, _dropOut);
	        sstprimeLatter = sstprimeLatter * sstprimeLatterMask;

			  v_otherInput.push_back(sstprimeFormer);
			  v_otherInput.push_back(sstprimeLatter);
		}
		concat(v_otherInput, otherFeature);


      concat(poolmerge, gatedmerge, otherFeature, allmerge);

      _olayer_linear.ComputeForwardScore(allmerge, output);

      // get delta for each output
      if(options.lossFunction == 1) {
          int goldLabel = example.goldLabel;
          int negLabel = -1;
          int optLabel = -1;
          for(int i=0;i<_labelSize;i++) {
        	  if(optLabel<0 || output[0][i]>output[0][optLabel])
        		  optLabel = i;

        	  if((i!=goldLabel) && (negLabel<0 || output[0][i]>output[0][negLabel]))
        		  negLabel  = i;
          }

          if(options.omitOther && goldLabel==OTHER_LABEL) {
        	  cost += log(1.0+exp(options.gamma*(options.mNeg+output[0][negLabel])));
          } else {
        	  cost += log(1.0+exp(options.gamma*(options.mPos-output[0][goldLabel]))) +
        			  log(1.0+exp(options.gamma*(options.mNeg+output[0][negLabel])));
          }

          if (optLabel == goldLabel)
        	  _eval.correct_label_count++;
          _eval.overall_label_count++;

          for(int i=0;i<_labelSize;i++) {
        	  double delta = 0;
        	  if(options.omitOther && goldLabel == OTHER_LABEL) {
        		  if(i==negLabel)
        			  delta = options.gamma*exp(options.gamma*(options.mNeg+output[0][negLabel]))/(example_num*(1.0+exp(options.gamma*(options.mNeg+output[0][negLabel]))));
        		  else
        			  delta = 0;
        	  } else {
        		  if(i==goldLabel)
        			  delta = -options.gamma*exp(options.gamma*(options.mPos-output[0][goldLabel]))/(example_num*(1.0+exp(options.gamma*(options.mPos-output[0][goldLabel]))));
        		  else if(i==negLabel)
        			  delta = options.gamma*exp(options.gamma*(options.mNeg+output[0][negLabel]))/(example_num*(1.0+exp(options.gamma*(options.mNeg+output[0][negLabel]))));
    			  else
    				  delta = 0;
        	  }
        	  outputLoss[0][i] += delta;
          }
      } else
    	  cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      _olayer_linear.ComputeBackwardLoss(allmerge, output, outputLoss, allmergeLoss);

      unconcat(poolmergeLoss, gatedmergeLoss, otherFeatureLoss, allmergeLoss);

      // put the loss back into other features
	  vector<Tensor<xpu, 2, dtype> > v_otherInputLoss;

		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
			v_otherInputLoss.push_back(nerprimeFormerLoss);
			v_otherInputLoss.push_back(nerprimeLatterLoss);
		}
		if((_channel & 16) == 16) {
			v_otherInputLoss.push_back(posprimeFormerLoss);
			v_otherInputLoss.push_back(posprimeLatterLoss);
		}
		if((_channel & 32) == 32) {
			v_otherInputLoss.push_back(sstprimeFormerLoss);
			v_otherInputLoss.push_back(sstprimeLatterLoss);
		}

		unconcat(v_otherInputLoss, otherFeatureLoss);



      interactLoss += gatedmergeLoss;
      unconcat(beforerepresentLoss, middlerepresentLoss, afterrepresentLoss, poolmergeLoss);


      _target_attention.ComputeBackwardLoss(input_span[0], input_span[2], input_span[1],
          reset_left, reset_right, interact_middle,
          update_left, update_right, update_interact,
          interact, interactLoss,
          input_spanLoss[0], input_spanLoss[2], input_spanLoss[1]);


      _represent_transform[0].ComputeBackwardLoss(beforerepresent, input_span[0], input_spanLoss[0], beforerepresentLoss);
      _represent_transform[1].ComputeBackwardLoss(middlerepresent, input_span[1], input_spanLoss[1], middlerepresentLoss);
      _represent_transform[2].ComputeBackwardLoss(afterrepresent, input_span[2], input_spanLoss[2], afterrepresentLoss);

      offset = 0;
      //before
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], beforerepresentLoss);

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

      offset +=  _poolfunctions;
      //middle
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], middlerepresentLoss);

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

	  offset +=  _poolfunctions;
      //after
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], afterrepresentLoss);

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

      _rnn_left.ComputeBackwardLoss(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left, rnn_hidden_leftLoss, inputLoss);
      _rnn_right.ComputeBackwardLoss(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right, rnn_hidden_rightLoss, inputLoss);



      // word context
      windowlized_backward(wordrepresentLoss, inputLoss, _wordcontext);

      for (int idx = 0; idx < seq_size; idx++) {
        wordprimeLoss[idx] += wordrepresentLoss[idx];
      }

      if (_words.bEmbFineTune()) {
          int idx = 0;
          for (int i = 0; i < example.m_before.size(); i++) {
              wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
              _words.EmbLoss(example.m_before[i], wordprimeLoss[idx]);

        	  idx++;
          }

          for (int i = 0; i < example.m_middle.size(); i++) {
              wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
              _words.EmbLoss(example.m_middle[i], wordprimeLoss[idx]);

        	  idx++;
          }

          for (int i = 0; i < example.m_after.size(); i++) {
              wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
              _words.EmbLoss(example.m_after[i], wordprimeLoss[idx]);

        	  idx++;
          }

      }

		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
	          nerprimeFormerLoss = nerprimeFormerLoss * nerprimeFormerMask;
	          nerprimeLatterLoss = nerprimeLatterLoss * nerprimeLatterMask;

			_ner.EmbLoss(example.m_features[example.formerTkEnd].ner, nerprimeFormerLoss);
			_ner.EmbLoss(example.m_features[example.latterTkEnd].ner, nerprimeLatterLoss);
		}
		if((_channel & 16) == 16) {
	          posprimeFormerLoss = posprimeFormerLoss * posprimeFormerMask;
	          posprimeLatterLoss = posprimeLatterLoss * posprimeLatterMask;

			_pos.EmbLoss(example.m_features[example.formerTkEnd].pos, posprimeFormerLoss);
			_pos.EmbLoss(example.m_features[example.latterTkEnd].pos, posprimeLatterLoss);
		}
		if((_channel & 32) == 32) {
	          sstprimeFormerLoss = sstprimeFormerLoss * sstprimeFormerMask;
	          sstprimeLatterLoss = sstprimeLatterLoss * sstprimeLatterMask;

			_sst.EmbLoss(example.m_features[example.formerTkEnd].sst, sstprimeFormerLoss);
			_sst.EmbLoss(example.m_features[example.latterTkEnd].sst, sstprimeLatterLoss);
		}

      //release
      FreeSpace(&wordprime);
      FreeSpace(&wordprimeLoss);
      FreeSpace(&wordprimeMask);
      FreeSpace(&wordrepresent);
      FreeSpace(&wordrepresentLoss);

      FreeSpace(&nerprimeFormer);
      FreeSpace(&nerprimeFormerLoss);
      FreeSpace(&nerprimeFormerMask);
      FreeSpace(&nerprimeLatter);
      FreeSpace(&nerprimeLatterLoss);
      FreeSpace(&nerprimeLatterMask);
      FreeSpace(&posprimeFormer);
      FreeSpace(&posprimeFormerLoss);
      FreeSpace(&posprimeFormerMask);
      FreeSpace(&posprimeLatter);
      FreeSpace(&posprimeLatterLoss);
      FreeSpace(&posprimeLatterMask);
      FreeSpace(&sstprimeFormer);
      FreeSpace(&sstprimeFormerLoss);
      FreeSpace(&sstprimeFormerMask);
      FreeSpace(&sstprimeLatter);
      FreeSpace(&sstprimeLatterLoss);
      FreeSpace(&sstprimeLatterMask);
      FreeSpace(&otherFeature);
      FreeSpace(&otherFeatureLoss);

      FreeSpace(&input);
      FreeSpace(&inputLoss);


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

      FreeSpace(&rnn_hidden_merge);
      FreeSpace(&rnn_hidden_mergeLoss);

      FreeSpace(&project);
      FreeSpace(&projectLoss);

      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[idm]));
        FreeSpace(&(poolLoss[idm]));
        FreeSpace(&(poolIndex[idm]));
      }

      for (int idm = 0; idm < 3; idm++) {
        FreeSpace(&(input_span[idm]));
        FreeSpace(&(input_spanLoss[idm]));
      }

      FreeSpace(&reset_left);
      FreeSpace(&reset_right);
      FreeSpace(&interact_middle);
      FreeSpace(&update_left);
      FreeSpace(&update_right);
      FreeSpace(&update_interact);
      FreeSpace(&(interact));
      FreeSpace(&(interactLoss));


      FreeSpace(&beforerepresent);
      FreeSpace(&beforerepresentLoss);
      FreeSpace(&middlerepresent);
      FreeSpace(&middlerepresentLoss);
	  FreeSpace(&afterrepresent);
      FreeSpace(&afterrepresentLoss);

      FreeSpace(&poolmerge);
      FreeSpace(&poolmergeLoss);
      FreeSpace(&gatedmerge);
      FreeSpace(&gatedmergeLoss);
      FreeSpace(&allmerge);
      FreeSpace(&allmergeLoss);
      FreeSpace(&output);
      FreeSpace(&outputLoss);

    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  int predict(const Example& example, vector<dtype>& results) {

    int seq_size = example.m_before.size()+example.m_middle.size()+example.m_after.size();
    if(seq_size==0)
  	  return OTHER_LABEL-1;

    int offset = 0;

    Tensor<xpu, 3, dtype> input;
    Tensor<xpu, 3, dtype> project;

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

    Tensor<xpu, 3, dtype> rnn_hidden_merge;

    vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> gatedmerge;
    Tensor<xpu, 2, dtype> allmerge;
    Tensor<xpu, 2, dtype> output;

    //gated interaction part
    Tensor<xpu, 2, dtype> input_span[3];
    Tensor<xpu, 2, dtype> reset_left, reset_right, interact_middle;
    Tensor<xpu, 2, dtype> update_left, update_right, update_interact;
    Tensor<xpu, 2, dtype> interact;


    Tensor<xpu, 3, dtype> wordprime, wordrepresent;

    Tensor<xpu, 2, dtype> otherFeature;
    Tensor<xpu, 2, dtype> nerprimeFormer;
    Tensor<xpu, 2, dtype> nerprimeLatter;
    Tensor<xpu, 2, dtype> posprimeFormer;
    Tensor<xpu, 2, dtype> posprimeLatter;
    Tensor<xpu, 2, dtype> sstprimeFormer;
    Tensor<xpu, 2, dtype> sstprimeLatter;

    hash_set<int> beforeIndex, middleIndex, afterIndex;
    Tensor<xpu, 2, dtype> beforerepresent;
	  Tensor<xpu, 2, dtype> middlerepresent;
	  Tensor<xpu, 2, dtype> afterrepresent;

    static hash_set<int>::iterator it;

    //initialize
    wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
    wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

	  nerprimeFormer = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  nerprimeLatter = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  posprimeFormer = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  posprimeLatter = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  sstprimeFormer = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  sstprimeLatter = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
	  otherFeature = NewTensor<xpu>(Shape2(1, _otherInputSize), 0.0);


    input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

    rnn_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2 * _rnnHiddenSize), 0.0);

    project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);

    for (int idm = 0; idm < _poolmanners; idm++) {
      pool[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
    }

    beforerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    middlerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
	  afterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);


    for (int idm = 0; idm < 3; idm++) {
      input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    }

    reset_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    reset_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    interact_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    update_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    update_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    update_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);



    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
    gatedmerge = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
    allmerge = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize + _otherInputSize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

    //forward propagation
    //input setting, and linear setting
    int idx = 0;
    for (int i = 0; i < example.m_before.size(); i++) {

        beforeIndex.insert(idx);

        _words.GetEmb(example.m_before[i], wordprime[idx]);

  	  idx++;
    }

    for (int i = 0; i < example.m_middle.size(); i++) {

        middleIndex.insert(idx);

        _words.GetEmb(example.m_middle[i], wordprime[idx]);


  	  idx++;
    }

    for (int i = 0; i < example.m_after.size(); i++) {

        afterIndex.insert(idx);

        _words.GetEmb(example.m_after[i], wordprime[idx]);


  	  idx++;
    }

    for (int idx = 0; idx < seq_size; idx++) {
      wordrepresent[idx] += wordprime[idx];
    }

    windowlized(wordrepresent, input, _wordcontext);

    _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
    _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

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
      avgpool_forward(project, pool[offset], poolIndex[offset], beforeIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], beforeIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], beforeIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], beforeIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], beforeIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], beforerepresent);


    offset +=  _poolfunctions;
    //middle
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], middleIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], middleIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], middleIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], middleIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], middleIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], middlerepresent);


	  offset += _poolfunctions;
    //after
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], afterIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], afterIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], afterIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], afterIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], afterIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], afterrepresent);


    _represent_transform[0].ComputeForwardScore(beforerepresent, input_span[0]);
    _represent_transform[1].ComputeForwardScore(middlerepresent, input_span[1]);
    _represent_transform[2].ComputeForwardScore(afterrepresent, input_span[2]);

    _target_attention.ComputeForwardScore(input_span[0], input_span[2], input_span[1],
        reset_left, reset_right, interact_middle,
        update_left, update_right, update_interact,
        interact);




    concat(beforerepresent, middlerepresent, afterrepresent, poolmerge);
    gatedmerge += interact;

    // put other features into the penultimate layer
     vector<Tensor<xpu, 2, dtype> > v_otherInput;
		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
			  _ner.GetEmb(example.m_features[example.formerTkEnd].ner, nerprimeFormer);
			  _ner.GetEmb(example.m_features[example.latterTkEnd].ner, nerprimeLatter);
			  v_otherInput.push_back(nerprimeFormer);
			  v_otherInput.push_back(nerprimeLatter);
		}
		if((_channel & 16) == 16) {
			_pos.GetEmb(example.m_features[example.formerTkEnd].pos, posprimeFormer);
			_pos.GetEmb(example.m_features[example.latterTkEnd].pos, posprimeLatter);
			  v_otherInput.push_back(posprimeFormer);
			  v_otherInput.push_back(posprimeLatter);
		}
		if((_channel & 32) == 32) {
			_sst.GetEmb(example.m_features[example.formerTkEnd].sst, sstprimeFormer);
			_sst.GetEmb(example.m_features[example.latterTkEnd].sst, sstprimeLatter);
			  v_otherInput.push_back(sstprimeFormer);
			  v_otherInput.push_back(sstprimeLatter);
		}
		concat(v_otherInput, otherFeature);

    concat(poolmerge, gatedmerge, otherFeature, allmerge);


    _olayer_linear.ComputeForwardScore(allmerge, output);

    // decode algorithm
    int optLabel = -1;
    if(options.lossFunction == 1) {
        for(int i=0;i<output.size(1);i++)
        	results.push_back(output[0][i]);
    } else
    	optLabel = softmax_predict(output, results);

    //release
    FreeSpace(&wordprime);
    FreeSpace(&wordrepresent);

    FreeSpace(&nerprimeFormer);
    FreeSpace(&nerprimeLatter);
    FreeSpace(&posprimeFormer);
    FreeSpace(&posprimeLatter);
    FreeSpace(&sstprimeFormer);
    FreeSpace(&sstprimeLatter);
    FreeSpace(&otherFeature);

    FreeSpace(&input);

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

    FreeSpace(&rnn_hidden_merge);

    FreeSpace(&project);

    for (int idm = 0; idm < _poolmanners; idm++) {
      FreeSpace(&(pool[idm]));
      FreeSpace(&(poolIndex[idm]));
    }

    for (int idm = 0; idm < 3; idm++) {
      FreeSpace(&(input_span[idm]));
    }


    FreeSpace(&reset_left);
    FreeSpace(&reset_right);
    FreeSpace(&interact_middle);
    FreeSpace(&update_left);
    FreeSpace(&update_right);
    FreeSpace(&update_interact);
    FreeSpace(&(interact));


    FreeSpace(&beforerepresent);
    FreeSpace(&middlerepresent);
	  FreeSpace(&afterrepresent);

    FreeSpace(&poolmerge);
    FreeSpace(&gatedmerge);
    FreeSpace(&allmerge);
    FreeSpace(&output);

    return optLabel;
  }

  dtype computeScore(const Example& example) {}

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {

    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _target_attention.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    for (int idx = 0; idx < 3; idx++) {
      _represent_transform[idx].updateAdaGrad(nnRegular, adaAlpha, adaEps);
    }

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sst.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _ner.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _pos.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {}

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 3, dtype> Wd, Tensor<xpu, 3, dtype> gradWd, const string& mark, int iter) {}

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {}

  void checkgrads(const vector<Example>& examples, int iter) {

    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

    checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

    checkgrad(examples, _rnn_left._rnn_update._WL, _rnn_left._rnn_update._gradWL, "_rnn_left._rnn_update._WL", iter);
    checkgrad(examples, _rnn_left._rnn_update._WR, _rnn_left._rnn_update._gradWR, "_rnn_left._rnn_update._WR", iter);
    checkgrad(examples, _rnn_left._rnn_update._b, _rnn_left._rnn_update._gradb, "_rnn_left._rnn_update._b", iter);
    checkgrad(examples, _rnn_left._rnn_reset._WL, _rnn_left._rnn_reset._gradWL, "_rnn_left._rnn_reset._WL", iter);
    checkgrad(examples, _rnn_left._rnn_reset._WR, _rnn_left._rnn_reset._gradWR, "_rnn_left._rnn_reset._WR", iter);
    checkgrad(examples, _rnn_left._rnn_reset._b, _rnn_left._rnn_reset._gradb, "_rnn_left._rnn_reset._b", iter);
    checkgrad(examples, _rnn_left._rnn._WL, _rnn_left._rnn._gradWL, "_rnn_left._rnn._WL", iter);
    checkgrad(examples, _rnn_left._rnn._WR, _rnn_left._rnn._gradWR, "_rnn_left._rnn._WR", iter);
    checkgrad(examples, _rnn_left._rnn._b, _rnn_left._rnn._gradb, "_rnn_left._rnn._b", iter);

    checkgrad(examples, _rnn_right._rnn_update._WL, _rnn_right._rnn_update._gradWL, "_rnn_right._rnn_update._WL", iter);
    checkgrad(examples, _rnn_right._rnn_update._WR, _rnn_right._rnn_update._gradWR, "_rnn_right._rnn_update._WR", iter);
    checkgrad(examples, _rnn_right._rnn_update._b, _rnn_right._rnn_update._gradb, "_rnn_right._rnn_update._b", iter);
    checkgrad(examples, _rnn_right._rnn_reset._WL, _rnn_right._rnn_reset._gradWL, "_rnn_right._rnn_reset._WL", iter);
    checkgrad(examples, _rnn_right._rnn_reset._WR, _rnn_right._rnn_reset._gradWR, "_rnn_right._rnn_reset._WR", iter);
    checkgrad(examples, _rnn_right._rnn_reset._b, _rnn_right._rnn_reset._gradb, "_rnn_right._rnn_reset._b", iter);
    checkgrad(examples, _rnn_right._rnn._WL, _rnn_right._rnn._gradWL, "_rnn_right._rnn._WL", iter);
    checkgrad(examples, _rnn_right._rnn._WR, _rnn_right._rnn._gradWR, "_rnn_right._rnn._WR", iter);
    checkgrad(examples, _rnn_right._rnn._b, _rnn_right._rnn._gradb, "_rnn_right._rnn._b", iter);

    checkgrad(examples, _target_attention._reset_left._WL, _target_attention._reset_left._gradWL, "_target_attention._reset_left._WL", iter);
    checkgrad(examples, _target_attention._reset_left._WR, _target_attention._reset_left._gradWR, "_target_attention._reset_left._WR", iter);
    checkgrad(examples, _target_attention._reset_left._b, _target_attention._reset_left._gradb, "_target_attention._reset_left._b", iter);

    checkgrad(examples, _target_attention. _reset_right._WL, _target_attention. _reset_right._gradWL, "_target_attention. _reset_right._WL", iter);
    checkgrad(examples, _target_attention. _reset_right._WR, _target_attention. _reset_right._gradWR, "_target_attention. _reset_right._WR", iter);
    checkgrad(examples, _target_attention. _reset_right._b, _target_attention. _reset_right._gradb, "_target_attention. _reset_right._b", iter);

    checkgrad(examples, _target_attention._update_left._WL, _target_attention._update_left._gradWL, "_target_attention._update_left._WL", iter);
    checkgrad(examples, _target_attention._update_left._WR, _target_attention._update_left._gradWR, "_target_attention._update_left._WR", iter);
    checkgrad(examples, _target_attention._update_left._b, _target_attention._update_left._gradb, "_target_attention._update_left._b", iter);

    checkgrad(examples, _target_attention._update_right._WL, _target_attention._update_right._gradWL, "_target_attention._update_right._WL", iter);
    checkgrad(examples, _target_attention._update_right._WR, _target_attention._update_right._gradWR, "_target_attention._update_right._WR", iter);
    checkgrad(examples, _target_attention._update_right._b, _target_attention._update_right._gradb, "_target_attention._update_right._b", iter);

    checkgrad(examples, _target_attention._update_tilde._WL, _target_attention._update_tilde._gradWL, "_target_attention._update_tilde._WL", iter);
    checkgrad(examples, _target_attention._update_tilde._WR, _target_attention._update_tilde._gradWR, "_target_attention._update_tilde._WR", iter);
    checkgrad(examples, _target_attention._update_tilde._b, _target_attention._update_tilde._gradb, "_target_attention._update_tilde._b", iter);

    checkgrad(examples, _target_attention._recursive_tilde._WL, _target_attention._recursive_tilde._gradWL, "_target_attention._recursive_tilde._WL", iter);
    checkgrad(examples, _target_attention._recursive_tilde._WR, _target_attention._recursive_tilde._gradWR, "_target_attention._recursive_tilde._WR", iter);
    checkgrad(examples, _target_attention._recursive_tilde._b, _target_attention._recursive_tilde._gradb, "_target_attention._recursive_tilde._b", iter);

    for (int idx = 0; idx < 3; idx++) {
      stringstream ssposition;
      ssposition << "[" << idx << "]";

      checkgrad(examples, _represent_transform[idx]._W, _represent_transform[idx]._gradW, "_represent_transform" + ssposition.str() + "._W", iter);
      checkgrad(examples, _represent_transform[idx]._b, _represent_transform[idx]._gradb, "_represent_transform" + ssposition.str() + "._b", iter);
    }

    checkgrad(examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);

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

#endif /* SRC_PoolExGRNNClassifier_H_ */
