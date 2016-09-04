
#ifndef SRC_PoolExGRNNClassifier_5TUPLE_H_
#define SRC_PoolExGRNNClassifier_5TUPLE_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "N3L.h"
#include <cmath>

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// use rank loss and can omit other
template<typename xpu>
class PoolExGRNNClassifier_5tuple {
public:
	PoolExGRNNClassifier_5tuple() {
    _dropOut = 0.5;
  }
  ~PoolExGRNNClassifier_5tuple() {

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
  UniLayer<xpu> _represent_transform[5];
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

  Options options;
  int _otherInputSize;
  int _channel;
  int _otherDim;
  int _omitOther;

public:


  inline void init(const NRMat<dtype>& wordEmb, int wordcontext, int rnnHiddenSize,
		  int hiddensize, int labelSize, int channel, int otherDim, Options options) {
	  this->options = options;
    _wordcontext = wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _omitOther = options.omitOther;
    _labelSize = _omitOther ? labelSize-1:labelSize;
    _token_representation_size = _wordDim;
    _poolfunctions = 5;
    _poolmanners = _poolfunctions * 5; //( before, former, middle , latter, after) * (avg, max, min, std, pro)
    _inputsize = _wordwindow * _token_representation_size;
    // put other emb to the input of rnn
    _channel = channel;
    _otherDim = otherDim;
    _otherInputSize = 0;
	if((_channel & 2) == 2) {
		_otherInputSize += _otherDim;
	}
	if((_channel & 4) == 4) {
		_otherInputSize += _otherDim;
	}
	if((_channel & 8) == 8) {
		_otherInputSize += _otherDim;
	}
	if((_channel & 16) == 16) {
		_otherInputSize += _otherDim;
	}
	if((_channel & 32) == 32) {
		_otherInputSize += _otherDim;
	}



    _hiddensize = hiddensize;
    _rnnHiddenSize = rnnHiddenSize;

    _targetdim = _hiddensize;

    _poolsize = _poolmanners * _hiddensize;
    _gatedsize = _targetdim;

    _words.initial(wordEmb);

    for (int idx = 0; idx < 5; idx++) {
      _represent_transform[idx].initial(_targetdim, _poolfunctions * _hiddensize, true, (idx + 1) * 100 + 60, 0);
    }

    _target_attention.initial(_targetdim, _targetdim, 100);

    _rnn_left.initial(_rnnHiddenSize, _inputsize+_otherInputSize, true, 10);
    _rnn_right.initial(_rnnHiddenSize, _inputsize+_otherInputSize, false, 40);

    _tanh_project.initial(_hiddensize, 2 * _rnnHiddenSize, true, 70, 0);
    _olayer_linear.initial(_labelSize, _poolsize + 2*_gatedsize, false, 80, 2);

    _remove = 0;

    cout<<"PoolExGRNNClassifier_5tuple initial"<<endl;
  }

  inline void release() {
    _words.release();
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

      int seq_size = example.m_features.size();

      Tensor<xpu, 3, dtype> input, inputLoss;
      Tensor<xpu, 3, dtype> inputAndOther, inputAndOtherLoss;
      Tensor<xpu, 3, dtype> project, projectLoss;

      Tensor<xpu, 3, dtype> rnn_hidden_left, rnn_hidden_leftLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current;
      Tensor<xpu, 3, dtype> rnn_hidden_right, rnn_hidden_rightLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current;

      Tensor<xpu, 3, dtype> rnn_hidden_merge, rnn_hidden_mergeLoss;

      vector<Tensor<xpu, 2, dtype> > pool(_poolmanners), poolLoss(_poolmanners);
      vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

      Tensor<xpu, 2, dtype> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, dtype> gatedmergeFormer, gatedmergeLossFormer;
      Tensor<xpu, 2, dtype> gatedmergeLatter, gatedmergeLossLatter;
      Tensor<xpu, 2, dtype> allmerge, allmergeLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;

      //gated interaction part
      Tensor<xpu, 2, dtype> former_input_span[3], former_input_spanLoss[3];
      Tensor<xpu, 2, dtype> former_reset_left, former_reset_right, former_interact_middle;
      Tensor<xpu, 2, dtype> former_update_left, former_update_right, former_update_interact;
      Tensor<xpu, 2, dtype> former_interact, former_interactLoss;

      Tensor<xpu, 2, dtype> latter_input_span[3], latter_input_spanLoss[3];
      Tensor<xpu, 2, dtype> latter_reset_left, latter_reset_right, latter_interact_middle;
      Tensor<xpu, 2, dtype> latter_update_left, latter_update_right, latter_update_interact;
      Tensor<xpu, 2, dtype> latter_interact, latter_interactLoss;

      Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, dtype> wordrepresent, wordrepresentLoss;

      Tensor<xpu, 3, dtype> nerprime, nerprimeLoss;
      Tensor<xpu, 3, dtype> posprime, posprimeLoss;
      Tensor<xpu, 3, dtype> sstprime, sstprimeLoss;

      hash_set<int> beforeIndex, formerIndex, middleIndex, latterIndex, afterIndex;
      Tensor<xpu, 2, dtype> beforerepresent, beforerepresentLoss;
      Tensor<xpu, 2, dtype> formerrepresent, formerrepresentLoss;
	  Tensor<xpu, 2, dtype> middlerepresent, middlerepresentLoss;
      Tensor<xpu, 2, dtype> latterrepresent, latterrepresentLoss;
	  Tensor<xpu, 2, dtype> afterrepresent, afterrepresentLoss;

      static hash_set<int>::iterator it;

      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 1.0);
      wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);
      wordrepresentLoss = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);


	  nerprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
	  nerprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
	  posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
	  posprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
	  sstprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
	  sstprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);



      input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);
      inputLoss = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);
      inputAndOther = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize+_otherInputSize), 0.0);
      inputAndOtherLoss = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize+_otherInputSize), 0.0);

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

      formerrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
      formerrepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

      middlerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
      middlerepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

	  latterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
      latterrepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

	  afterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
      afterrepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

      for (int idm = 0; idm < 3; idm++) {
        former_input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
        former_input_spanLoss[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      }

      former_reset_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      former_reset_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      former_interact_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      former_update_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      former_update_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      former_update_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      former_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      former_interactLoss = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);

      for (int idm = 0; idm < 3; idm++) {
        latter_input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
        latter_input_spanLoss[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      }

      latter_reset_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      latter_reset_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      latter_interact_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      latter_update_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      latter_update_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      latter_update_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      latter_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      latter_interactLoss = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);

      poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      gatedmergeFormer = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
      gatedmergeLossFormer = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
      gatedmergeLatter = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
      gatedmergeLossLatter = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
      allmerge = NewTensor<xpu>(Shape2(1, _poolsize + 2*_gatedsize), 0.0);
      allmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize + 2*_gatedsize), 0.0);
      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      //forward propagation
      //input setting, and linear setting
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);

        const vector<int>& words = feature.words;
        if (idx < example.formerTkBegin) {
          beforeIndex.insert(idx);
        } else if(idx >= example.formerTkBegin && idx <= example.formerTkEnd) {
			formerIndex.insert(idx);
		} else if (idx >= example.latterTkBegin && idx <= example.latterTkEnd) {
          latterIndex.insert(idx);
        } else if (idx > example.latterTkEnd) {
          afterIndex.insert(idx);
        } else {
          middleIndex.insert(idx);
        }

       _words.GetEmb(words[0], wordprime[idx]);

		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
			_ner.GetEmb(feature.ner, nerprime[idx]);
		}
		if((_channel & 16) == 16) {
			_pos.GetEmb(feature.pos, posprime[idx]);
			//cout<<feature.pos<<" ";
		}
		if((_channel & 32) == 32) {
			_sst.GetEmb(feature.sst, sstprime[idx]);
		}

        //dropout
        dropoutcol(wordprimeMask[idx], _dropOut);
        wordprime[idx] = wordprime[idx] * wordprimeMask[idx];
      }

      for (int idx = 0; idx < seq_size; idx++) {
        wordrepresent[idx] += wordprime[idx];
      }

      windowlized(wordrepresent, input, _wordcontext);


      // put other emb to the input of rnn
      for (int idx = 0; idx < seq_size; idx++) {
    	  vector<Tensor<xpu, 2, dtype> > v_otherInput;

    	  v_otherInput.push_back(input[idx]);

    		if((_channel & 2) == 2) {

			}
			if((_channel & 4) == 4) {

			}
			if((_channel & 8) == 8) {
				v_otherInput.push_back(nerprime[idx]);
			}
			if((_channel & 16) == 16) {
				v_otherInput.push_back(posprime[idx]);
			}
			if((_channel & 32) == 32) {
				v_otherInput.push_back(sstprime[idx]);
			}

			concat(v_otherInput, inputAndOther[idx]);

      }


      _rnn_left.ComputeForwardScore(inputAndOther, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
      _rnn_right.ComputeForwardScore(inputAndOther, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

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

      offset = _poolfunctions;
      //former
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(project, pool[offset], poolIndex[offset], formerIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], formerIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], formerIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], formerIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(project, pool[offset + 4], poolIndex[offset + 4], formerIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], formerrepresent);

      offset = 2 * _poolfunctions;
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

	  offset = 3 * _poolfunctions;
      //latter
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(project, pool[offset], poolIndex[offset], latterIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], latterIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], latterIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], latterIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(project, pool[offset + 4], poolIndex[offset + 4], latterIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], latterrepresent);

	  offset = 4 * _poolfunctions;
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

      _represent_transform[0].ComputeForwardScore(beforerepresent, former_input_span[0]);
      _represent_transform[1].ComputeForwardScore(formerrepresent, former_input_span[1]);
      _represent_transform[2].ComputeForwardScore(middlerepresent, former_input_span[2]);

      _represent_transform[2].ComputeForwardScore(middlerepresent, latter_input_span[0]);
      _represent_transform[3].ComputeForwardScore(latterrepresent, latter_input_span[1]);
      _represent_transform[4].ComputeForwardScore(afterrepresent, latter_input_span[2]);

      _target_attention.ComputeForwardScore(former_input_span[0], former_input_span[2], former_input_span[1],
          former_reset_left, former_reset_right, former_interact_middle,
          former_update_left, former_update_right, former_update_interact,
          former_interact);

      _target_attention.ComputeForwardScore(latter_input_span[0], latter_input_span[2], latter_input_span[1],
          latter_reset_left, latter_reset_right, latter_interact_middle,
          latter_update_left, latter_update_right, latter_update_interact,
          latter_interact);


      concat(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent, poolmerge);
      gatedmergeFormer += former_interact;
      gatedmergeLatter += latter_interact;
      concat(poolmerge, gatedmergeFormer, gatedmergeLatter, allmerge);

      _olayer_linear.ComputeForwardScore(allmerge, output);
      //cout<<"1"<<endl;
      // get delta for each output
      //cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);
      int goldLabel = example.goldLabel;
      int negLabel = -1;
      int optLabel = -1;
      for(int i=0;i<_labelSize;i++) {
    	  if(optLabel<0 || output[0][i]>output[0][optLabel])
    		  optLabel = i;

    	  if((i!=goldLabel) && (negLabel<0 || output[0][i]>output[0][negLabel]))
    		  negLabel  = i;
      }

      if(_omitOther && goldLabel==OTHER_LABEL) {
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

      //cout<<"2"<<endl;
      // loss backward propagation
      _olayer_linear.ComputeBackwardLoss(allmerge, output, outputLoss, allmergeLoss);


      unconcat(poolmergeLoss, gatedmergeLossFormer, gatedmergeLossLatter, allmergeLoss);
      former_interactLoss += gatedmergeLossFormer;
      latter_interactLoss += gatedmergeLossLatter;
      unconcat(beforerepresentLoss, formerrepresentLoss, middlerepresentLoss, latterrepresentLoss, afterrepresentLoss, poolmergeLoss);


      _target_attention.ComputeBackwardLoss(former_input_span[0], former_input_span[2], former_input_span[1],
          former_reset_left, former_reset_right, former_interact_middle,
          former_update_left, former_update_right, former_update_interact,
          former_interact, former_interactLoss,
          former_input_spanLoss[0], former_input_spanLoss[1], former_input_spanLoss[2]);

      _target_attention.ComputeBackwardLoss(latter_input_span[0], latter_input_span[2], latter_input_span[1],
          latter_reset_left, latter_reset_right, latter_interact_middle,
          latter_update_left, latter_update_right, latter_update_interact,
          latter_interact, latter_interactLoss,
          latter_input_spanLoss[0], latter_input_spanLoss[1], latter_input_spanLoss[2]);

      _represent_transform[0].ComputeBackwardLoss(beforerepresent, former_input_span[0], former_input_spanLoss[0], beforerepresentLoss);
      _represent_transform[1].ComputeBackwardLoss(formerrepresent, former_input_span[1], former_input_spanLoss[1], formerrepresentLoss);
      _represent_transform[2].ComputeBackwardLoss(middlerepresent, former_input_span[2], former_input_spanLoss[2], middlerepresentLoss);

      _represent_transform[2].ComputeBackwardLoss(middlerepresent, latter_input_span[0], latter_input_spanLoss[0], middlerepresentLoss);
      _represent_transform[3].ComputeBackwardLoss(latterrepresent, latter_input_span[1], latter_input_spanLoss[1], latterrepresentLoss);
      _represent_transform[4].ComputeBackwardLoss(afterrepresent, latter_input_span[2], latter_input_spanLoss[2], afterrepresentLoss);

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

      offset = _poolfunctions;
      //former
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], formerrepresentLoss);

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

      offset = 2 * _poolfunctions;
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

	  offset = 3 * _poolfunctions;
      //latter
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], latterrepresentLoss);

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

	  offset = 4 * _poolfunctions;
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

      _rnn_left.ComputeBackwardLoss(inputAndOther, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left, rnn_hidden_leftLoss, inputAndOtherLoss);
      _rnn_right.ComputeBackwardLoss(inputAndOther, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right, rnn_hidden_rightLoss, inputAndOtherLoss);
	  


      // put other emb to the input of rnn
      for (int idx = 0; idx < seq_size; idx++) {
    	  vector<Tensor<xpu, 2, dtype> > v_otherInputLoss;

    	  v_otherInputLoss.push_back(inputLoss[idx]);

    		if((_channel & 2) == 2) {

			}
			if((_channel & 4) == 4) {

			}
			if((_channel & 8) == 8) {
				v_otherInputLoss.push_back(nerprimeLoss[idx]);
			}
			if((_channel & 16) == 16) {
				v_otherInputLoss.push_back(posprimeLoss[idx]);
			}
			if((_channel & 32) == 32) {
				v_otherInputLoss.push_back(sstprimeLoss[idx]);
			}

			unconcat(v_otherInputLoss, inputAndOtherLoss[idx]);


      }


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

			if((_channel & 2) == 2) {

			}
			if((_channel & 4) == 4) {

			}
			if((_channel & 8) == 8) {
				_ner.EmbLoss(feature.ner, nerprimeLoss[idx]);
			}
			if((_channel & 16) == 16) {
				_pos.EmbLoss(feature.pos, posprimeLoss[idx]);
			}
			if((_channel & 32) == 32) {
				_sst.EmbLoss(feature.sst, sstprimeLoss[idx]);
			}
        }
      }

      //release
      FreeSpace(&wordprime);
      FreeSpace(&wordprimeLoss);
      FreeSpace(&wordprimeMask);
      FreeSpace(&wordrepresent);
      FreeSpace(&wordrepresentLoss);

      FreeSpace(&nerprime);
      FreeSpace(&nerprimeLoss);
      FreeSpace(&posprime);
      FreeSpace(&posprimeLoss);
      FreeSpace(&sstprime);
      FreeSpace(&sstprimeLoss);

      FreeSpace(&input);
      FreeSpace(&inputLoss);

      FreeSpace(&inputAndOther);
      FreeSpace(&inputAndOtherLoss);

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
        FreeSpace(&(former_input_span[idm]));
        FreeSpace(&(former_input_spanLoss[idm]));
      }

      for (int idm = 0; idm < 3; idm++) {
        FreeSpace(&(latter_input_span[idm]));
        FreeSpace(&(latter_input_spanLoss[idm]));
      }

      FreeSpace(&former_reset_left);
      FreeSpace(&former_reset_right);
      FreeSpace(&former_interact_middle);
      FreeSpace(&former_update_left);
      FreeSpace(&former_update_right);
      FreeSpace(&former_update_interact);
      FreeSpace(&(former_interact));
      FreeSpace(&(former_interactLoss));

      FreeSpace(&latter_reset_left);
      FreeSpace(&latter_reset_right);
      FreeSpace(&latter_interact_middle);
      FreeSpace(&latter_update_left);
      FreeSpace(&latter_update_right);
      FreeSpace(&latter_update_interact);
      FreeSpace(&(latter_interact));
      FreeSpace(&(latter_interactLoss));


      FreeSpace(&beforerepresent);
      FreeSpace(&beforerepresentLoss);
      FreeSpace(&formerrepresent);
      FreeSpace(&formerrepresentLoss);
      FreeSpace(&middlerepresent);
      FreeSpace(&middlerepresentLoss);
	  FreeSpace(&latterrepresent);
      FreeSpace(&latterrepresentLoss);
	  FreeSpace(&afterrepresent);
      FreeSpace(&afterrepresentLoss);

      FreeSpace(&poolmerge);
      FreeSpace(&poolmergeLoss);
      FreeSpace(&gatedmergeFormer);
      FreeSpace(&gatedmergeLossFormer);
      FreeSpace(&gatedmergeLatter);
      FreeSpace(&gatedmergeLossLatter);
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
	  const vector<Feature>& features = example.m_features;
    int seq_size = features.size();
    int offset = 0;

    Tensor<xpu, 3, dtype> input;
    Tensor<xpu, 3, dtype> inputAndOther;
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
    Tensor<xpu, 2, dtype> gatedmergeFormer;
    Tensor<xpu, 2, dtype> gatedmergeLatter;
    Tensor<xpu, 2, dtype> allmerge;
    Tensor<xpu, 2, dtype> output;

    //gated interaction part
    Tensor<xpu, 2, dtype> former_input_span[3];
    Tensor<xpu, 2, dtype> former_reset_left, former_reset_right, former_interact_middle;
    Tensor<xpu, 2, dtype> former_update_left, former_update_right, former_update_interact;
    Tensor<xpu, 2, dtype> former_interact;

    Tensor<xpu, 2, dtype> latter_input_span[3];
    Tensor<xpu, 2, dtype> latter_reset_left, latter_reset_right, latter_interact_middle;
    Tensor<xpu, 2, dtype> latter_update_left, latter_update_right, latter_update_interact;
    Tensor<xpu, 2, dtype> latter_interact;

    Tensor<xpu, 3, dtype> wordprime, wordrepresent;

    Tensor<xpu, 3, dtype> nerprime;
    Tensor<xpu, 3, dtype> posprime;
    Tensor<xpu, 3, dtype> sstprime;

    hash_set<int> beforeIndex, formerIndex, middleIndex, latterIndex, afterIndex;
    Tensor<xpu, 2, dtype> beforerepresent, beforerepresentLoss;
    Tensor<xpu, 2, dtype> formerrepresent, formerrepresentLoss;
	  Tensor<xpu, 2, dtype> middlerepresent, middlerepresentLoss;
    Tensor<xpu, 2, dtype> latterrepresent, latterrepresentLoss;
	  Tensor<xpu, 2, dtype> afterrepresent, afterrepresentLoss;

    static hash_set<int>::iterator it;

    //initialize
    wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
    wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

	  nerprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
	  posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
	  sstprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);


    input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);
    inputAndOther = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize+_otherInputSize), 0.0);

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
    formerrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    middlerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
	  latterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
	  afterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);


    for (int idm = 0; idm < 3; idm++) {
      former_input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    }

    former_reset_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    former_reset_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    former_interact_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    former_update_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    former_update_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    former_update_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    former_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);

    for (int idm = 0; idm < 3; idm++) {
      latter_input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    }

    latter_reset_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    latter_reset_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    latter_interact_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    latter_update_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    latter_update_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    latter_update_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    latter_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);


    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
    gatedmergeFormer = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
    gatedmergeLatter = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
    allmerge = NewTensor<xpu>(Shape2(1, _poolsize + 2*_gatedsize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out

      const vector<int>& words = feature.words;
      if (idx < example.formerTkBegin) {
        beforeIndex.insert(idx);
      } else if(idx >= example.formerTkBegin && idx <= example.formerTkEnd) {
			formerIndex.insert(idx);
		} else if (idx >= example.latterTkBegin && idx <= example.latterTkEnd) {
        latterIndex.insert(idx);
      } else if (idx > example.latterTkEnd) {
        afterIndex.insert(idx);
      } else {
        middleIndex.insert(idx);
      }

      _words.GetEmb(words[0], wordprime[idx]);

		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
			_ner.GetEmb(feature.ner, nerprime[idx]);
		}
		if((_channel & 16) == 16) {
			_pos.GetEmb(feature.pos, posprime[idx]);
			//cout<<feature.pos<<" ";
		}
		if((_channel & 32) == 32) {
			_sst.GetEmb(feature.sst, sstprime[idx]);
		}

    }

    for (int idx = 0; idx < seq_size; idx++) {
      wordrepresent[idx] += wordprime[idx];
    }

    windowlized(wordrepresent, input, _wordcontext);

    // put other emb to the input of rnn
    for (int idx = 0; idx < seq_size; idx++) {
  	  vector<Tensor<xpu, 2, dtype> > v_otherInput;

  	  v_otherInput.push_back(input[idx]);

  		if((_channel & 2) == 2) {

			}
			if((_channel & 4) == 4) {

			}
			if((_channel & 8) == 8) {
				v_otherInput.push_back(nerprime[idx]);
			}
			if((_channel & 16) == 16) {
				v_otherInput.push_back(posprime[idx]);
			}
			if((_channel & 32) == 32) {
				v_otherInput.push_back(sstprime[idx]);
			}

			concat(v_otherInput, inputAndOther[idx]);

    }

    _rnn_left.ComputeForwardScore(inputAndOther, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
    _rnn_right.ComputeForwardScore(inputAndOther, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

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

    offset = _poolfunctions;
    //former
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], formerIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], formerIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], formerIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], formerIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], formerIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], formerrepresent);

    offset = 2 * _poolfunctions;
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

	  offset = 3 * _poolfunctions;
    //latter
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], latterIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], latterIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], latterIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], latterIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], latterIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], latterrepresent);

	  offset = 4 * _poolfunctions;
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


    _represent_transform[0].ComputeForwardScore(beforerepresent, former_input_span[0]);
    _represent_transform[1].ComputeForwardScore(formerrepresent, former_input_span[1]);
    _represent_transform[2].ComputeForwardScore(middlerepresent, former_input_span[2]);

    _represent_transform[2].ComputeForwardScore(middlerepresent, latter_input_span[0]);
    _represent_transform[3].ComputeForwardScore(latterrepresent, latter_input_span[1]);
    _represent_transform[4].ComputeForwardScore(afterrepresent, latter_input_span[2]);

    _target_attention.ComputeForwardScore(former_input_span[0], former_input_span[2], former_input_span[1],
        former_reset_left, former_reset_right, former_interact_middle,
        former_update_left, former_update_right, former_update_interact,
        former_interact);

    _target_attention.ComputeForwardScore(latter_input_span[0], latter_input_span[2], latter_input_span[1],
        latter_reset_left, latter_reset_right, latter_interact_middle,
        latter_update_left, latter_update_right, latter_update_interact,
        latter_interact);


    concat(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent, poolmerge);
    gatedmergeFormer += former_interact;
    gatedmergeLatter += latter_interact;
    concat(poolmerge, gatedmergeFormer, gatedmergeLatter, allmerge);


    _olayer_linear.ComputeForwardScore(allmerge, output);

    // decode algorithm
    //int optLabel = softmax_predict(output, results);
    int optLabel = -1;
    for(int i=0;i<output.size(1);i++)
    	results.push_back(output[0][i]);

    //release
    FreeSpace(&wordprime);
    FreeSpace(&wordrepresent);

    FreeSpace(&nerprime);
    FreeSpace(&posprime);
    FreeSpace(&sstprime);

    FreeSpace(&input);
    FreeSpace(&inputAndOther);

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
      FreeSpace(&(former_input_span[idm]));
    }
    
    for (int idm = 0; idm < 3; idm++) {
      FreeSpace(&(latter_input_span[idm]));
    }

    FreeSpace(&former_reset_left);
    FreeSpace(&former_reset_right);
    FreeSpace(&former_interact_middle);
    FreeSpace(&former_update_left);
    FreeSpace(&former_update_right);
    FreeSpace(&former_update_interact);
    FreeSpace(&(former_interact));

    FreeSpace(&latter_reset_left);
    FreeSpace(&latter_reset_right);
    FreeSpace(&latter_interact_middle);
    FreeSpace(&latter_update_left);
    FreeSpace(&latter_update_right);
    FreeSpace(&latter_update_interact);
    FreeSpace(&(latter_interact));

    FreeSpace(&beforerepresent);
    FreeSpace(&beforerepresentLoss);
    FreeSpace(&formerrepresent);
    FreeSpace(&formerrepresentLoss);
    FreeSpace(&middlerepresent);
    FreeSpace(&middlerepresentLoss);
	  FreeSpace(&latterrepresent);
    FreeSpace(&latterrepresentLoss);
	  FreeSpace(&afterrepresent);
    FreeSpace(&afterrepresentLoss);

    FreeSpace(&poolmerge);
    FreeSpace(&gatedmergeFormer);
    FreeSpace(&gatedmergeLatter);
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

    for (int idx = 0; idx < 5; idx++) {
      _represent_transform[idx].updateAdaGrad(nnRegular, adaAlpha, adaEps);
    }

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
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
