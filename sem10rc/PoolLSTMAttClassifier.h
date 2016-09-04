
#ifndef SRC_PoolLSTMAttClassifier_H_
#define SRC_PoolLSTMAttClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "N3L.h"
#include "Attention2.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class PoolLSTMAttClassifier {
public:
	PoolLSTMAttClassifier() {
    _dropOut = 0.5;
  }
  ~PoolLSTMAttClassifier() {

  }

public:
  LookupTable<xpu> _words;

  LookupTable<xpu> _randomWord;
  LookupTable<xpu> _character;
  LookupTable<xpu> _pos;
  LookupTable<xpu> _sst;
  LookupTable<xpu> _ner;

  int _wordcontext, _wordwindow;
  int _wordDim;
  int _wordWindowSize;

  int _randomWordContext;
  int _randomWordDim;
  int _randomWordWindowSize;

  int _charactercontext;
  int _characterWindowSize;

  int _inputsize;
  int _hiddensize;
  int _rnnHiddenSize;

  UniLayer<xpu> _olayer_linear;
  UniLayer<xpu> _tanh_project;

  LSTM<xpu> _rnn_left;
  LSTM<xpu> _rnn_right;

  LSTM<xpu> _rnn_character_left;
  LSTM<xpu> _rnn_character_right;

  Attention2<xpu> _attention;

  int _poolmanners;
  int _poolfunctions;

  int _poolsize;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3, min, 4, std, 5, pro

  Options options;

  int _otherDim;

  int _attSize;

public:

  inline void init(const NRMat<dtype>& wordEmb, Options options) {
	  this->options = options;
    _wordcontext = options.wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordDim = options.wordEmbSize;
    _randomWordDim = options.entity_embsize;
    _otherDim = options.otherEmbSize;
    _charactercontext = options.charactercontext;
    _randomWordContext = options.wordcontext;

    _labelSize = MAX_RELATION;

    _poolfunctions = 4;
    _poolmanners = _poolfunctions * 5; //( left, right, target) * (avg, max, min, std, pro)

    _wordWindowSize = _wordwindow * _wordDim;
    _characterWindowSize = (2*_charactercontext + 1) * _otherDim;
    _randomWordWindowSize = (2*_randomWordContext + 1) * _randomWordDim;

     _inputsize = _wordWindowSize;
	if((options.channelMode & 2) == 2) {
		_inputsize += _randomWordWindowSize;
	}
	if((options.channelMode & 4) == 4) {
		_inputsize += _otherDim*2;
	}
	if((options.channelMode & 16) == 16) {
		_inputsize += _otherDim;
	}
	if((options.channelMode & 32) == 32) {
		_inputsize += _otherDim;
	}


    _hiddensize = options.wordEmbSize;
    _rnnHiddenSize = options.rnnHiddenSize;

    _attSize = _poolfunctions * _hiddensize;
    _poolsize = _poolmanners * _hiddensize + 3*_attSize;

    _words.initial(wordEmb);

    _rnn_left.initial(_rnnHiddenSize, _inputsize, true, 10);
    _rnn_right.initial(_rnnHiddenSize, _inputsize, false, 40);

    _rnn_character_left.initial(_otherDim, _characterWindowSize, true, 20);
    _rnn_character_right.initial(_otherDim, _characterWindowSize, false, 30);

    _tanh_project.initial(_hiddensize, 2 * _rnnHiddenSize, true, 70, 0);

    _attention.initial(_attSize, _attSize, _attSize, false, 90);

    _olayer_linear.initial(_labelSize, _poolsize, false, 80, 2);

    _remove = 0;

    cout<<"PoolLSTMAttClassifier initial"<<endl;
  }

  inline void release() {
    _words.release();

    _randomWord.release();
    _character.release();
    _sst.release();
    _ner.release();
    _pos.release();

    _olayer_linear.release();
    _tanh_project.release();
    _rnn_left.release();
    _rnn_right.release();

    _rnn_character_left.release();
    _rnn_character_right.release();

    _attention.release();
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
      Tensor<xpu, 3, dtype> project, projectLoss;

      vector< Tensor<xpu, 3, dtype> > rnn_character_left(seq_size), rnn_character_leftLoss(seq_size);
      vector< Tensor<xpu, 3, dtype> > rnn_character_left_iy(seq_size), rnn_character_left_oy(seq_size),
    		  rnn_character_left_fy(seq_size), rnn_character_left_mcy(seq_size),
			  rnn_character_left_cy(seq_size), rnn_character_left_my(seq_size);

      vector< Tensor<xpu, 3, dtype> > rnn_character_right(seq_size), rnn_character_rightLoss(seq_size);
      vector< Tensor<xpu, 3, dtype> > rnn_character_right_iy(seq_size), rnn_character_right_oy(seq_size),
    		  rnn_character_right_fy(seq_size), rnn_character_right_mcy(seq_size),
			  rnn_character_right_cy(seq_size), rnn_character_right_my(seq_size);

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
      Tensor<xpu, 2, dtype> output, outputLoss;

      Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, dtype> wordWindow, wordWindowLoss;

      Tensor<xpu, 3, dtype> randomWordprime, randomWordprimeLoss, randomWordprimeMask;
      Tensor<xpu, 3, dtype> randomWordWindow, randomWordWindowLoss;
      vector< Tensor<xpu, 3, dtype> > characterprime(seq_size), characterprimeLoss(seq_size), characterprimeMask(seq_size);
      vector< Tensor<xpu, 3, dtype> > characterWindow(seq_size), characterWindowLoss(seq_size);
      Tensor<xpu, 3, dtype> posprime, posprimeLoss, posprimeMask;
      Tensor<xpu, 3, dtype> sstprime, sstprimeLoss, sstprimeMask;

      hash_set<int> beforeIndex, formerIndex, middleIndex, latterIndex, afterIndex;
      Tensor<xpu, 2, dtype> beforerepresent, beforerepresentLoss;
      Tensor<xpu, 2, dtype> formerrepresent, formerrepresentLoss;
	  Tensor<xpu, 2, dtype> middlerepresent, middlerepresentLoss;
      Tensor<xpu, 2, dtype> latterrepresent, latterrepresentLoss;
	  Tensor<xpu, 2, dtype> afterrepresent, afterrepresentLoss;

		vector<Tensor<xpu, 2, dtype> > xMExp(3);
		vector<Tensor<xpu, 2, dtype> > xExp(3);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex(3);
		Tensor<xpu, 2, dtype> xSum;
		vector<Tensor<xpu, 2, dtype> > att_y(3);
		vector<Tensor<xpu, 2, dtype> > att_ly(3);

      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 1.0);

      wordWindow = NewTensor<xpu>(Shape3(seq_size, 1, _wordWindowSize), 0.0);
      wordWindowLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordWindowSize), 0.0);

		if((options.channelMode & 2) == 2) {
		      randomWordprime = NewTensor<xpu>(Shape3(seq_size, 1, _randomWordDim), 0.0);
		      randomWordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _randomWordDim), 0.0);
		      randomWordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _randomWordDim), 1.0);
		      randomWordWindow = NewTensor<xpu>(Shape3(seq_size, 1, _randomWordWindowSize), 0.0);
		      randomWordWindowLoss = NewTensor<xpu>(Shape3(seq_size, 1, _randomWordWindowSize), 0.0);
		}
			// character initial later
		if((options.channelMode & 16) == 16) {
		      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
		      posprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
		      posprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);
		}
		if((options.channelMode & 32) == 32) {
		      sstprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
		      sstprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
		      sstprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);
		}

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

      poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);


		for (int idx = 0; idx < 3; idx++) {
			xMExp[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
			xExp[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
			xPoolIndex[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
			att_y[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
			att_ly[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
		}
		xSum = NewTensor<xpu>(Shape2(1, _attSize), d_zero);


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
        //dropout
        //if (_words.bEmbFineTune())
        	dropoutcol(wordprimeMask[idx], _dropOut);
        wordprime[idx] = wordprime[idx] * wordprimeMask[idx];

    	if((options.channelMode & 2) == 2) {
		   _randomWord.GetEmb(feature.randomWord, randomWordprime[idx]);
			//dropout
			dropoutcol(randomWordprimeMask[idx], _dropOut);
			randomWordprime[idx] = randomWordprime[idx] * randomWordprimeMask[idx];
    	}
    	if((options.channelMode & 4) == 4) {
    		characterprime[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    		characterprimeLoss[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    		characterprimeMask[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 1.0));

  	      for(int i=0;i<feature.characters.size();i++) {
  	 		   _character.GetEmb(feature.characters[i], characterprime[idx][i]);
  	 			//dropout
  	 			dropoutcol(characterprimeMask[idx][i], 0.2);
  	 			characterprime[idx][i] = characterprime[idx][i] * characterprimeMask[idx][i];
  	      }

  		characterWindow[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _characterWindowSize), 0.0));
  		characterWindowLoss[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _characterWindowSize), 0.0));

    	      rnn_character_left_iy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_left_oy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_left_fy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_left_mcy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_left_cy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_left_my[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_left[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_leftLoss[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));

    	      rnn_character_right_iy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_right_oy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_right_fy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_right_mcy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_right_cy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_right_my[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_right[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
    	      rnn_character_rightLoss[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));

    	}
    	if((options.channelMode & 16) == 16) {
 		   _pos.GetEmb(feature.pos, posprime[idx]);
 			//dropout
 			dropoutcol(posprimeMask[idx], _dropOut);
 			posprime[idx] = posprime[idx] * posprimeMask[idx];
    	}
    	if((options.channelMode & 32) == 32) {
 		   _sst.GetEmb(feature.sst, sstprime[idx]);
 			//dropout
 			dropoutcol(sstprimeMask[idx], _dropOut);
 			sstprime[idx] = sstprime[idx] * sstprimeMask[idx];
    	}

      }


      windowlized(wordprime, wordWindow, _wordcontext);

		if((options.channelMode & 2) == 2) {
			windowlized(randomWordprime, randomWordWindow, _randomWordContext);
		}

      for(int idx=0;idx<seq_size;idx++) {
    	  vector<Tensor<xpu, 2, dtype> > v_Input;
    	  v_Input.push_back(wordWindow[idx]);

			if((options.channelMode & 2) == 2) {
				v_Input.push_back(randomWordWindow[idx]);
			}
			if((options.channelMode & 4) == 4) {

				windowlized(characterprime[idx], characterWindow[idx], _charactercontext);

			      _rnn_character_left.ComputeForwardScore(characterWindow[idx], rnn_character_left_iy[idx], rnn_character_left_oy[idx], rnn_character_left_fy[idx],
			    		  rnn_character_left_mcy[idx], rnn_character_left_cy[idx], rnn_character_left_my[idx], rnn_character_left[idx]);
			      _rnn_character_right.ComputeForwardScore(characterWindow[idx], rnn_character_right_iy[idx], rnn_character_right_oy[idx], rnn_character_right_fy[idx],
			          		  rnn_character_right_mcy[idx], rnn_character_right_cy[idx], rnn_character_right_my[idx], rnn_character_right[idx]);

			      v_Input.push_back(rnn_character_left[idx][rnn_character_left[idx].size(0)-1]);
			      v_Input.push_back(rnn_character_right[idx][0]);
			}
			if((options.channelMode & 16) == 16) {
				v_Input.push_back(posprime[idx]);
			}
			if((options.channelMode & 32) == 32) {
				v_Input.push_back(sstprime[idx]);
			}

			concat(v_Input, input[idx]);
      }


      _rnn_left.ComputeForwardScore(input, rnn_hidden_left_iy, rnn_hidden_left_oy, rnn_hidden_left_fy,
    		  rnn_hidden_left_mcy, rnn_hidden_left_cy, rnn_hidden_left_my, rnn_hidden_left);
      _rnn_right.ComputeForwardScore(input, rnn_hidden_right_iy, rnn_hidden_right_oy, rnn_hidden_right_fy,
          		  rnn_hidden_right_mcy, rnn_hidden_right_cy, rnn_hidden_right_my, rnn_hidden_right);

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

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], beforerepresent);

      offset += _poolfunctions;
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

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], formerrepresent);

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

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], middlerepresent);

      offset += _poolfunctions;
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

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], latterrepresent);

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

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], afterrepresent);

      // attention
      vector<Tensor<xpu, 2, dtype> > att_x;
      att_x.push_back(beforerepresent);
      att_x.push_back(middlerepresent);
      att_x.push_back(afterrepresent);
		_attention.ComputeForwardScore(att_x, formerrepresent, latterrepresent,
				xMExp, xExp, xSum,
				xPoolIndex, att_y);

      concat8(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent,
    		  att_y[0], att_y[1], att_y[2],
    		  poolmerge);

      _olayer_linear.ComputeForwardScore(poolmerge, output);

      // get delta for each output
      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      _olayer_linear.ComputeBackwardLoss(poolmerge, output, outputLoss, poolmergeLoss);

      unconcat8(beforerepresentLoss, formerrepresentLoss, middlerepresentLoss, latterrepresentLoss, afterrepresentLoss,
    		  att_ly[0], att_ly[1], att_ly[2],
    		  poolmergeLoss);

      // attention
      vector<Tensor<xpu, 2, dtype> > att_lx;
      att_lx.push_back(beforerepresentLoss);
      att_lx.push_back(middlerepresentLoss);
      att_lx.push_back(afterrepresentLoss);
		_attention.ComputeBackwardLoss(att_x, formerrepresent, latterrepresent,
				xMExp, xExp, xSum,
				xPoolIndex, att_y, att_ly, att_lx, formerrepresentLoss, latterrepresentLoss);


      offset = 0;
      //before
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], beforerepresentLoss);

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

      offset += _poolfunctions;
      //former
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], formerrepresentLoss);

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

      offset += _poolfunctions;
      //middle
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], middlerepresentLoss);

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

      offset += _poolfunctions;
      //latter
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], latterrepresentLoss);

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

      offset += _poolfunctions;
      //after
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], afterrepresentLoss);

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

      for (int idx = 0; idx < seq_size; idx++) {
        _tanh_project.ComputeBackwardLoss(rnn_hidden_merge[idx], project[idx], projectLoss[idx], rnn_hidden_mergeLoss[idx]);
      }

      for (int idx = 0; idx < seq_size; idx++) {
        unconcat(rnn_hidden_leftLoss[idx], rnn_hidden_rightLoss[idx], rnn_hidden_mergeLoss[idx]);
      }

      _rnn_left.ComputeBackwardLoss(input, rnn_hidden_left_iy, rnn_hidden_left_oy, rnn_hidden_left_fy,
    		  rnn_hidden_left_mcy, rnn_hidden_left_cy, rnn_hidden_left_my, rnn_hidden_left,
			  rnn_hidden_leftLoss, inputLoss);
      _rnn_right.ComputeBackwardLoss(input, rnn_hidden_right_iy, rnn_hidden_right_oy, rnn_hidden_right_fy,
          		  rnn_hidden_right_mcy, rnn_hidden_right_cy, rnn_hidden_right_my, rnn_hidden_right,
				  rnn_hidden_rightLoss, inputLoss);


      for(int idx=0;idx<seq_size;idx++) {
    	  vector<Tensor<xpu, 2, dtype> > v_Input_loss;
    	  v_Input_loss.push_back(wordWindowLoss[idx]);

			if((options.channelMode & 2) == 2) {
				v_Input_loss.push_back(randomWordWindowLoss[idx]);
			}
			if((options.channelMode & 4) == 4) {
				v_Input_loss.push_back(rnn_character_leftLoss[idx][rnn_character_leftLoss[idx].size(0)-1]);
				v_Input_loss.push_back(rnn_character_rightLoss[idx][0]);
			}
			if((options.channelMode & 16) == 16) {
				v_Input_loss.push_back(posprimeLoss[idx]);
			}
			if((options.channelMode & 32) == 32) {
				v_Input_loss.push_back(sstprimeLoss[idx]);
			}

			unconcat(v_Input_loss, inputLoss[idx]);

			if((options.channelMode & 4) == 4) {
			      _rnn_character_left.ComputeBackwardLoss(characterWindow[idx], rnn_character_left_iy[idx], rnn_character_left_oy[idx], rnn_character_left_fy[idx],
			    		  rnn_character_left_mcy[idx], rnn_character_left_cy[idx], rnn_character_left_my[idx], rnn_character_left[idx],
						  rnn_character_leftLoss[idx], characterWindowLoss[idx]);
			      _rnn_character_right.ComputeBackwardLoss(characterWindow[idx], rnn_character_right_iy[idx], rnn_character_right_oy[idx], rnn_character_right_fy[idx],
			          		  rnn_character_right_mcy[idx], rnn_character_right_cy[idx], rnn_character_right_my[idx], rnn_character_right[idx],
							  rnn_character_rightLoss[idx], characterWindowLoss[idx]);

				windowlized_backward(characterprimeLoss[idx], characterWindowLoss[idx], _charactercontext);

			}
      }


		if((options.channelMode & 2) == 2) {
			windowlized_backward(randomWordprimeLoss, randomWordWindowLoss, _randomWordContext);
		}

      // word context
      windowlized_backward(wordprimeLoss, wordWindowLoss, _wordcontext);


      if (_words.bEmbFineTune()) {
        for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];
          const vector<int>& words = feature.words;
          wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
          _words.EmbLoss(words[0], wordprimeLoss[idx]);
        }
      }

      for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];

			if((options.channelMode & 2) == 2) {
				randomWordprimeLoss[idx] = randomWordprimeLoss[idx] * randomWordprimeMask[idx];
				_randomWord.EmbLoss(feature.randomWord, randomWordprimeLoss[idx]);
			}
	    	if((options.channelMode & 4) == 4) {
	    		for(int i=0;i<feature.characters.size();i++) {
	    			characterprimeLoss[idx][i] = characterprimeLoss[idx][i] * characterprimeMask[idx][i];
	    			_character.EmbLoss(feature.characters[i], characterprimeLoss[idx][i]);
	    		}
	    	}
	    	if((options.channelMode & 16) == 16) {
				posprimeLoss[idx] = posprimeLoss[idx] * posprimeMask[idx];
				_pos.EmbLoss(feature.pos, posprimeLoss[idx]);
	    	}
			if((options.channelMode & 32) == 32) {
				sstprimeLoss[idx] = sstprimeLoss[idx] * sstprimeMask[idx];
				_sst.EmbLoss(feature.sst, sstprimeLoss[idx]);
			}
      }

      //release
      FreeSpace(&wordprime);
      FreeSpace(&wordprimeLoss);
      FreeSpace(&wordprimeMask);
      FreeSpace(&wordWindow);
      FreeSpace(&wordWindowLoss);

		if((options.channelMode & 2) == 2) {
		      FreeSpace(&randomWordprime);
		      FreeSpace(&randomWordprimeLoss);
		      FreeSpace(&randomWordprimeMask);
		      FreeSpace(&randomWordWindow);
		      FreeSpace(&randomWordWindowLoss);
		}
		if((options.channelMode & 4) == 4) {
		      for (int idx = 0; idx < seq_size; idx++) {

				      FreeSpace(&(characterprime[idx]));
				      FreeSpace(&(characterprimeLoss[idx]));
				      FreeSpace(&(characterprimeMask[idx]));
				      FreeSpace(&(characterWindow[idx]));
				      FreeSpace(&(characterWindowLoss[idx]));

				      FreeSpace(&(rnn_character_left_iy[idx]));
				      FreeSpace(&(rnn_character_left_oy[idx]));
				      FreeSpace(&(rnn_character_left_fy[idx]));
				      FreeSpace(&(rnn_character_left_mcy[idx]));
				      FreeSpace(&(rnn_character_left_cy[idx]));
				      FreeSpace(&(rnn_character_left_my[idx]));
				      FreeSpace(&(rnn_character_left[idx]));
				      FreeSpace(&(rnn_character_leftLoss[idx]));

				      FreeSpace(&(rnn_character_right_iy[idx]));
				      FreeSpace(&(rnn_character_right_oy[idx]));
				      FreeSpace(&(rnn_character_right_fy[idx]));
				      FreeSpace(&(rnn_character_right_mcy[idx]));
				      FreeSpace(&(rnn_character_right_cy[idx]));
				      FreeSpace(&(rnn_character_right_my[idx]));
				      FreeSpace(&(rnn_character_right[idx]));
				      FreeSpace(&(rnn_character_rightLoss[idx]));
		      }
		}
		if((options.channelMode & 16) == 16) {
		      FreeSpace(&posprime);
		      FreeSpace(&posprimeLoss);
		      FreeSpace(&posprimeMask);
		}
		if((options.channelMode & 32) == 32) {
		      FreeSpace(&sstprime);
		      FreeSpace(&sstprimeLoss);
		      FreeSpace(&sstprimeMask);
		}

      FreeSpace(&input);
      FreeSpace(&inputLoss);

      FreeSpace(&rnn_hidden_left_iy);
      FreeSpace(&rnn_hidden_left_oy);
      FreeSpace(&rnn_hidden_left_fy);
      FreeSpace(&rnn_hidden_left_mcy);
      FreeSpace(&rnn_hidden_left_cy);
      FreeSpace(&rnn_hidden_left_my);
      FreeSpace(&rnn_hidden_left);
      FreeSpace(&rnn_hidden_leftLoss);

      FreeSpace(&rnn_hidden_right_iy);
      FreeSpace(&rnn_hidden_right_oy);
      FreeSpace(&rnn_hidden_right_fy);
      FreeSpace(&rnn_hidden_right_mcy);
      FreeSpace(&rnn_hidden_right_cy);
      FreeSpace(&rnn_hidden_right_my);
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
      FreeSpace(&output);
      FreeSpace(&outputLoss);

		for (int idx = 0; idx < 3; idx++) {
			FreeSpace(&(xMExp[idx]));
			FreeSpace(&(xExp[idx]));
			FreeSpace(&(xPoolIndex[idx]));
			FreeSpace(&(att_y[idx]));
			FreeSpace(&(att_ly[idx]));
		}
		FreeSpace(&xSum);

    }

    return cost;
  }

  int predict(const Example& example, vector<dtype>& results) {

		const vector<Feature>& features = example.m_features;
		int seq_size = features.size();
		int offset = 0;

		Tensor<xpu, 3, dtype> input;
		Tensor<xpu, 3, dtype> project;

		vector< Tensor<xpu, 3, dtype> > rnn_character_left(seq_size);
		vector< Tensor<xpu, 3, dtype> > rnn_character_left_iy(seq_size), rnn_character_left_oy(seq_size),
				  rnn_character_left_fy(seq_size), rnn_character_left_mcy(seq_size),
				  rnn_character_left_cy(seq_size), rnn_character_left_my(seq_size);

		vector< Tensor<xpu, 3, dtype> > rnn_character_right(seq_size);
		vector< Tensor<xpu, 3, dtype> > rnn_character_right_iy(seq_size), rnn_character_right_oy(seq_size),
				  rnn_character_right_fy(seq_size), rnn_character_right_mcy(seq_size),
				  rnn_character_right_cy(seq_size), rnn_character_right_my(seq_size);

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

		Tensor<xpu, 3, dtype> wordprime;
		Tensor<xpu, 3, dtype> wordWindow;
		Tensor<xpu, 3, dtype> randomWordprime;
		Tensor<xpu, 3, dtype> randomWordWindow;
		vector< Tensor<xpu, 3, dtype> > characterprime(seq_size);
		vector< Tensor<xpu, 3, dtype> > characterWindow(seq_size);
		Tensor<xpu, 3, dtype> posprime;
		Tensor<xpu, 3, dtype> sstprime;

			hash_set<int> beforeIndex, formerIndex, middleIndex, latterIndex, afterIndex;
		  Tensor<xpu, 2, dtype> beforerepresent;
		  Tensor<xpu, 2, dtype> formerrepresent;
			  Tensor<xpu, 2, dtype> middlerepresent;
		  Tensor<xpu, 2, dtype> latterrepresent;
			  Tensor<xpu, 2, dtype> afterrepresent;

				vector<Tensor<xpu, 2, dtype> > xMExp(3);
				vector<Tensor<xpu, 2, dtype> > xExp(3);
				vector<Tensor<xpu, 2, dtype> > xPoolIndex(3);
				Tensor<xpu, 2, dtype> xSum;
				vector<Tensor<xpu, 2, dtype> > att_y(3);


		//initialize
		wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
		wordWindow = NewTensor<xpu>(Shape3(seq_size, 1, _wordWindowSize), 0.0);

			if((options.channelMode & 2) == 2) {
				  randomWordprime = NewTensor<xpu>(Shape3(seq_size, 1, _randomWordDim), 0.0);
				  randomWordWindow = NewTensor<xpu>(Shape3(seq_size, 1, _randomWordWindowSize), 0.0);
			}
				// character initial later
			if((options.channelMode & 16) == 16) {
				  posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
			}
			if((options.channelMode & 32) == 32) {
				  sstprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
			}

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

		beforerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
		formerrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
		middlerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
			  latterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
			  afterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

		poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
		output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

		for (int idx = 0; idx < 3; idx++) {
			xMExp[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
			xExp[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
			xPoolIndex[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
			att_y[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
		}
		xSum = NewTensor<xpu>(Shape2(1, _attSize), d_zero);

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

			if((options.channelMode & 2) == 2) {
			   _randomWord.GetEmb(feature.randomWord, randomWordprime[idx]);
			}
			if((options.channelMode & 4) == 4) {
				characterprime[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));

				  for(int i=0;i<feature.characters.size();i++) {
					   _character.GetEmb(feature.characters[i], characterprime[idx][i]);
				  }

				characterWindow[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _characterWindowSize), 0.0));

				  rnn_character_left_iy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left_oy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left_fy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left_mcy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left_cy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left_my[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));

				  rnn_character_right_iy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right_oy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right_fy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right_mcy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right_cy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right_my[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));

			}
			if((options.channelMode & 16) == 16) {
				   _pos.GetEmb(feature.pos, posprime[idx]);
			}
			if((options.channelMode & 32) == 32) {
				   _sst.GetEmb(feature.sst, sstprime[idx]);
			}
		}

		windowlized(wordprime, wordWindow, _wordcontext);

			if((options.channelMode & 2) == 2) {
				windowlized(randomWordprime, randomWordWindow, _randomWordContext);
			}

		for(int idx=0;idx<seq_size;idx++) {
			  vector<Tensor<xpu, 2, dtype> > v_Input;
			  v_Input.push_back(wordWindow[idx]);

				if((options.channelMode & 2) == 2) {
					v_Input.push_back(randomWordWindow[idx]);
				}
				if((options.channelMode & 4) == 4) {

					windowlized(characterprime[idx], characterWindow[idx], _charactercontext);

					  _rnn_character_left.ComputeForwardScore(characterWindow[idx], rnn_character_left_iy[idx], rnn_character_left_oy[idx], rnn_character_left_fy[idx],
							  rnn_character_left_mcy[idx], rnn_character_left_cy[idx], rnn_character_left_my[idx], rnn_character_left[idx]);
					  _rnn_character_right.ComputeForwardScore(characterWindow[idx], rnn_character_right_iy[idx], rnn_character_right_oy[idx], rnn_character_right_fy[idx],
								  rnn_character_right_mcy[idx], rnn_character_right_cy[idx], rnn_character_right_my[idx], rnn_character_right[idx]);

					  v_Input.push_back(rnn_character_left[idx][rnn_character_left[idx].size(0)-1]);
					  v_Input.push_back(rnn_character_right[idx][0]);
				}
				if((options.channelMode & 16) == 16) {
					v_Input.push_back(posprime[idx]);
				}
				if((options.channelMode & 32) == 32) {
					v_Input.push_back(sstprime[idx]);
				}

				concat(v_Input, input[idx]);
		}

		_rnn_left.ComputeForwardScore(input, rnn_hidden_left_iy, rnn_hidden_left_oy, rnn_hidden_left_fy,
				  rnn_hidden_left_mcy, rnn_hidden_left_cy, rnn_hidden_left_my, rnn_hidden_left);
		_rnn_right.ComputeForwardScore(input, rnn_hidden_right_iy, rnn_hidden_right_oy, rnn_hidden_right_fy,
					  rnn_hidden_right_mcy, rnn_hidden_right_cy, rnn_hidden_right_my, rnn_hidden_right);


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

		concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], beforerepresent);

		offset += _poolfunctions;
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

		concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], formerrepresent);

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

		concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], middlerepresent);

		offset += _poolfunctions;
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

		concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], latterrepresent);

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

		concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], afterrepresent);

	      // attention
	      vector<Tensor<xpu, 2, dtype> > att_x;
	      att_x.push_back(beforerepresent);
	      att_x.push_back(middlerepresent);
	      att_x.push_back(afterrepresent);
			_attention.ComputeForwardScore(att_x, formerrepresent, latterrepresent,
					xMExp, xExp, xSum,
					xPoolIndex, att_y);

	      concat8(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent,
	    		  att_y[0], att_y[1], att_y[2],
	    		  poolmerge);

		_olayer_linear.ComputeForwardScore(poolmerge, output);

		// decode algorithm
		int optLabel = softmax_predict(output, results);

		//release
		FreeSpace(&wordprime);
		FreeSpace(&wordWindow);

			if((options.channelMode & 2) == 2) {
				  FreeSpace(&randomWordprime);
				  FreeSpace(&randomWordWindow);
			}
			if((options.channelMode & 4) == 4) {
				  for (int idx = 0; idx < seq_size; idx++) {

						  FreeSpace(&(characterprime[idx]));
						  FreeSpace(&(characterWindow[idx]));

						  FreeSpace(&(rnn_character_left_iy[idx]));
						  FreeSpace(&(rnn_character_left_oy[idx]));
						  FreeSpace(&(rnn_character_left_fy[idx]));
						  FreeSpace(&(rnn_character_left_mcy[idx]));
						  FreeSpace(&(rnn_character_left_cy[idx]));
						  FreeSpace(&(rnn_character_left_my[idx]));
						  FreeSpace(&(rnn_character_left[idx]));

						  FreeSpace(&(rnn_character_right_iy[idx]));
						  FreeSpace(&(rnn_character_right_oy[idx]));
						  FreeSpace(&(rnn_character_right_fy[idx]));
						  FreeSpace(&(rnn_character_right_mcy[idx]));
						  FreeSpace(&(rnn_character_right_cy[idx]));
						  FreeSpace(&(rnn_character_right_my[idx]));
						  FreeSpace(&(rnn_character_right[idx]));
				  }
			}
			if((options.channelMode & 16) == 16) {
				  FreeSpace(&posprime);
			}
			if((options.channelMode & 32) == 32) {
				  FreeSpace(&sstprime);
			}

		FreeSpace(&input);

		FreeSpace(&rnn_hidden_left_iy);
		FreeSpace(&rnn_hidden_left_oy);
		FreeSpace(&rnn_hidden_left_fy);
		FreeSpace(&rnn_hidden_left_mcy);
		FreeSpace(&rnn_hidden_left_cy);
		FreeSpace(&rnn_hidden_left_my);
		FreeSpace(&rnn_hidden_left);

		FreeSpace(&rnn_hidden_right_iy);
		FreeSpace(&rnn_hidden_right_oy);
		FreeSpace(&rnn_hidden_right_fy);
		FreeSpace(&rnn_hidden_right_mcy);
		FreeSpace(&rnn_hidden_right_cy);
		FreeSpace(&rnn_hidden_right_my);
		FreeSpace(&rnn_hidden_right);

		FreeSpace(&rnn_hidden_merge);

		FreeSpace(&project);

		for (int idm = 0; idm < _poolmanners; idm++) {
		  FreeSpace(&(pool[idm]));
		  FreeSpace(&(poolIndex[idm]));
		}

		FreeSpace(&beforerepresent);
		FreeSpace(&formerrepresent);
		FreeSpace(&middlerepresent);
			  FreeSpace(&latterrepresent);
			  FreeSpace(&afterrepresent);

		FreeSpace(&poolmerge);
		FreeSpace(&output);

		for (int idx = 0; idx < 3; idx++) {
			FreeSpace(&(xMExp[idx]));
			FreeSpace(&(xExp[idx]));
			FreeSpace(&(xPoolIndex[idx]));
			FreeSpace(&(att_y[idx]));
		}
		FreeSpace(&xSum);


		return optLabel;

  }

  dtype computeScore(const Example& example) {

		const vector<Feature>& features = example.m_features;
		int seq_size = features.size();
		int offset = 0;

		Tensor<xpu, 3, dtype> input;
		Tensor<xpu, 3, dtype> project;

		vector< Tensor<xpu, 3, dtype> > rnn_character_left(seq_size);
		vector< Tensor<xpu, 3, dtype> > rnn_character_left_iy(seq_size), rnn_character_left_oy(seq_size),
				  rnn_character_left_fy(seq_size), rnn_character_left_mcy(seq_size),
				  rnn_character_left_cy(seq_size), rnn_character_left_my(seq_size);

		vector< Tensor<xpu, 3, dtype> > rnn_character_right(seq_size);
		vector< Tensor<xpu, 3, dtype> > rnn_character_right_iy(seq_size), rnn_character_right_oy(seq_size),
				  rnn_character_right_fy(seq_size), rnn_character_right_mcy(seq_size),
				  rnn_character_right_cy(seq_size), rnn_character_right_my(seq_size);

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

		Tensor<xpu, 3, dtype> wordprime;
		Tensor<xpu, 3, dtype> wordWindow;
		Tensor<xpu, 3, dtype> randomWordprime;
		Tensor<xpu, 3, dtype> randomWordWindow;
		vector< Tensor<xpu, 3, dtype> > characterprime(seq_size);
		vector< Tensor<xpu, 3, dtype> > characterWindow(seq_size);
		Tensor<xpu, 3, dtype> posprime;
		Tensor<xpu, 3, dtype> sstprime;

			hash_set<int> beforeIndex, formerIndex, middleIndex, latterIndex, afterIndex;
		  Tensor<xpu, 2, dtype> beforerepresent;
		  Tensor<xpu, 2, dtype> formerrepresent;
			  Tensor<xpu, 2, dtype> middlerepresent;
		  Tensor<xpu, 2, dtype> latterrepresent;
			  Tensor<xpu, 2, dtype> afterrepresent;

				vector<Tensor<xpu, 2, dtype> > xMExp(3);
				vector<Tensor<xpu, 2, dtype> > xExp(3);
				vector<Tensor<xpu, 2, dtype> > xPoolIndex(3);
				Tensor<xpu, 2, dtype> xSum;
				vector<Tensor<xpu, 2, dtype> > att_y(3);


		//initialize
		wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
		wordWindow = NewTensor<xpu>(Shape3(seq_size, 1, _wordWindowSize), 0.0);

			if((options.channelMode & 2) == 2) {
				  randomWordprime = NewTensor<xpu>(Shape3(seq_size, 1, _randomWordDim), 0.0);
				  randomWordWindow = NewTensor<xpu>(Shape3(seq_size, 1, _randomWordWindowSize), 0.0);
			}
				// character initial later
			if((options.channelMode & 16) == 16) {
				  posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
			}
			if((options.channelMode & 32) == 32) {
				  sstprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
			}

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

		beforerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
		formerrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
		middlerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
			  latterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
			  afterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

		poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
		output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

		for (int idx = 0; idx < 3; idx++) {
			xMExp[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
			xExp[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
			xPoolIndex[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
			att_y[idx] = NewTensor<xpu>(Shape2(1, _attSize), d_zero);
		}
		xSum = NewTensor<xpu>(Shape2(1, _attSize), d_zero);

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

			if((options.channelMode & 2) == 2) {
			   _randomWord.GetEmb(feature.randomWord, randomWordprime[idx]);
			}
			if((options.channelMode & 4) == 4) {
				characterprime[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));

				  for(int i=0;i<feature.characters.size();i++) {
					   _character.GetEmb(feature.characters[i], characterprime[idx][i]);
				  }

				characterWindow[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _characterWindowSize), 0.0));

				  rnn_character_left_iy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left_oy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left_fy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left_mcy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left_cy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left_my[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_left[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));

				  rnn_character_right_iy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right_oy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right_fy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right_mcy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right_cy[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right_my[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));
				  rnn_character_right[idx] = (NewTensor<xpu>(Shape3(feature.characters.size(), 1, _otherDim), 0.0));

			}
			if((options.channelMode & 16) == 16) {
				   _pos.GetEmb(feature.pos, posprime[idx]);
			}
			if((options.channelMode & 32) == 32) {
				   _sst.GetEmb(feature.sst, sstprime[idx]);
			}
		}

		windowlized(wordprime, wordWindow, _wordcontext);

			if((options.channelMode & 2) == 2) {
				windowlized(randomWordprime, randomWordWindow, _randomWordContext);
			}

		for(int idx=0;idx<seq_size;idx++) {
			  vector<Tensor<xpu, 2, dtype> > v_Input;
			  v_Input.push_back(wordWindow[idx]);

				if((options.channelMode & 2) == 2) {
					v_Input.push_back(randomWordWindow[idx]);
				}
				if((options.channelMode & 4) == 4) {

					windowlized(characterprime[idx], characterWindow[idx], _charactercontext);

					  _rnn_character_left.ComputeForwardScore(characterWindow[idx], rnn_character_left_iy[idx], rnn_character_left_oy[idx], rnn_character_left_fy[idx],
							  rnn_character_left_mcy[idx], rnn_character_left_cy[idx], rnn_character_left_my[idx], rnn_character_left[idx]);
					  _rnn_character_right.ComputeForwardScore(characterWindow[idx], rnn_character_right_iy[idx], rnn_character_right_oy[idx], rnn_character_right_fy[idx],
								  rnn_character_right_mcy[idx], rnn_character_right_cy[idx], rnn_character_right_my[idx], rnn_character_right[idx]);

					  v_Input.push_back(rnn_character_left[idx][rnn_character_left[idx].size(0)-1]);
					  v_Input.push_back(rnn_character_right[idx][0]);
				}
				if((options.channelMode & 16) == 16) {
					v_Input.push_back(posprime[idx]);
				}
				if((options.channelMode & 32) == 32) {
					v_Input.push_back(sstprime[idx]);
				}

				concat(v_Input, input[idx]);
		}

		_rnn_left.ComputeForwardScore(input, rnn_hidden_left_iy, rnn_hidden_left_oy, rnn_hidden_left_fy,
				  rnn_hidden_left_mcy, rnn_hidden_left_cy, rnn_hidden_left_my, rnn_hidden_left);
		_rnn_right.ComputeForwardScore(input, rnn_hidden_right_iy, rnn_hidden_right_oy, rnn_hidden_right_fy,
					  rnn_hidden_right_mcy, rnn_hidden_right_cy, rnn_hidden_right_my, rnn_hidden_right);


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

		concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], beforerepresent);

		offset += _poolfunctions;
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

		concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], formerrepresent);

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

		concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], middlerepresent);

		offset += _poolfunctions;
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

		concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], latterrepresent);

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

		concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], afterrepresent);

	      // attention
	      vector<Tensor<xpu, 2, dtype> > att_x;
	      att_x.push_back(beforerepresent);
	      att_x.push_back(middlerepresent);
	      att_x.push_back(afterrepresent);
			_attention.ComputeForwardScore(att_x, formerrepresent, latterrepresent,
					xMExp, xExp, xSum,
					xPoolIndex, att_y);

	      concat8(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent,
	    		  att_y[0], att_y[1], att_y[2],
	    		  poolmerge);

		_olayer_linear.ComputeForwardScore(poolmerge, output);

		// decode algorithm
		  dtype cost = softmax_cost(output, example.m_labels);

		//release
		FreeSpace(&wordprime);
		FreeSpace(&wordWindow);

			if((options.channelMode & 2) == 2) {
				  FreeSpace(&randomWordprime);
				  FreeSpace(&randomWordWindow);
			}
			if((options.channelMode & 4) == 4) {
				  for (int idx = 0; idx < seq_size; idx++) {

						  FreeSpace(&(characterprime[idx]));
						  FreeSpace(&(characterWindow[idx]));

						  FreeSpace(&(rnn_character_left_iy[idx]));
						  FreeSpace(&(rnn_character_left_oy[idx]));
						  FreeSpace(&(rnn_character_left_fy[idx]));
						  FreeSpace(&(rnn_character_left_mcy[idx]));
						  FreeSpace(&(rnn_character_left_cy[idx]));
						  FreeSpace(&(rnn_character_left_my[idx]));
						  FreeSpace(&(rnn_character_left[idx]));

						  FreeSpace(&(rnn_character_right_iy[idx]));
						  FreeSpace(&(rnn_character_right_oy[idx]));
						  FreeSpace(&(rnn_character_right_fy[idx]));
						  FreeSpace(&(rnn_character_right_mcy[idx]));
						  FreeSpace(&(rnn_character_right_cy[idx]));
						  FreeSpace(&(rnn_character_right_my[idx]));
						  FreeSpace(&(rnn_character_right[idx]));
				  }
			}
			if((options.channelMode & 16) == 16) {
				  FreeSpace(&posprime);
			}
			if((options.channelMode & 32) == 32) {
				  FreeSpace(&sstprime);
			}

		FreeSpace(&input);

		FreeSpace(&rnn_hidden_left_iy);
		FreeSpace(&rnn_hidden_left_oy);
		FreeSpace(&rnn_hidden_left_fy);
		FreeSpace(&rnn_hidden_left_mcy);
		FreeSpace(&rnn_hidden_left_cy);
		FreeSpace(&rnn_hidden_left_my);
		FreeSpace(&rnn_hidden_left);

		FreeSpace(&rnn_hidden_right_iy);
		FreeSpace(&rnn_hidden_right_oy);
		FreeSpace(&rnn_hidden_right_fy);
		FreeSpace(&rnn_hidden_right_mcy);
		FreeSpace(&rnn_hidden_right_cy);
		FreeSpace(&rnn_hidden_right_my);
		FreeSpace(&rnn_hidden_right);

		FreeSpace(&rnn_hidden_merge);

		FreeSpace(&project);

		for (int idm = 0; idm < _poolmanners; idm++) {
		  FreeSpace(&(pool[idm]));
		  FreeSpace(&(poolIndex[idm]));
		}

		FreeSpace(&beforerepresent);
		FreeSpace(&formerrepresent);
		FreeSpace(&middlerepresent);
			  FreeSpace(&latterrepresent);
			  FreeSpace(&afterrepresent);

		FreeSpace(&poolmerge);
		FreeSpace(&output);

		for (int idx = 0; idx < 3; idx++) {
			FreeSpace(&(xMExp[idx]));
			FreeSpace(&(xExp[idx]));
			FreeSpace(&(xPoolIndex[idx]));
			FreeSpace(&(att_y[idx]));
		}
		FreeSpace(&xSum);


		return cost;

}

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);

	if((options.channelMode & 2) == 2) {
	    _randomWord.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}
	if((options.channelMode & 4) == 4) {
	    _rnn_character_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	    _rnn_character_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_character.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}
	if((options.channelMode & 16) == 16) {
		_pos.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}
	if((options.channelMode & 32) == 32) {
		_sst.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}

	_attention.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int idx = 0; idx < Wd.size(1); idx++)
      idCols.push_back(idx);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 3, dtype> Wd, Tensor<xpu, 3, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols, idThirds;
    idRows.clear();
    idCols.clear();
    idThirds.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int i = 0; i < Wd.size(1); i++)
      idCols.push_back(i);
    for (int i = 0; i < Wd.size(2); i++)
      idThirds.push_back(i);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());
    random_shuffle(idThirds.begin(), idThirds.end());

    int check_i = idRows[0], check_j = idCols[0], check_k = idThirds[0];

    dtype orginValue = Wd[check_i][check_j][check_k];

    Wd[check_i][check_j][check_k] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j][check_k] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j][check_k];

    printf("Iteration %d, Checking gradient for %s[%d][%d][%d]:\t", iter, mark.c_str(), check_i, check_j, check_k);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j][check_k] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {
    if(indexes.size() == 0) return;
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    static hash_set<int>::iterator it;
    if (bRow) {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idRows.push_back(*it);
      for (int idx = 0; idx < Wd.size(1); idx++)
        idCols.push_back(idx);
    } else {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idCols.push_back(*it);
      for (int idx = 0; idx < Wd.size(0); idx++)
        idRows.push_back(idx);
    }

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgrads(const vector<Example>& examples, int iter) {

    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

    checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

    checkgrad(examples, _rnn_left._lstm_output._W1, _rnn_left._lstm_output._gradW1, "_rnn_left._lstm_output._W1", iter);
    checkgrad(examples, _rnn_left._lstm_output._W2, _rnn_left._lstm_output._gradW2, "_rnn_left._lstm_output._W2", iter);
    checkgrad(examples, _rnn_left._lstm_output._W3, _rnn_left._lstm_output._gradW3, "_rnn_left._lstm_output._W3", iter);
    checkgrad(examples, _rnn_left._lstm_output._b, _rnn_left._lstm_output._gradb, "_rnn_left._lstm_output._b", iter);
    checkgrad(examples, _rnn_left._lstm_input._W1, _rnn_left._lstm_input._gradW1, "_rnn_left._lstm_input._W1", iter);
    checkgrad(examples, _rnn_left._lstm_input._W2, _rnn_left._lstm_input._gradW2, "_rnn_left._lstm_input._W2", iter);
    checkgrad(examples, _rnn_left._lstm_input._W3, _rnn_left._lstm_input._gradW3, "_rnn_left._lstm_input._W3", iter);
    checkgrad(examples, _rnn_left._lstm_input._b, _rnn_left._lstm_input._gradb, "_rnn_left._lstm_input._b", iter);
    checkgrad(examples, _rnn_left._lstm_forget._W1, _rnn_left._lstm_forget._gradW1, "_rnn_left._lstm_forget._W1", iter);
    checkgrad(examples, _rnn_left._lstm_forget._W2, _rnn_left._lstm_forget._gradW2, "_rnn_left._lstm_forget._W2", iter);
    checkgrad(examples, _rnn_left._lstm_forget._W3, _rnn_left._lstm_forget._gradW3, "_rnn_left._lstm_forget._W3", iter);
    checkgrad(examples, _rnn_left._lstm_forget._b, _rnn_left._lstm_forget._gradb, "_rnn_left._lstm_forget._b", iter);
    checkgrad(examples, _rnn_left._lstm_cell._WL, _rnn_left._lstm_cell._gradWL, "_rnn_left._lstm_cell._WL", iter);
    checkgrad(examples, _rnn_left._lstm_cell._WR, _rnn_left._lstm_cell._gradWR, "_rnn_left._lstm_cell._WR", iter);
    checkgrad(examples, _rnn_left._lstm_cell._b, _rnn_left._lstm_cell._gradb, "_rnn_left._lstm_cell._b", iter);

    checkgrad(examples, _rnn_right._lstm_output._W1, _rnn_right._lstm_output._gradW1, "_rnn_right._lstm_output._W1", iter);
    checkgrad(examples, _rnn_right._lstm_output._W2, _rnn_right._lstm_output._gradW2, "_rnn_right._lstm_output._W2", iter);
    checkgrad(examples, _rnn_right._lstm_output._W3, _rnn_right._lstm_output._gradW3, "_rnn_right._lstm_output._W3", iter);
    checkgrad(examples, _rnn_right._lstm_output._b, _rnn_right._lstm_output._gradb, "_rnn_right._lstm_output._b", iter);
    checkgrad(examples, _rnn_right._lstm_input._W1, _rnn_right._lstm_input._gradW1, "_rnn_right._lstm_input._W1", iter);
    checkgrad(examples, _rnn_right._lstm_input._W2, _rnn_right._lstm_input._gradW2, "_rnn_right._lstm_input._W2", iter);
    checkgrad(examples, _rnn_right._lstm_input._W3, _rnn_right._lstm_input._gradW3, "_rnn_right._lstm_input._W3", iter);
    checkgrad(examples, _rnn_right._lstm_input._b, _rnn_right._lstm_input._gradb, "_rnn_right._lstm_input._b", iter);
    checkgrad(examples, _rnn_right._lstm_forget._W1, _rnn_right._lstm_forget._gradW1, "_rnn_right._lstm_forget._W1", iter);
    checkgrad(examples, _rnn_right._lstm_forget._W2, _rnn_right._lstm_forget._gradW2, "_rnn_right._lstm_forget._W2", iter);
    checkgrad(examples, _rnn_right._lstm_forget._W3, _rnn_right._lstm_forget._gradW3, "_rnn_right._lstm_forget._W3", iter);
    checkgrad(examples, _rnn_right._lstm_forget._b, _rnn_right._lstm_forget._gradb, "_rnn_right._lstm_forget._b", iter);
    checkgrad(examples, _rnn_right._lstm_cell._WL, _rnn_right._lstm_cell._gradWL, "_rnn_right._lstm_cell._WL", iter);
    checkgrad(examples, _rnn_right._lstm_cell._WR, _rnn_right._lstm_cell._gradWR, "_rnn_right._lstm_cell._WR", iter);
    checkgrad(examples, _rnn_right._lstm_cell._b, _rnn_right._lstm_cell._gradb, "_rnn_right._lstm_cell._b", iter);

    checkgrad(examples, _rnn_character_left._lstm_output._W1, _rnn_character_left._lstm_output._gradW1, "_rnn_character_left._lstm_output._W1", iter);
    checkgrad(examples, _rnn_character_left._lstm_output._W2, _rnn_character_left._lstm_output._gradW2, "_rnn_character_left._lstm_output._W2", iter);
    checkgrad(examples, _rnn_character_left._lstm_output._W3, _rnn_character_left._lstm_output._gradW3, "_rnn_character_left._lstm_output._W3", iter);
    checkgrad(examples, _rnn_character_left._lstm_output._b, _rnn_character_left._lstm_output._gradb, "_rnn_character_left._lstm_output._b", iter);
    checkgrad(examples, _rnn_character_left._lstm_input._W1, _rnn_character_left._lstm_input._gradW1, "_rnn_character_left._lstm_input._W1", iter);
    checkgrad(examples, _rnn_character_left._lstm_input._W2, _rnn_character_left._lstm_input._gradW2, "_rnn_character_left._lstm_input._W2", iter);
    checkgrad(examples, _rnn_character_left._lstm_input._W3, _rnn_character_left._lstm_input._gradW3, "_rnn_character_left._lstm_input._W3", iter);
    checkgrad(examples, _rnn_character_left._lstm_input._b, _rnn_character_left._lstm_input._gradb, "_rnn_character_left._lstm_input._b", iter);
    checkgrad(examples, _rnn_character_left._lstm_forget._W1, _rnn_character_left._lstm_forget._gradW1, "_rnn_character_left._lstm_forget._W1", iter);
    checkgrad(examples, _rnn_character_left._lstm_forget._W2, _rnn_character_left._lstm_forget._gradW2, "_rnn_character_left._lstm_forget._W2", iter);
    checkgrad(examples, _rnn_character_left._lstm_forget._W3, _rnn_character_left._lstm_forget._gradW3, "_rnn_character_left._lstm_forget._W3", iter);
    checkgrad(examples, _rnn_character_left._lstm_forget._b, _rnn_character_left._lstm_forget._gradb, "_rnn_character_left._lstm_forget._b", iter);
    checkgrad(examples, _rnn_character_left._lstm_cell._WL, _rnn_character_left._lstm_cell._gradWL, "_rnn_character_left._lstm_cell._WL", iter);
    checkgrad(examples, _rnn_character_left._lstm_cell._WR, _rnn_character_left._lstm_cell._gradWR, "_rnn_character_left._lstm_cell._WR", iter);
    checkgrad(examples, _rnn_character_left._lstm_cell._b, _rnn_character_left._lstm_cell._gradb, "_rnn_character_left._lstm_cell._b", iter);

    checkgrad(examples, _rnn_character_right._lstm_output._W1, _rnn_character_right._lstm_output._gradW1, "_rnn_character_right._lstm_output._W1", iter);
    checkgrad(examples, _rnn_character_right._lstm_output._W2, _rnn_character_right._lstm_output._gradW2, "_rnn_character_right._lstm_output._W2", iter);
    checkgrad(examples, _rnn_character_right._lstm_output._W3, _rnn_character_right._lstm_output._gradW3, "_rnn_character_right._lstm_output._W3", iter);
    checkgrad(examples, _rnn_character_right._lstm_output._b, _rnn_character_right._lstm_output._gradb, "_rnn_character_right._lstm_output._b", iter);
    checkgrad(examples, _rnn_character_right._lstm_input._W1, _rnn_character_right._lstm_input._gradW1, "_rnn_character_right._lstm_input._W1", iter);
    checkgrad(examples, _rnn_character_right._lstm_input._W2, _rnn_character_right._lstm_input._gradW2, "_rnn_character_right._lstm_input._W2", iter);
    checkgrad(examples, _rnn_character_right._lstm_input._W3, _rnn_character_right._lstm_input._gradW3, "_rnn_character_right._lstm_input._W3", iter);
    checkgrad(examples, _rnn_character_right._lstm_input._b, _rnn_character_right._lstm_input._gradb, "_rnn_character_right._lstm_input._b", iter);
    checkgrad(examples, _rnn_character_right._lstm_forget._W1, _rnn_character_right._lstm_forget._gradW1, "_rnn_character_right._lstm_forget._W1", iter);
    checkgrad(examples, _rnn_character_right._lstm_forget._W2, _rnn_character_right._lstm_forget._gradW2, "_rnn_character_right._lstm_forget._W2", iter);
    checkgrad(examples, _rnn_character_right._lstm_forget._W3, _rnn_character_right._lstm_forget._gradW3, "_rnn_character_right._lstm_forget._W3", iter);
    checkgrad(examples, _rnn_character_right._lstm_forget._b, _rnn_character_right._lstm_forget._gradb, "_rnn_character_right._lstm_forget._b", iter);
    checkgrad(examples, _rnn_character_right._lstm_cell._WL, _rnn_character_right._lstm_cell._gradWL, "_rnn_character_right._lstm_cell._WL", iter);
    checkgrad(examples, _rnn_character_right._lstm_cell._WR, _rnn_character_right._lstm_cell._gradWR, "_rnn_character_right._lstm_cell._WR", iter);
    checkgrad(examples, _rnn_character_right._lstm_cell._b, _rnn_character_right._lstm_cell._gradb, "_rnn_character_right._lstm_cell._b", iter);

    checkgrad(examples, _attention._tri_gates._W1, _attention._tri_gates._gradW1, "_attention._tri_gates._W1", iter);
    checkgrad(examples, _attention._tri_gates._W2, _attention._tri_gates._gradW2, "_attention._tri_gates._W2", iter);
    checkgrad(examples, _attention._tri_gates._W3, _attention._tri_gates._gradW3, "_attention._tri_gates._W3", iter);
    checkgrad(examples, _attention._tri_gates._b, _attention._tri_gates._gradb, "_attention._tri_gates._b", iter);
    checkgrad(examples, _attention._uni_gates._W, _attention._uni_gates._gradW, "_attention._uni_gates._W", iter);
    checkgrad(examples, _attention._uni_gates._b, _attention._uni_gates._gradb, "_attention._uni_gates._b", iter);

    checkgrad(examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);
    checkgrad(examples, _randomWord._E, _randomWord._gradE, "_randomWord._E", iter, _randomWord._indexers);
    checkgrad(examples, _character._E, _character._gradE, "_character._E", iter, _character._indexers);
    checkgrad(examples, _pos._E, _pos._gradE, "_pos._E", iter, _pos._indexers);
    checkgrad(examples, _sst._E, _sst._gradE, "_sst._E", iter, _sst._indexers);
  }


  void concat8(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2,
		  Tensor<xpu, 2, dtype> w3, Tensor<xpu, 2, dtype> w4,
		  Tensor<xpu, 2, dtype> w5, Tensor<xpu, 2, dtype> w6,
		  Tensor<xpu, 2, dtype> w7, Tensor<xpu, 2, dtype> w8,
		  Tensor<xpu, 2, dtype> w) {
    int row = w.size(0);
    int col = w.size(1);
    int col1 = w1.size(1);
    int col2 = w2.size(1);
    int col3 = w3.size(1);
    int col4 = w4.size(1);
    int col5 = w5.size(1);
    int col6 = w6.size(1);
    int col7 = w7.size(1);
    int col8 = w8.size(1);
    if (col1 + col2 + col3 + col4 + col5 + col6 + col7 + col8 != col) {
      std::cerr << "col check error!" << std::endl;
      return;
    }
    int offset;
    w = 0.0;
    for (int idx = 0; idx < row; idx++) {
      offset = 0;
      for (int idy = 0; idy < col1; idy++) {
        w[idx][offset] += w1[idx][idy];
        offset++;
      }
      for (int idy = 0; idy < col2; idy++) {
        w[idx][offset] += w2[idx][idy];
        offset++;
      }
      for (int idy = 0; idy < col3; idy++) {
        w[idx][offset] += w3[idx][idy];
        offset++;
      }
      for (int idy = 0; idy < col4; idy++) {
        w[idx][offset] += w4[idx][idy];
        offset++;
      }
      for (int idy = 0; idy < col5; idy++) {
        w[idx][offset] += w5[idx][idy];
        offset++;
      }
      for (int idy = 0; idy < col6; idy++) {
        w[idx][offset] += w6[idx][idy];
        offset++;
      }
      for (int idy = 0; idy < col7; idy++) {
        w[idx][offset] += w7[idx][idy];
        offset++;
      }
      for (int idy = 0; idy < col8; idy++) {
        w[idx][offset] += w8[idx][idy];
        offset++;
      }
    }
    return;
  }

  void unconcat8(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2,
		  Tensor<xpu, 2, dtype> w3, Tensor<xpu, 2, dtype> w4,
		  Tensor<xpu, 2, dtype> w5, Tensor<xpu, 2, dtype> w6,
		  Tensor<xpu, 2, dtype> w7, Tensor<xpu, 2, dtype> w8,
		  Tensor<xpu, 2, dtype> w, bool bclear = false) {
    int row = w.size(0);
    int col = w.size(1);
    int col1 = w1.size(1);
    int col2 = w2.size(1);
    int col3 = w3.size(1);
    int col4 = w4.size(1);
    int col5 = w5.size(1);
    int col6 = w6.size(1);
    int col7 = w7.size(1);
    int col8 = w8.size(1);
    if (col1 + col2 + col3 + col4 + col5 + col6 + col7 + col8 != col) {
      std::cerr << "col check error!" << std::endl;
      return;
    }
    int offset;
    if (bclear) {
      w1 = 0.0;
      w2 = 0.0;
      w3 = 0.0;
      w4 = 0.0;
      w5 = 0.0;
      w6 = 0.0;
      w7 = 0.0;
      w8 = 0.0;
    }
    for (int idx = 0; idx < row; idx++) {
      offset = 0;
      for (int idy = 0; idy < col1; idy++) {
        w1[idx][idy] += w[idx][offset];
        offset++;
      }
      for (int idy = 0; idy < col2; idy++) {
        w2[idx][idy] += w[idx][offset];
        offset++;
      }
      for (int idy = 0; idy < col3; idy++) {
        w3[idx][idy] += w[idx][offset];
        offset++;
      }
      for (int idy = 0; idy < col4; idy++) {
        w4[idx][idy] += w[idx][offset];
        offset++;
      }
      for (int idy = 0; idy < col5; idy++) {
        w5[idx][idy] += w[idx][offset];
        offset++;
      }
      for (int idy = 0; idy < col6; idy++) {
        w6[idx][idy] += w[idx][offset];
        offset++;
      }
      for (int idy = 0; idy < col7; idy++) {
        w7[idx][idy] += w[idx][offset];
        offset++;
      }
      for (int idy = 0; idy < col8; idy++) {
        w8[idx][idy] += w[idx][offset];
        offset++;
      }
    }
    return;
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
