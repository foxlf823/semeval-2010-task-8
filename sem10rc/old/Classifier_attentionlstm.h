/*
 * Classifier.h
 *
 *  Created on: Dec 28, 2015
 *      Author: fox
 */

#ifndef CLASSIFIER_LSTM_H_
#define CLASSIFIER_LSTM_H_

#include <iostream>

#include <assert.h>

#include "Example.h"
#include "Metric.h"
#include "N3L.h"
#include "Options.h"
#include "AttentionPooling.h"
#include "utils.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class Classifier_attentionlstm {
public:
	Options options;


	int _wordDim;
	int _outputSize;
	int _hidden_input_size;

	// input
	// word channel
	LookupTable<xpu> _words;
	LSTM<xpu> unit_before;
	LSTM<xpu> unit_middle;
	LSTM<xpu> unit_after;
	AttentionPooling<xpu> unit_att_before;
	AttentionPooling<xpu> unit_att_middle;
	AttentionPooling<xpu> unit_att_after;
	LSTM<xpu> unit_entityFormer;
	LSTM<xpu> unit_entityLatter;

	LookupTable<xpu> _wordnet;
	LSTM<xpu> unit_before_wordnet;
	LSTM<xpu> unit_middle_wordnet;
	LSTM<xpu> unit_after_wordnet;
	AttentionPooling<xpu> unit_att_before_wordnet;
	AttentionPooling<xpu> unit_att_middle_wordnet;
	AttentionPooling<xpu> unit_att_after_wordnet;
	LSTM<xpu> unit_entityFormer_wordnet;
	LSTM<xpu> unit_entityLatter_wordnet;

	LookupTable<xpu> _brown;
	LSTM<xpu> unit_before_brown;
	LSTM<xpu> unit_middle_brown;
	LSTM<xpu> unit_after_brown;
	AttentionPooling<xpu> unit_att_before_brown;
	AttentionPooling<xpu> unit_att_middle_brown;
	AttentionPooling<xpu> unit_att_after_brown;
	LSTM<xpu> unit_entityFormer_brown;
	LSTM<xpu> unit_entityLatter_brown;

	LookupTable<xpu> _bigram;
	LSTM<xpu> unit_before_bigram;
	LSTM<xpu> unit_middle_bigram;
	LSTM<xpu> unit_after_bigram;
	AttentionPooling<xpu> unit_att_before_bigram;
	AttentionPooling<xpu> unit_att_middle_bigram;
	AttentionPooling<xpu> unit_att_after_bigram;
	LSTM<xpu> unit_entityFormer_bigram;
	LSTM<xpu> unit_entityLatter_bigram;

	LookupTable<xpu> _pos;
	LSTM<xpu> unit_before_pos;
	LSTM<xpu> unit_middle_pos;
	LSTM<xpu> unit_after_pos;
	AttentionPooling<xpu> unit_att_before_pos;
	AttentionPooling<xpu> unit_att_middle_pos;
	AttentionPooling<xpu> unit_att_after_pos;
	LSTM<xpu> unit_entityFormer_pos;
	LSTM<xpu> unit_entityLatter_pos;

	LookupTable<xpu> _sst;
	LSTM<xpu> unit_before_sst;
	LSTM<xpu> unit_middle_sst;
	LSTM<xpu> unit_after_sst;
	AttentionPooling<xpu> unit_att_before_sst;
	AttentionPooling<xpu> unit_att_middle_sst;
	AttentionPooling<xpu> unit_att_after_sst;
	LSTM<xpu> unit_entityFormer_sst;
	LSTM<xpu> unit_entityLatter_sst;

	// hidden
	UniLayer<xpu> hidden_layer;

	// output
	UniLayer<xpu> output_layer;
	// softmax loss corresponds no class in n3l

	Metric _eval;

	bool bWord = false;
	bool bWordnet = false;
	bool bBrown = false;
	bool bBigram = false;
	bool bPos = false;
	bool bSst = false;

	Classifier_attentionlstm(const Options& options):options(options) {

	}
/*	virtual ~Classifier_attentionlstm() {

	}*/

	void release() {

		output_layer.release();

		hidden_layer.release();


		if(bWord) {
			unit_entityFormer.release();
			unit_entityLatter.release();
			if(options.attention) {
				unit_att_before.release();
				unit_att_middle.release();
				unit_att_after.release();
			}
			unit_before.release();
			unit_middle.release();
			unit_after.release();
			_words.release();
		}

		if(bWordnet) {
			_wordnet.release();
			unit_before_wordnet.release();
			unit_middle_wordnet.release();
			unit_after_wordnet.release();
			if(options.attention) {
				unit_att_before_wordnet.release();
				unit_att_middle_wordnet.release();
				unit_att_after_wordnet.release();
			}
			unit_entityFormer_wordnet.release();
			unit_entityLatter_wordnet.release();
		}

		if(bBrown) {
			_brown.release();
			unit_before_brown.release();
			unit_middle_brown.release();
			unit_after_brown.release();
			if(options.attention) {
				unit_att_before_brown.release();
				unit_att_middle_brown.release();
				unit_att_after_brown.release();
			}
			unit_entityFormer_brown.release();
			unit_entityLatter_brown.release();
		}

		if(bBigram) {
			_bigram.release();
			unit_before_bigram.release();
			unit_middle_bigram.release();
			unit_after_bigram.release();
			if(options.attention) {
				unit_att_before_bigram.release();
				unit_att_middle_bigram.release();
				unit_att_after_bigram.release();
			}
			unit_entityFormer_bigram.release();
			unit_entityLatter_bigram.release();
		}

		if(bPos) {
			_pos.release();
			unit_before_pos.release();
			unit_middle_pos.release();
			unit_after_pos.release();
			if(options.attention) {
				unit_att_before_pos.release();
				unit_att_middle_pos.release();
				unit_att_after_pos.release();
			}
			unit_entityFormer_pos.release();
			unit_entityLatter_pos.release();
		}

		if(bSst) {
			_sst.release();
			unit_before_sst.release();
			unit_middle_sst.release();
			unit_after_sst.release();
			if(options.attention) {
				unit_att_before_sst.release();
				unit_att_middle_sst.release();
				unit_att_after_sst.release();
			}
			unit_entityFormer_sst.release();
			unit_entityLatter_sst.release();
		}
	}


	void init(const NRMat<dtype>& wordEmb, const NRMat<dtype>& wordnetEmb, const NRMat<dtype>& brownEmb,
			const NRMat<dtype>& bigramEmb, const NRMat<dtype>& posEmb, const NRMat<dtype>& sstEmb) {
		int model = options.channelMode;
		if((model & 1) == 1)
			bWord = true;
		if((model & 2) == 2)
			bWordnet = true;
		if((model & 4) == 4)
			bBrown = true;
		if((model & 8) == 8)
			bBigram = true;
		if((model & 16) == 16)
			bPos = true;
		if((model & 32) == 32)
			bSst = true;

		assert(bWord==true);

	    _wordDim = wordEmb.ncols();
	    _outputSize = MAX_RELATION;

	    if(bWord) {
			_words.initial(wordEmb);
			_words.setEmbFineTune(options.wordEmbFineTune);
			unit_entityFormer.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_entityLatter.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_before.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_middle.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_after.initial(options.context_embsize, _wordDim, (int)time(0));
			if(options.attention) {
				unit_att_before.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_middle.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_after.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
			}
	    }

	    if(bWordnet) {
			_wordnet.initial(wordnetEmb);
			_wordnet.setEmbFineTune(true);
			unit_entityFormer_wordnet.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_entityLatter_wordnet.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_before_wordnet.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_middle_wordnet.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_after_wordnet.initial(options.context_embsize, _wordDim, (int)time(0));
			if(options.attention) {
				unit_att_before_wordnet.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_middle_wordnet.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_after_wordnet.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
			}
	    }

	    if(bBrown) {
			_brown.initial(brownEmb);
			_brown.setEmbFineTune(true);
			unit_entityFormer_brown.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_entityLatter_brown.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_before_brown.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_middle_brown.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_after_brown.initial(options.context_embsize, _wordDim, (int)time(0));
			if(options.attention) {
				unit_att_before_brown.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_middle_brown.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_after_brown.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
			}
	    }

	    if(bBigram) {
			_bigram.initial(bigramEmb);
			_bigram.setEmbFineTune(true);
			unit_entityFormer_bigram.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_entityLatter_bigram.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_before_bigram.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_middle_bigram.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_after_bigram.initial(options.context_embsize, _wordDim, (int)time(0));
			if(options.attention) {
				unit_att_before_bigram.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_middle_bigram.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_after_bigram.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
			}
	    }

	    if(bPos) {
			_pos.initial(posEmb);
			_pos.setEmbFineTune(true);
			unit_entityFormer_pos.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_entityLatter_pos.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_before_pos.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_middle_pos.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_after_pos.initial(options.context_embsize, _wordDim, (int)time(0));
			if(options.attention) {
				unit_att_before_pos.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_middle_pos.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_after_pos.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
			}
	    }

	    if(bSst) {
			_sst.initial(sstEmb);
			_sst.setEmbFineTune(true);
			unit_entityFormer_sst.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_entityLatter_sst.initial(options.entity_embsize, _wordDim, (int)time(0));
			unit_before_sst.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_middle_sst.initial(options.context_embsize, _wordDim, (int)time(0));
			unit_after_sst.initial(options.context_embsize, _wordDim, (int)time(0));
			if(options.attention) {
				unit_att_before_sst.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_middle_sst.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
				unit_att_after_sst.initial(options.context_embsize, options.context_embsize, true, (int)time(0));
			}
	    }

	    _hidden_input_size = 0;
	    int lstmSize = options.context_embsize*3+options.entity_embsize*2;

	    if(bWord) {
	    	_hidden_input_size += lstmSize;
	    }
	    if(bWordnet) {
	    	_hidden_input_size += lstmSize;
	    }
	    if(bBrown) {
	    	_hidden_input_size += lstmSize;
	    }
	    if(bBigram) {
	    	_hidden_input_size += lstmSize;
	    }
	    if(bPos) {
	    	_hidden_input_size += lstmSize;
	    }
	    if(bSst) {
	    	_hidden_input_size += lstmSize;
	    }


	    hidden_layer.initial(options.hiddenSize, _hidden_input_size, true, (int)time(0), 0);

	    output_layer.initial(_outputSize, options.hiddenSize, true, (int)time(0), 2);

	}

	void predict(const Example& example, vector<double>& scores) {
		int beforeSize = example.m_before.size();
		int enFormerSize = example.m_entityFormer.size();
		int enLatterSize = example.m_entityLatter.size();
		int middleSize = example.m_middle.size();
		int afterSize = example.m_after.size();

		// word channel
		vector<Tensor<xpu, 2, dtype> > input_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > input_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > input_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > input_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > input_after(afterSize);
		if(bWord) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				input_entityFormer[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_words.GetEmb(example.m_entityFormer[idx], input_entityFormer[idx]);
			}
			for (int idx = 0; idx < enLatterSize; idx++) {
				input_entityLatter[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_words.GetEmb(example.m_entityLatter[idx], input_entityLatter[idx]);
			}
			for (int idx = 0; idx < beforeSize; idx++) {
				input_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_words.GetEmb(example.m_before[idx], input_before[idx]);
			}
			for (int idx = 0; idx < middleSize; idx++) {
				input_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_words.GetEmb(example.m_middle[idx], input_middle[idx]);
			}
			for (int idx = 0; idx < afterSize; idx++) {
				input_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_words.GetEmb(example.m_after[idx], input_after[idx]);
			}
		}


		// wordnet channel
		vector<Tensor<xpu, 2, dtype> > input_entityFormer_wordnet(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > input_entityLatter_wordnet(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > input_before_wordnet(beforeSize);
		vector<Tensor<xpu, 2, dtype> > input_middle_wordnet(middleSize);
		vector<Tensor<xpu, 2, dtype> > input_after_wordnet(afterSize);
		if(bWordnet) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				input_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_wordnet.GetEmb(example.m_entityFormer_wordnet[idx], input_entityFormer_wordnet[idx]);
			}
			for (int idx = 0; idx < enLatterSize; idx++) {
				input_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_wordnet.GetEmb(example.m_entityLatter_wordnet[idx], input_entityLatter_wordnet[idx]);
			}
			for (int idx = 0; idx < beforeSize; idx++) {
				input_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_wordnet.GetEmb(example.m_before_wordnet[idx], input_before_wordnet[idx]);
			}
			for (int idx = 0; idx < middleSize; idx++) {
				input_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_wordnet.GetEmb(example.m_middle_wordnet[idx], input_middle_wordnet[idx]);
			}
			for (int idx = 0; idx < afterSize; idx++) {
				input_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_wordnet.GetEmb(example.m_after_wordnet[idx], input_after_wordnet[idx]);
			}
		}


		// brown channel
		vector<Tensor<xpu, 2, dtype> > input_entityFormer_brown(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > input_entityLatter_brown(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > input_before_brown(beforeSize);
		vector<Tensor<xpu, 2, dtype> > input_middle_brown(middleSize);
		vector<Tensor<xpu, 2, dtype> > input_after_brown(afterSize);
		if(bBrown) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				input_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_brown.GetEmb(example.m_entityFormer_brown[idx], input_entityFormer_brown[idx]);
			}
			for (int idx = 0; idx < enLatterSize; idx++) {
				input_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_brown.GetEmb(example.m_entityLatter_brown[idx], input_entityLatter_brown[idx]);
			}
			for (int idx = 0; idx < beforeSize; idx++) {
				input_before_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_brown.GetEmb(example.m_before_brown[idx], input_before_brown[idx]);
			}
			for (int idx = 0; idx < middleSize; idx++) {
				input_middle_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_brown.GetEmb(example.m_middle_brown[idx], input_middle_brown[idx]);
			}
			for (int idx = 0; idx < afterSize; idx++) {
				input_after_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_brown.GetEmb(example.m_after_brown[idx], input_after_brown[idx]);
			}
		}


		// bigram channel
		vector<Tensor<xpu, 2, dtype> > input_entityFormer_bigram(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > input_entityLatter_bigram(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > input_before_bigram(beforeSize);
		vector<Tensor<xpu, 2, dtype> > input_middle_bigram(middleSize);
		vector<Tensor<xpu, 2, dtype> > input_after_bigram(afterSize);
		if(bBigram) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				input_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_bigram.GetEmb(example.m_entityFormer_bigram[idx], input_entityFormer_bigram[idx]);
			}
			for (int idx = 0; idx < enLatterSize; idx++) {
				input_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_bigram.GetEmb(example.m_entityLatter_bigram[idx], input_entityLatter_bigram[idx]);
			}
			for (int idx = 0; idx < beforeSize; idx++) {
				input_before_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_bigram.GetEmb(example.m_before_bigram[idx], input_before_bigram[idx]);
			}
			for (int idx = 0; idx < middleSize; idx++) {
				input_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_bigram.GetEmb(example.m_middle_bigram[idx], input_middle_bigram[idx]);
			}
			for (int idx = 0; idx < afterSize; idx++) {
				input_after_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_bigram.GetEmb(example.m_after_bigram[idx], input_after_bigram[idx]);
			}
		}


		// pos channel
		vector<Tensor<xpu, 2, dtype> > input_entityFormer_pos(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > input_entityLatter_pos(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > input_before_pos(beforeSize);
		vector<Tensor<xpu, 2, dtype> > input_middle_pos(middleSize);
		vector<Tensor<xpu, 2, dtype> > input_after_pos(afterSize);
		if(bPos) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				input_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_pos.GetEmb(example.m_entityFormer_pos[idx], input_entityFormer_pos[idx]);
			}
			for (int idx = 0; idx < enLatterSize; idx++) {
				input_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_pos.GetEmb(example.m_entityLatter_pos[idx], input_entityLatter_pos[idx]);
			}
			for (int idx = 0; idx < beforeSize; idx++) {
				input_before_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_pos.GetEmb(example.m_before_pos[idx], input_before_pos[idx]);
			}
			for (int idx = 0; idx < middleSize; idx++) {
				input_middle_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_pos.GetEmb(example.m_middle_pos[idx], input_middle_pos[idx]);
			}
			for (int idx = 0; idx < afterSize; idx++) {
				input_after_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_pos.GetEmb(example.m_after_pos[idx], input_after_pos[idx]);
			}
		}

		// sst channel
		vector<Tensor<xpu, 2, dtype> > input_entityFormer_sst(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > input_entityLatter_sst(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > input_before_sst(beforeSize);
		vector<Tensor<xpu, 2, dtype> > input_middle_sst(middleSize);
		vector<Tensor<xpu, 2, dtype> > input_after_sst(afterSize);
		if(bSst) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				input_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_sst.GetEmb(example.m_entityFormer_sst[idx], input_entityFormer_sst[idx]);
			}
			for (int idx = 0; idx < enLatterSize; idx++) {
				input_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_sst.GetEmb(example.m_entityLatter_sst[idx], input_entityLatter_sst[idx]);
			}
			for (int idx = 0; idx < beforeSize; idx++) {
				input_before_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_sst.GetEmb(example.m_before_sst[idx], input_before_sst[idx]);
			}
			for (int idx = 0; idx < middleSize; idx++) {
				input_middle_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_sst.GetEmb(example.m_middle_sst[idx], input_middle_sst[idx]);
			}
			for (int idx = 0; idx < afterSize; idx++) {
				input_after_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_sst.GetEmb(example.m_after_sst[idx], input_after_sst[idx]);
			}
		}


		Tensor<xpu, 2, dtype> hidden = NewTensor<xpu>(Shape2(1, options.hiddenSize), d_zero);
		Tensor<xpu, 2, dtype> output = NewTensor<xpu>(Shape2(1, _outputSize), d_zero);

		// word channel
		vector<Tensor<xpu, 2, dtype> > iy_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > my_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > y_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > iy_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > my_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > y_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > iy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > oy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > fy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > mcy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > cy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > my_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > y_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > iy_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > oy_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > fy_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > mcy_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > cy_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > my_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > y_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > iy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > oy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > fy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > cy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > my_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > y_after(afterSize);

		vector<Tensor<xpu, 2, dtype> > xMExp_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xExp_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_before(beforeSize);
		Tensor<xpu, 2, dtype> xSum_before;
		Tensor<xpu, 2, dtype> y_att_before;
		vector<Tensor<xpu, 2, dtype> > xMExp_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > xExp_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle(middleSize);
		Tensor<xpu, 2, dtype> xSum_middle;
		Tensor<xpu, 2, dtype> y_att_middle;
		vector<Tensor<xpu, 2, dtype> > xMExp_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > xExp_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_after(afterSize);
		Tensor<xpu, 2, dtype> xSum_after;
		Tensor<xpu, 2, dtype> y_att_after;
		if(bWord) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				iy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityFormer.ComputeForwardScore(input_entityFormer, iy_entityFormer, oy_entityFormer,
					fy_entityFormer, mcy_entityFormer,cy_entityFormer, my_entityFormer, y_entityFormer);

			for (int idx = 0; idx < enLatterSize; idx++) {
				iy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityLatter.ComputeForwardScore(input_entityLatter, iy_entityLatter, oy_entityLatter,
					fy_entityLatter, mcy_entityLatter,cy_entityLatter, my_entityLatter, y_entityLatter);

			for (int idx = 0; idx < beforeSize; idx++) {
				iy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_before.ComputeForwardScore(input_before, iy_before, oy_before,
					fy_before, mcy_before,cy_before, my_before, y_before);
			for (int idx = 0; idx < middleSize; idx++) {
				iy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_middle.ComputeForwardScore(input_middle, iy_middle, oy_middle,
					fy_middle, mcy_middle,cy_middle, my_middle, y_middle);
			for (int idx = 0; idx < afterSize; idx++) {
				iy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_after.ComputeForwardScore(input_after, iy_after, oy_after,
					fy_after, mcy_after,cy_after, my_after, y_after);

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					xMExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_before.ComputeForwardScore(y_before, y_before,
						xMExp_before, xExp_before, xSum_before,
						xPoolIndex_before, y_att_before);

				for (int idx = 0; idx < middleSize; idx++) {
					xMExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_middle.ComputeForwardScore(y_middle, y_middle,
						xMExp_middle, xExp_middle, xSum_middle,
						xPoolIndex_middle, y_att_middle);

				for (int idx = 0; idx < afterSize; idx++) {
					xMExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_after.ComputeForwardScore(y_after, y_after,
						xMExp_after, xExp_after, xSum_after,
						xPoolIndex_after, y_att_after);
			}
		}


		// wordnet channel
		vector<Tensor<xpu, 2, dtype> > iy_entityFormer_wordnet(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityFormer_wordnet(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityFormer_wordnet(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityFormer_wordnet(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityFormer_wordnet(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > my_entityFormer_wordnet(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > y_entityFormer_wordnet(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > iy_entityLatter_wordnet(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityLatter_wordnet(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityLatter_wordnet(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityLatter_wordnet(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityLatter_wordnet(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > my_entityLatter_wordnet(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > y_entityLatter_wordnet(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > iy_before_wordnet(beforeSize);
		vector<Tensor<xpu, 2, dtype> > oy_before_wordnet(beforeSize);
		vector<Tensor<xpu, 2, dtype> > fy_before_wordnet(beforeSize);
		vector<Tensor<xpu, 2, dtype> > mcy_before_wordnet(beforeSize);
		vector<Tensor<xpu, 2, dtype> > cy_before_wordnet(beforeSize);
		vector<Tensor<xpu, 2, dtype> > my_before_wordnet(beforeSize);
		vector<Tensor<xpu, 2, dtype> > y_before_wordnet(beforeSize);
		vector<Tensor<xpu, 2, dtype> > iy_middle_wordnet(middleSize);
		vector<Tensor<xpu, 2, dtype> > oy_middle_wordnet(middleSize);
		vector<Tensor<xpu, 2, dtype> > fy_middle_wordnet(middleSize);
		vector<Tensor<xpu, 2, dtype> > mcy_middle_wordnet(middleSize);
		vector<Tensor<xpu, 2, dtype> > cy_middle_wordnet(middleSize);
		vector<Tensor<xpu, 2, dtype> > my_middle_wordnet(middleSize);
		vector<Tensor<xpu, 2, dtype> > y_middle_wordnet(middleSize);
		vector<Tensor<xpu, 2, dtype> > iy_after_wordnet(afterSize);
		vector<Tensor<xpu, 2, dtype> > oy_after_wordnet(afterSize);
		vector<Tensor<xpu, 2, dtype> > fy_after_wordnet(afterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_after_wordnet(afterSize);
		vector<Tensor<xpu, 2, dtype> > cy_after_wordnet(afterSize);
		vector<Tensor<xpu, 2, dtype> > my_after_wordnet(afterSize);
		vector<Tensor<xpu, 2, dtype> > y_after_wordnet(afterSize);

		vector<Tensor<xpu, 2, dtype> > xMExp_before_wordnet(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xExp_before_wordnet(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_before_wordnet(beforeSize);
		Tensor<xpu, 2, dtype> xSum_before_wordnet;
		Tensor<xpu, 2, dtype> y_att_before_wordnet;
		vector<Tensor<xpu, 2, dtype> > xMExp_middle_wordnet(middleSize);
		vector<Tensor<xpu, 2, dtype> > xExp_middle_wordnet(middleSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle_wordnet(middleSize);
		Tensor<xpu, 2, dtype> xSum_middle_wordnet;
		Tensor<xpu, 2, dtype> y_att_middle_wordnet;
		vector<Tensor<xpu, 2, dtype> > xMExp_after_wordnet(afterSize);
		vector<Tensor<xpu, 2, dtype> > xExp_after_wordnet(afterSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_after_wordnet(afterSize);
		Tensor<xpu, 2, dtype> xSum_after_wordnet;
		Tensor<xpu, 2, dtype> y_att_after_wordnet;
		if(bWordnet) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				iy_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityFormer_wordnet.ComputeForwardScore(input_entityFormer_wordnet, iy_entityFormer_wordnet
					, oy_entityFormer_wordnet, fy_entityFormer_wordnet, mcy_entityFormer_wordnet,
					cy_entityFormer_wordnet, my_entityFormer_wordnet, y_entityFormer_wordnet);
			for (int idx = 0; idx < enLatterSize; idx++) {
				iy_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityLatter_wordnet.ComputeForwardScore(input_entityLatter_wordnet, iy_entityLatter_wordnet,
					oy_entityLatter_wordnet,
					fy_entityLatter_wordnet, mcy_entityLatter_wordnet,cy_entityLatter_wordnet,
					my_entityLatter_wordnet, y_entityLatter_wordnet);
			for (int idx = 0; idx < beforeSize; idx++) {
				iy_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_before_wordnet.ComputeForwardScore(input_before_wordnet, iy_before_wordnet, oy_before_wordnet,
					fy_before_wordnet, mcy_before_wordnet,cy_before_wordnet, my_before_wordnet, y_before_wordnet);
			for (int idx = 0; idx < middleSize; idx++) {
				iy_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_middle_wordnet.ComputeForwardScore(input_middle_wordnet, iy_middle_wordnet, oy_middle_wordnet,
					fy_middle_wordnet, mcy_middle_wordnet,cy_middle_wordnet, my_middle_wordnet, y_middle_wordnet);
			for (int idx = 0; idx < afterSize; idx++) {
				iy_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_after_wordnet.ComputeForwardScore(input_after_wordnet, iy_after_wordnet, oy_after_wordnet,
					fy_after_wordnet, mcy_after_wordnet,cy_after_wordnet, my_after_wordnet, y_after_wordnet);

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					xMExp_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_before_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_before_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_before_wordnet.ComputeForwardScore(y_before_wordnet, y_before_wordnet,
						xMExp_before_wordnet, xExp_before_wordnet, xSum_before_wordnet,
						xPoolIndex_before_wordnet, y_att_before_wordnet);

				for (int idx = 0; idx < middleSize; idx++) {
					xMExp_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_middle_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_middle_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_middle_wordnet.ComputeForwardScore(y_middle_wordnet, y_middle_wordnet,
						xMExp_middle_wordnet, xExp_middle_wordnet, xSum_middle_wordnet,
						xPoolIndex_middle_wordnet, y_att_middle_wordnet);

				for (int idx = 0; idx < afterSize; idx++) {
					xMExp_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_after_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_after_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_after_wordnet.ComputeForwardScore(y_after_wordnet, y_after_wordnet,
						xMExp_after_wordnet, xExp_after_wordnet, xSum_after_wordnet,
						xPoolIndex_after_wordnet, y_att_after_wordnet);
			}
		}


		// brown channel
		vector<Tensor<xpu, 2, dtype> > iy_entityFormer_brown(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityFormer_brown(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityFormer_brown(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityFormer_brown(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityFormer_brown(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > my_entityFormer_brown(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > y_entityFormer_brown(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > iy_entityLatter_brown(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityLatter_brown(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityLatter_brown(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityLatter_brown(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityLatter_brown(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > my_entityLatter_brown(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > y_entityLatter_brown(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > iy_before_brown(beforeSize);
		vector<Tensor<xpu, 2, dtype> > oy_before_brown(beforeSize);
		vector<Tensor<xpu, 2, dtype> > fy_before_brown(beforeSize);
		vector<Tensor<xpu, 2, dtype> > mcy_before_brown(beforeSize);
		vector<Tensor<xpu, 2, dtype> > cy_before_brown(beforeSize);
		vector<Tensor<xpu, 2, dtype> > my_before_brown(beforeSize);
		vector<Tensor<xpu, 2, dtype> > y_before_brown(beforeSize);
		vector<Tensor<xpu, 2, dtype> > iy_middle_brown(middleSize);
		vector<Tensor<xpu, 2, dtype> > oy_middle_brown(middleSize);
		vector<Tensor<xpu, 2, dtype> > fy_middle_brown(middleSize);
		vector<Tensor<xpu, 2, dtype> > mcy_middle_brown(middleSize);
		vector<Tensor<xpu, 2, dtype> > cy_middle_brown(middleSize);
		vector<Tensor<xpu, 2, dtype> > my_middle_brown(middleSize);
		vector<Tensor<xpu, 2, dtype> > y_middle_brown(middleSize);
		vector<Tensor<xpu, 2, dtype> > iy_after_brown(afterSize);
		vector<Tensor<xpu, 2, dtype> > oy_after_brown(afterSize);
		vector<Tensor<xpu, 2, dtype> > fy_after_brown(afterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_after_brown(afterSize);
		vector<Tensor<xpu, 2, dtype> > cy_after_brown(afterSize);
		vector<Tensor<xpu, 2, dtype> > my_after_brown(afterSize);
		vector<Tensor<xpu, 2, dtype> > y_after_brown(afterSize);

		vector<Tensor<xpu, 2, dtype> > xMExp_before_brown(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xExp_before_brown(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_before_brown(beforeSize);
		Tensor<xpu, 2, dtype> xSum_before_brown;
		Tensor<xpu, 2, dtype> y_att_before_brown;
		vector<Tensor<xpu, 2, dtype> > xMExp_middle_brown(middleSize);
		vector<Tensor<xpu, 2, dtype> > xExp_middle_brown(middleSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle_brown(middleSize);
		Tensor<xpu, 2, dtype> xSum_middle_brown;
		Tensor<xpu, 2, dtype> y_att_middle_brown;
		vector<Tensor<xpu, 2, dtype> > xMExp_after_brown(afterSize);
		vector<Tensor<xpu, 2, dtype> > xExp_after_brown(afterSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_after_brown(afterSize);
		Tensor<xpu, 2, dtype> xSum_after_brown;
		Tensor<xpu, 2, dtype> y_att_after_brown;
		if(bBrown) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				iy_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityFormer_brown.ComputeForwardScore(input_entityFormer_brown, iy_entityFormer_brown
					, oy_entityFormer_brown, fy_entityFormer_brown, mcy_entityFormer_brown,
					cy_entityFormer_brown, my_entityFormer_brown, y_entityFormer_brown);
			for (int idx = 0; idx < enLatterSize; idx++) {
				iy_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityLatter_brown.ComputeForwardScore(input_entityLatter_brown, iy_entityLatter_brown,
					oy_entityLatter_brown,
					fy_entityLatter_brown, mcy_entityLatter_brown,cy_entityLatter_brown,
					my_entityLatter_brown, y_entityLatter_brown);
			for (int idx = 0; idx < beforeSize; idx++) {
				iy_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_before_brown.ComputeForwardScore(input_before_brown, iy_before_brown, oy_before_brown,
					fy_before_brown, mcy_before_brown,cy_before_brown, my_before_brown, y_before_brown);
			for (int idx = 0; idx < middleSize; idx++) {
				iy_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_middle_brown.ComputeForwardScore(input_middle_brown, iy_middle_brown, oy_middle_brown,
					fy_middle_brown, mcy_middle_brown,cy_middle_brown, my_middle_brown, y_middle_brown);
			for (int idx = 0; idx < afterSize; idx++) {
				iy_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_after_brown.ComputeForwardScore(input_after_brown, iy_after_brown, oy_after_brown,
					fy_after_brown, mcy_after_brown,cy_after_brown, my_after_brown, y_after_brown);

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					xMExp_before_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_before_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_before_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_before_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_before_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_before_brown.ComputeForwardScore(y_before_brown, y_before_brown,
						xMExp_before_brown, xExp_before_brown, xSum_before_brown,
						xPoolIndex_before_brown, y_att_before_brown);

				for (int idx = 0; idx < middleSize; idx++) {
					xMExp_middle_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_middle_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_middle_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_middle_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_middle_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_middle_brown.ComputeForwardScore(y_middle_brown, y_middle_brown,
						xMExp_middle_brown, xExp_middle_brown, xSum_middle_brown,
						xPoolIndex_middle_brown, y_att_middle_brown);

				for (int idx = 0; idx < afterSize; idx++) {
					xMExp_after_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_after_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_after_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_after_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_after_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_after_brown.ComputeForwardScore(y_after_brown, y_after_brown,
						xMExp_after_brown, xExp_after_brown, xSum_after_brown,
						xPoolIndex_after_brown, y_att_after_brown);
			}
		}

		// bigram channel
		vector<Tensor<xpu, 2, dtype> > iy_entityFormer_bigram(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityFormer_bigram(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityFormer_bigram(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityFormer_bigram(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityFormer_bigram(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > my_entityFormer_bigram(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > y_entityFormer_bigram(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > iy_entityLatter_bigram(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityLatter_bigram(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityLatter_bigram(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityLatter_bigram(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityLatter_bigram(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > my_entityLatter_bigram(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > y_entityLatter_bigram(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > iy_before_bigram(beforeSize);
		vector<Tensor<xpu, 2, dtype> > oy_before_bigram(beforeSize);
		vector<Tensor<xpu, 2, dtype> > fy_before_bigram(beforeSize);
		vector<Tensor<xpu, 2, dtype> > mcy_before_bigram(beforeSize);
		vector<Tensor<xpu, 2, dtype> > cy_before_bigram(beforeSize);
		vector<Tensor<xpu, 2, dtype> > my_before_bigram(beforeSize);
		vector<Tensor<xpu, 2, dtype> > y_before_bigram(beforeSize);
		vector<Tensor<xpu, 2, dtype> > iy_middle_bigram(middleSize);
		vector<Tensor<xpu, 2, dtype> > oy_middle_bigram(middleSize);
		vector<Tensor<xpu, 2, dtype> > fy_middle_bigram(middleSize);
		vector<Tensor<xpu, 2, dtype> > mcy_middle_bigram(middleSize);
		vector<Tensor<xpu, 2, dtype> > cy_middle_bigram(middleSize);
		vector<Tensor<xpu, 2, dtype> > my_middle_bigram(middleSize);
		vector<Tensor<xpu, 2, dtype> > y_middle_bigram(middleSize);
		vector<Tensor<xpu, 2, dtype> > iy_after_bigram(afterSize);
		vector<Tensor<xpu, 2, dtype> > oy_after_bigram(afterSize);
		vector<Tensor<xpu, 2, dtype> > fy_after_bigram(afterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_after_bigram(afterSize);
		vector<Tensor<xpu, 2, dtype> > cy_after_bigram(afterSize);
		vector<Tensor<xpu, 2, dtype> > my_after_bigram(afterSize);
		vector<Tensor<xpu, 2, dtype> > y_after_bigram(afterSize);

		vector<Tensor<xpu, 2, dtype> > xMExp_before_bigram(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xExp_before_bigram(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_before_bigram(beforeSize);
		Tensor<xpu, 2, dtype> xSum_before_bigram;
		Tensor<xpu, 2, dtype> y_att_before_bigram;
		vector<Tensor<xpu, 2, dtype> > xMExp_middle_bigram(middleSize);
		vector<Tensor<xpu, 2, dtype> > xExp_middle_bigram(middleSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle_bigram(middleSize);
		Tensor<xpu, 2, dtype> xSum_middle_bigram;
		Tensor<xpu, 2, dtype> y_att_middle_bigram;
		vector<Tensor<xpu, 2, dtype> > xMExp_after_bigram(afterSize);
		vector<Tensor<xpu, 2, dtype> > xExp_after_bigram(afterSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_after_bigram(afterSize);
		Tensor<xpu, 2, dtype> xSum_after_bigram;
		Tensor<xpu, 2, dtype> y_att_after_bigram;
		if(bBigram) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				iy_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityFormer_bigram.ComputeForwardScore(input_entityFormer_bigram, iy_entityFormer_bigram
					, oy_entityFormer_bigram, fy_entityFormer_bigram, mcy_entityFormer_bigram,
					cy_entityFormer_bigram, my_entityFormer_bigram, y_entityFormer_bigram);
			for (int idx = 0; idx < enLatterSize; idx++) {
				iy_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityLatter_bigram.ComputeForwardScore(input_entityLatter_bigram, iy_entityLatter_bigram,
					oy_entityLatter_bigram,
					fy_entityLatter_bigram, mcy_entityLatter_bigram,cy_entityLatter_bigram,
					my_entityLatter_bigram, y_entityLatter_bigram);
			for (int idx = 0; idx < beforeSize; idx++) {
				iy_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_before_bigram.ComputeForwardScore(input_before_bigram, iy_before_bigram, oy_before_bigram,
					fy_before_bigram, mcy_before_bigram,cy_before_bigram, my_before_bigram, y_before_bigram);
			for (int idx = 0; idx < middleSize; idx++) {
				iy_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_middle_bigram.ComputeForwardScore(input_middle_bigram, iy_middle_bigram, oy_middle_bigram,
					fy_middle_bigram, mcy_middle_bigram,cy_middle_bigram, my_middle_bigram, y_middle_bigram);
			for (int idx = 0; idx < afterSize; idx++) {
				iy_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_after_bigram.ComputeForwardScore(input_after_bigram, iy_after_bigram, oy_after_bigram,
					fy_after_bigram, mcy_after_bigram,cy_after_bigram, my_after_bigram, y_after_bigram);

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					xMExp_before_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_before_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_before_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_before_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_before_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_before_bigram.ComputeForwardScore(y_before_bigram, y_before_bigram,
						xMExp_before_bigram, xExp_before_bigram, xSum_before_bigram,
						xPoolIndex_before_bigram, y_att_before_bigram);

				for (int idx = 0; idx < middleSize; idx++) {
					xMExp_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_middle_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_middle_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_middle_bigram.ComputeForwardScore(y_middle_bigram, y_middle_bigram,
						xMExp_middle_bigram, xExp_middle_bigram, xSum_middle_bigram,
						xPoolIndex_middle_bigram, y_att_middle_bigram);

				for (int idx = 0; idx < afterSize; idx++) {
					xMExp_after_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_after_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_after_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_after_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_after_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_after_bigram.ComputeForwardScore(y_after_bigram, y_after_bigram,
						xMExp_after_bigram, xExp_after_bigram, xSum_after_bigram,
						xPoolIndex_after_bigram, y_att_after_bigram);
			}
		}

		// pos channel
		vector<Tensor<xpu, 2, dtype> > iy_entityFormer_pos(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityFormer_pos(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityFormer_pos(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityFormer_pos(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityFormer_pos(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > my_entityFormer_pos(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > y_entityFormer_pos(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > iy_entityLatter_pos(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityLatter_pos(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityLatter_pos(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityLatter_pos(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityLatter_pos(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > my_entityLatter_pos(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > y_entityLatter_pos(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > iy_before_pos(beforeSize);
		vector<Tensor<xpu, 2, dtype> > oy_before_pos(beforeSize);
		vector<Tensor<xpu, 2, dtype> > fy_before_pos(beforeSize);
		vector<Tensor<xpu, 2, dtype> > mcy_before_pos(beforeSize);
		vector<Tensor<xpu, 2, dtype> > cy_before_pos(beforeSize);
		vector<Tensor<xpu, 2, dtype> > my_before_pos(beforeSize);
		vector<Tensor<xpu, 2, dtype> > y_before_pos(beforeSize);
		vector<Tensor<xpu, 2, dtype> > iy_middle_pos(middleSize);
		vector<Tensor<xpu, 2, dtype> > oy_middle_pos(middleSize);
		vector<Tensor<xpu, 2, dtype> > fy_middle_pos(middleSize);
		vector<Tensor<xpu, 2, dtype> > mcy_middle_pos(middleSize);
		vector<Tensor<xpu, 2, dtype> > cy_middle_pos(middleSize);
		vector<Tensor<xpu, 2, dtype> > my_middle_pos(middleSize);
		vector<Tensor<xpu, 2, dtype> > y_middle_pos(middleSize);
		vector<Tensor<xpu, 2, dtype> > iy_after_pos(afterSize);
		vector<Tensor<xpu, 2, dtype> > oy_after_pos(afterSize);
		vector<Tensor<xpu, 2, dtype> > fy_after_pos(afterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_after_pos(afterSize);
		vector<Tensor<xpu, 2, dtype> > cy_after_pos(afterSize);
		vector<Tensor<xpu, 2, dtype> > my_after_pos(afterSize);
		vector<Tensor<xpu, 2, dtype> > y_after_pos(afterSize);

		vector<Tensor<xpu, 2, dtype> > xMExp_before_pos(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xExp_before_pos(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_before_pos(beforeSize);
		Tensor<xpu, 2, dtype> xSum_before_pos;
		Tensor<xpu, 2, dtype> y_att_before_pos;
		vector<Tensor<xpu, 2, dtype> > xMExp_middle_pos(middleSize);
		vector<Tensor<xpu, 2, dtype> > xExp_middle_pos(middleSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle_pos(middleSize);
		Tensor<xpu, 2, dtype> xSum_middle_pos;
		Tensor<xpu, 2, dtype> y_att_middle_pos;
		vector<Tensor<xpu, 2, dtype> > xMExp_after_pos(afterSize);
		vector<Tensor<xpu, 2, dtype> > xExp_after_pos(afterSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_after_pos(afterSize);
		Tensor<xpu, 2, dtype> xSum_after_pos;
		Tensor<xpu, 2, dtype> y_att_after_pos;
		if(bPos) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				iy_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityFormer_pos.ComputeForwardScore(input_entityFormer_pos, iy_entityFormer_pos
					, oy_entityFormer_pos, fy_entityFormer_pos, mcy_entityFormer_pos,
					cy_entityFormer_pos, my_entityFormer_pos, y_entityFormer_pos);
			for (int idx = 0; idx < enLatterSize; idx++) {
				iy_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityLatter_pos.ComputeForwardScore(input_entityLatter_pos, iy_entityLatter_pos,
					oy_entityLatter_pos,
					fy_entityLatter_pos, mcy_entityLatter_pos,cy_entityLatter_pos,
					my_entityLatter_pos, y_entityLatter_pos);
			for (int idx = 0; idx < beforeSize; idx++) {
				iy_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_before_pos.ComputeForwardScore(input_before_pos, iy_before_pos, oy_before_pos,
					fy_before_pos, mcy_before_pos,cy_before_pos, my_before_pos, y_before_pos);
			for (int idx = 0; idx < middleSize; idx++) {
				iy_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_middle_pos.ComputeForwardScore(input_middle_pos, iy_middle_pos, oy_middle_pos,
					fy_middle_pos, mcy_middle_pos,cy_middle_pos, my_middle_pos, y_middle_pos);
			for (int idx = 0; idx < afterSize; idx++) {
				iy_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_after_pos.ComputeForwardScore(input_after_pos, iy_after_pos, oy_after_pos,
					fy_after_pos, mcy_after_pos,cy_after_pos, my_after_pos, y_after_pos);

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					xMExp_before_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_before_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_before_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_before_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_before_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_before_pos.ComputeForwardScore(y_before_pos, y_before_pos,
						xMExp_before_pos, xExp_before_pos, xSum_before_pos,
						xPoolIndex_before_pos, y_att_before_pos);

				for (int idx = 0; idx < middleSize; idx++) {
					xMExp_middle_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_middle_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_middle_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_middle_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_middle_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_middle_pos.ComputeForwardScore(y_middle_pos, y_middle_pos,
						xMExp_middle_pos, xExp_middle_pos, xSum_middle_pos,
						xPoolIndex_middle_pos, y_att_middle_pos);

				for (int idx = 0; idx < afterSize; idx++) {
					xMExp_after_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_after_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_after_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_after_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_after_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_after_pos.ComputeForwardScore(y_after_pos, y_after_pos,
						xMExp_after_pos, xExp_after_pos, xSum_after_pos,
						xPoolIndex_after_pos, y_att_after_pos);
			}
		}

		// sst channel
		vector<Tensor<xpu, 2, dtype> > iy_entityFormer_sst(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityFormer_sst(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityFormer_sst(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityFormer_sst(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityFormer_sst(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > my_entityFormer_sst(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > y_entityFormer_sst(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > iy_entityLatter_sst(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityLatter_sst(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityLatter_sst(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityLatter_sst(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityLatter_sst(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > my_entityLatter_sst(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > y_entityLatter_sst(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > iy_before_sst(beforeSize);
		vector<Tensor<xpu, 2, dtype> > oy_before_sst(beforeSize);
		vector<Tensor<xpu, 2, dtype> > fy_before_sst(beforeSize);
		vector<Tensor<xpu, 2, dtype> > mcy_before_sst(beforeSize);
		vector<Tensor<xpu, 2, dtype> > cy_before_sst(beforeSize);
		vector<Tensor<xpu, 2, dtype> > my_before_sst(beforeSize);
		vector<Tensor<xpu, 2, dtype> > y_before_sst(beforeSize);
		vector<Tensor<xpu, 2, dtype> > iy_middle_sst(middleSize);
		vector<Tensor<xpu, 2, dtype> > oy_middle_sst(middleSize);
		vector<Tensor<xpu, 2, dtype> > fy_middle_sst(middleSize);
		vector<Tensor<xpu, 2, dtype> > mcy_middle_sst(middleSize);
		vector<Tensor<xpu, 2, dtype> > cy_middle_sst(middleSize);
		vector<Tensor<xpu, 2, dtype> > my_middle_sst(middleSize);
		vector<Tensor<xpu, 2, dtype> > y_middle_sst(middleSize);
		vector<Tensor<xpu, 2, dtype> > iy_after_sst(afterSize);
		vector<Tensor<xpu, 2, dtype> > oy_after_sst(afterSize);
		vector<Tensor<xpu, 2, dtype> > fy_after_sst(afterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_after_sst(afterSize);
		vector<Tensor<xpu, 2, dtype> > cy_after_sst(afterSize);
		vector<Tensor<xpu, 2, dtype> > my_after_sst(afterSize);
		vector<Tensor<xpu, 2, dtype> > y_after_sst(afterSize);

		vector<Tensor<xpu, 2, dtype> > xMExp_before_sst(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xExp_before_sst(beforeSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_before_sst(beforeSize);
		Tensor<xpu, 2, dtype> xSum_before_sst;
		Tensor<xpu, 2, dtype> y_att_before_sst;
		vector<Tensor<xpu, 2, dtype> > xMExp_middle_sst(middleSize);
		vector<Tensor<xpu, 2, dtype> > xExp_middle_sst(middleSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle_sst(middleSize);
		Tensor<xpu, 2, dtype> xSum_middle_sst;
		Tensor<xpu, 2, dtype> y_att_middle_sst;
		vector<Tensor<xpu, 2, dtype> > xMExp_after_sst(afterSize);
		vector<Tensor<xpu, 2, dtype> > xExp_after_sst(afterSize);
		vector<Tensor<xpu, 2, dtype> > xPoolIndex_after_sst(afterSize);
		Tensor<xpu, 2, dtype> xSum_after_sst;
		Tensor<xpu, 2, dtype> y_att_after_sst;
		if(bSst) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				iy_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityFormer_sst.ComputeForwardScore(input_entityFormer_sst, iy_entityFormer_sst
					, oy_entityFormer_sst, fy_entityFormer_sst, mcy_entityFormer_sst,
					cy_entityFormer_sst, my_entityFormer_sst, y_entityFormer_sst);
			for (int idx = 0; idx < enLatterSize; idx++) {
				iy_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityLatter_sst.ComputeForwardScore(input_entityLatter_sst, iy_entityLatter_sst,
					oy_entityLatter_sst,
					fy_entityLatter_sst, mcy_entityLatter_sst,cy_entityLatter_sst,
					my_entityLatter_sst, y_entityLatter_sst);
			for (int idx = 0; idx < beforeSize; idx++) {
				iy_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_before_sst.ComputeForwardScore(input_before_sst, iy_before_sst, oy_before_sst,
					fy_before_sst, mcy_before_sst,cy_before_sst, my_before_sst, y_before_sst);
			for (int idx = 0; idx < middleSize; idx++) {
				iy_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_middle_sst.ComputeForwardScore(input_middle_sst, iy_middle_sst, oy_middle_sst,
					fy_middle_sst, mcy_middle_sst,cy_middle_sst, my_middle_sst, y_middle_sst);
			for (int idx = 0; idx < afterSize; idx++) {
				iy_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			unit_after_sst.ComputeForwardScore(input_after_sst, iy_after_sst, oy_after_sst,
					fy_after_sst, mcy_after_sst,cy_after_sst, my_after_sst, y_after_sst);

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					xMExp_before_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_before_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_before_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_before_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_before_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_before_sst.ComputeForwardScore(y_before_sst, y_before_sst,
						xMExp_before_sst, xExp_before_sst, xSum_before_sst,
						xPoolIndex_before_sst, y_att_before_sst);

				for (int idx = 0; idx < middleSize; idx++) {
					xMExp_middle_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_middle_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_middle_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_middle_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_middle_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_middle_sst.ComputeForwardScore(y_middle_sst, y_middle_sst,
						xMExp_middle_sst, xExp_middle_sst, xSum_middle_sst,
						xPoolIndex_middle_sst, y_att_middle_sst);

				for (int idx = 0; idx < afterSize; idx++) {
					xMExp_after_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_after_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_after_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				xSum_after_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				y_att_after_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				unit_att_after_sst.ComputeForwardScore(y_after_sst, y_after_sst,
						xMExp_after_sst, xExp_after_sst, xSum_after_sst,
						xPoolIndex_after_sst, y_att_after_sst);
			}
		}

		vector<Tensor<xpu, 2, dtype> > v_hidden_input;

		// word channel
		if(bWord) {

			if(options.attention) {
				v_hidden_input.push_back(y_att_before);
			} else
				v_hidden_input.push_back(y_before[beforeSize-1]);

			v_hidden_input.push_back(y_entityFormer[enFormerSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_middle);
			} else
				v_hidden_input.push_back(y_middle[middleSize-1]);

			v_hidden_input.push_back(y_entityLatter[enLatterSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_after);
			} else
				v_hidden_input.push_back(y_after[afterSize-1]);
		}

		// wordnet channel
		if(bWordnet) {

			if(options.attention) {
				v_hidden_input.push_back(y_att_before_wordnet);
			} else
				v_hidden_input.push_back(y_before_wordnet[beforeSize-1]);

			v_hidden_input.push_back(y_entityFormer_wordnet[enFormerSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_middle_wordnet);
			} else
				v_hidden_input.push_back(y_middle_wordnet[middleSize-1]);

			v_hidden_input.push_back(y_entityLatter_wordnet[enLatterSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_after_wordnet);
			} else
				v_hidden_input.push_back(y_after_wordnet[afterSize-1]);
		}

		// brown channel
		if(bBrown) {

			if(options.attention) {
				v_hidden_input.push_back(y_att_before_brown);
			} else
				v_hidden_input.push_back(y_before_brown[beforeSize-1]);

			v_hidden_input.push_back(y_entityFormer_brown[enFormerSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_middle_brown);
			} else
				v_hidden_input.push_back(y_middle_brown[middleSize-1]);

			v_hidden_input.push_back(y_entityLatter_brown[enLatterSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_after_brown);
			} else
				v_hidden_input.push_back(y_after_brown[afterSize-1]);
		}

		// bigram channel
		if(bBigram) {

			if(options.attention) {
				v_hidden_input.push_back(y_att_before_bigram);
			} else
				v_hidden_input.push_back(y_before_bigram[beforeSize-1]);

			v_hidden_input.push_back(y_entityFormer_bigram[enFormerSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_middle_bigram);
			} else
				v_hidden_input.push_back(y_middle_bigram[middleSize-1]);

			v_hidden_input.push_back(y_entityLatter_bigram[enLatterSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_after_bigram);
			} else
				v_hidden_input.push_back(y_after_bigram[afterSize-1]);
		}

		// pos channel
		if(bPos) {

			if(options.attention) {
				v_hidden_input.push_back(y_att_before_pos);
			} else
				v_hidden_input.push_back(y_before_pos[beforeSize-1]);

			v_hidden_input.push_back(y_entityFormer_pos[enFormerSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_middle_pos);
			} else
				v_hidden_input.push_back(y_middle_pos[middleSize-1]);

			v_hidden_input.push_back(y_entityLatter_pos[enLatterSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_after_pos);
			} else
				v_hidden_input.push_back(y_after_pos[afterSize-1]);
		}

		// sst channel
		if(bSst) {

			if(options.attention) {
				v_hidden_input.push_back(y_att_before_sst);
			} else
				v_hidden_input.push_back(y_before_sst[beforeSize-1]);

			v_hidden_input.push_back(y_entityFormer_sst[enFormerSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_middle_sst);
			} else
				v_hidden_input.push_back(y_middle_sst[middleSize-1]);

			v_hidden_input.push_back(y_entityLatter_sst[enLatterSize-1]);

			if(options.attention) {
				v_hidden_input.push_back(y_att_after_sst);
			} else
				v_hidden_input.push_back(y_after_sst[afterSize-1]);
		}


		Tensor<xpu, 2, dtype> hidden_input = NewTensor<xpu>(Shape2(1, _hidden_input_size), d_zero);
		concat(v_hidden_input, hidden_input);

		hidden_layer.ComputeForwardScore(hidden_input, hidden);

		output_layer.ComputeForwardScore(hidden, output);

		for(int i=0;i<scores.size();i++) {
			scores[i] = output[0][i];
		}


		FreeSpace(&hidden_input);
		FreeSpace(&hidden);

		FreeSpace(&output);

		// word channel
		if(bWord) {
			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(input_before[idx]));
				FreeSpace(&(iy_before[idx]));
				FreeSpace(&(oy_before[idx]));
				FreeSpace(&(fy_before[idx]));
				FreeSpace(&(mcy_before[idx]));
				FreeSpace(&(cy_before[idx]));
				FreeSpace(&(my_before[idx]));
				FreeSpace(&(y_before[idx]));
			}
			for (int idx = 0; idx < enFormerSize; idx++) {
				FreeSpace(&(input_entityFormer[idx]));
				FreeSpace(&(iy_entityFormer[idx]));
				FreeSpace(&(oy_entityFormer[idx]));
				FreeSpace(&(fy_entityFormer[idx]));
				FreeSpace(&(mcy_entityFormer[idx]));
				FreeSpace(&(cy_entityFormer[idx]));
				FreeSpace(&(my_entityFormer[idx]));
				FreeSpace(&(y_entityFormer[idx]));
			}
			for (int idx = 0; idx < enLatterSize; idx++) {
				FreeSpace(&(input_entityLatter[idx]));
				FreeSpace(&(iy_entityLatter[idx]));
				FreeSpace(&(oy_entityLatter[idx]));
				FreeSpace(&(fy_entityLatter[idx]));
				FreeSpace(&(mcy_entityLatter[idx]));
				FreeSpace(&(cy_entityLatter[idx]));
				FreeSpace(&(my_entityLatter[idx]));
				FreeSpace(&(y_entityLatter[idx]));
			}
			for (int idx = 0; idx < middleSize; idx++) {
				FreeSpace(&(input_middle[idx]));
				FreeSpace(&(iy_middle[idx]));
				FreeSpace(&(oy_middle[idx]));
				FreeSpace(&(fy_middle[idx]));
				FreeSpace(&(mcy_middle[idx]));
				FreeSpace(&(cy_middle[idx]));
				FreeSpace(&(my_middle[idx]));
				FreeSpace(&(y_middle[idx]));
			}
			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(input_after[idx]));
				FreeSpace(&(iy_after[idx]));
				FreeSpace(&(oy_after[idx]));
				FreeSpace(&(fy_after[idx]));
				FreeSpace(&(mcy_after[idx]));
				FreeSpace(&(cy_after[idx]));
				FreeSpace(&(my_after[idx]));
				FreeSpace(&(y_after[idx]));
			}
			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(xMExp_before[idx]));
					FreeSpace(&(xExp_before[idx]));
					FreeSpace(&(xPoolIndex_before[idx]));
				}
				FreeSpace(&(xSum_before));
				FreeSpace(&(y_att_before));

				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(xMExp_middle[idx]));
					FreeSpace(&(xExp_middle[idx]));
					FreeSpace(&(xPoolIndex_middle[idx]));
				}
				FreeSpace(&(xSum_middle));
				FreeSpace(&(y_att_middle));

				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(xMExp_after[idx]));
					FreeSpace(&(xExp_after[idx]));
					FreeSpace(&(xPoolIndex_after[idx]));
				}
				FreeSpace(&(xSum_after));
				FreeSpace(&(y_att_after));
			}
		}

		// wordnet channel
		if(bWordnet) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				FreeSpace(&(input_entityFormer_wordnet[idx]));
				FreeSpace(&(iy_entityFormer_wordnet[idx]));
				FreeSpace(&(oy_entityFormer_wordnet[idx]));
				FreeSpace(&(fy_entityFormer_wordnet[idx]));
				FreeSpace(&(mcy_entityFormer_wordnet[idx]));
				FreeSpace(&(cy_entityFormer_wordnet[idx]));
				FreeSpace(&(my_entityFormer_wordnet[idx]));
				FreeSpace(&(y_entityFormer_wordnet[idx]));
			}

			for (int idx = 0; idx < enLatterSize; idx++) {
				FreeSpace(&(input_entityLatter_wordnet[idx]));
				FreeSpace(&(iy_entityLatter_wordnet[idx]));
				FreeSpace(&(oy_entityLatter_wordnet[idx]));
				FreeSpace(&(fy_entityLatter_wordnet[idx]));
				FreeSpace(&(mcy_entityLatter_wordnet[idx]));
				FreeSpace(&(cy_entityLatter_wordnet[idx]));
				FreeSpace(&(my_entityLatter_wordnet[idx]));
				FreeSpace(&(y_entityLatter_wordnet[idx]));
			}

			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(input_before_wordnet[idx]));
				FreeSpace(&(iy_before_wordnet[idx]));
				FreeSpace(&(oy_before_wordnet[idx]));
				FreeSpace(&(fy_before_wordnet[idx]));
				FreeSpace(&(mcy_before_wordnet[idx]));
				FreeSpace(&(cy_before_wordnet[idx]));
				FreeSpace(&(my_before_wordnet[idx]));
				FreeSpace(&(y_before_wordnet[idx]));
			}
			for (int idx = 0; idx < middleSize; idx++) {
				FreeSpace(&(input_middle_wordnet[idx]));
				FreeSpace(&(iy_middle_wordnet[idx]));
				FreeSpace(&(oy_middle_wordnet[idx]));
				FreeSpace(&(fy_middle_wordnet[idx]));
				FreeSpace(&(mcy_middle_wordnet[idx]));
				FreeSpace(&(cy_middle_wordnet[idx]));
				FreeSpace(&(my_middle_wordnet[idx]));
				FreeSpace(&(y_middle_wordnet[idx]));
			}
			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(input_after_wordnet[idx]));
				FreeSpace(&(iy_after_wordnet[idx]));
				FreeSpace(&(oy_after_wordnet[idx]));
				FreeSpace(&(fy_after_wordnet[idx]));
				FreeSpace(&(mcy_after_wordnet[idx]));
				FreeSpace(&(cy_after_wordnet[idx]));
				FreeSpace(&(my_after_wordnet[idx]));
				FreeSpace(&(y_after_wordnet[idx]));
			}

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(xMExp_before_wordnet[idx]));
					FreeSpace(&(xExp_before_wordnet[idx]));
					FreeSpace(&(xPoolIndex_before_wordnet[idx]));
				}
				FreeSpace(&(xSum_before_wordnet));
				FreeSpace(&(y_att_before_wordnet));

				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(xMExp_middle_wordnet[idx]));
					FreeSpace(&(xExp_middle_wordnet[idx]));
					FreeSpace(&(xPoolIndex_middle_wordnet[idx]));
				}
				FreeSpace(&(xSum_middle_wordnet));
				FreeSpace(&(y_att_middle_wordnet));

				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(xMExp_after_wordnet[idx]));
					FreeSpace(&(xExp_after_wordnet[idx]));
					FreeSpace(&(xPoolIndex_after_wordnet[idx]));
				}
				FreeSpace(&(xSum_after_wordnet));
				FreeSpace(&(y_att_after_wordnet));
			}
		}

		// brown channel
		if(bBrown) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				FreeSpace(&(input_entityFormer_brown[idx]));
				FreeSpace(&(iy_entityFormer_brown[idx]));
				FreeSpace(&(oy_entityFormer_brown[idx]));
				FreeSpace(&(fy_entityFormer_brown[idx]));
				FreeSpace(&(mcy_entityFormer_brown[idx]));
				FreeSpace(&(cy_entityFormer_brown[idx]));
				FreeSpace(&(my_entityFormer_brown[idx]));
				FreeSpace(&(y_entityFormer_brown[idx]));
			}

			for (int idx = 0; idx < enLatterSize; idx++) {
				FreeSpace(&(input_entityLatter_brown[idx]));
				FreeSpace(&(iy_entityLatter_brown[idx]));
				FreeSpace(&(oy_entityLatter_brown[idx]));
				FreeSpace(&(fy_entityLatter_brown[idx]));
				FreeSpace(&(mcy_entityLatter_brown[idx]));
				FreeSpace(&(cy_entityLatter_brown[idx]));
				FreeSpace(&(my_entityLatter_brown[idx]));
				FreeSpace(&(y_entityLatter_brown[idx]));
			}

			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(input_before_brown[idx]));
				FreeSpace(&(iy_before_brown[idx]));
				FreeSpace(&(oy_before_brown[idx]));
				FreeSpace(&(fy_before_brown[idx]));
				FreeSpace(&(mcy_before_brown[idx]));
				FreeSpace(&(cy_before_brown[idx]));
				FreeSpace(&(my_before_brown[idx]));
				FreeSpace(&(y_before_brown[idx]));
			}
			for (int idx = 0; idx < middleSize; idx++) {
				FreeSpace(&(input_middle_brown[idx]));
				FreeSpace(&(iy_middle_brown[idx]));
				FreeSpace(&(oy_middle_brown[idx]));
				FreeSpace(&(fy_middle_brown[idx]));
				FreeSpace(&(mcy_middle_brown[idx]));
				FreeSpace(&(cy_middle_brown[idx]));
				FreeSpace(&(my_middle_brown[idx]));
				FreeSpace(&(y_middle_brown[idx]));
			}
			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(input_after_brown[idx]));
				FreeSpace(&(iy_after_brown[idx]));
				FreeSpace(&(oy_after_brown[idx]));
				FreeSpace(&(fy_after_brown[idx]));
				FreeSpace(&(mcy_after_brown[idx]));
				FreeSpace(&(cy_after_brown[idx]));
				FreeSpace(&(my_after_brown[idx]));
				FreeSpace(&(y_after_brown[idx]));
			}

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(xMExp_before_brown[idx]));
					FreeSpace(&(xExp_before_brown[idx]));
					FreeSpace(&(xPoolIndex_before_brown[idx]));
				}
				FreeSpace(&(xSum_before_brown));
				FreeSpace(&(y_att_before_brown));

				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(xMExp_middle_brown[idx]));
					FreeSpace(&(xExp_middle_brown[idx]));
					FreeSpace(&(xPoolIndex_middle_brown[idx]));
				}
				FreeSpace(&(xSum_middle_brown));
				FreeSpace(&(y_att_middle_brown));

				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(xMExp_after_brown[idx]));
					FreeSpace(&(xExp_after_brown[idx]));
					FreeSpace(&(xPoolIndex_after_brown[idx]));
				}
				FreeSpace(&(xSum_after_brown));
				FreeSpace(&(y_att_after_brown));
			}
		}

		// bigram channel
		if(bBigram) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				FreeSpace(&(input_entityFormer_bigram[idx]));
				FreeSpace(&(iy_entityFormer_bigram[idx]));
				FreeSpace(&(oy_entityFormer_bigram[idx]));
				FreeSpace(&(fy_entityFormer_bigram[idx]));
				FreeSpace(&(mcy_entityFormer_bigram[idx]));
				FreeSpace(&(cy_entityFormer_bigram[idx]));
				FreeSpace(&(my_entityFormer_bigram[idx]));
				FreeSpace(&(y_entityFormer_bigram[idx]));
			}
			for (int idx = 0; idx < enLatterSize; idx++) {
				FreeSpace(&(input_entityLatter_bigram[idx]));
				FreeSpace(&(iy_entityLatter_bigram[idx]));
				FreeSpace(&(oy_entityLatter_bigram[idx]));
				FreeSpace(&(fy_entityLatter_bigram[idx]));
				FreeSpace(&(mcy_entityLatter_bigram[idx]));
				FreeSpace(&(cy_entityLatter_bigram[idx]));
				FreeSpace(&(my_entityLatter_bigram[idx]));
				FreeSpace(&(y_entityLatter_bigram[idx]));
			}

			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(input_before_bigram[idx]));
				FreeSpace(&(iy_before_bigram[idx]));
				FreeSpace(&(oy_before_bigram[idx]));
				FreeSpace(&(fy_before_bigram[idx]));
				FreeSpace(&(mcy_before_bigram[idx]));
				FreeSpace(&(cy_before_bigram[idx]));
				FreeSpace(&(my_before_bigram[idx]));
				FreeSpace(&(y_before_bigram[idx]));
			}
			for (int idx = 0; idx < middleSize; idx++) {
				FreeSpace(&(input_middle_bigram[idx]));
				FreeSpace(&(iy_middle_bigram[idx]));
				FreeSpace(&(oy_middle_bigram[idx]));
				FreeSpace(&(fy_middle_bigram[idx]));
				FreeSpace(&(mcy_middle_bigram[idx]));
				FreeSpace(&(cy_middle_bigram[idx]));
				FreeSpace(&(my_middle_bigram[idx]));
				FreeSpace(&(y_middle_bigram[idx]));
			}
			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(input_after_bigram[idx]));
				FreeSpace(&(iy_after_bigram[idx]));
				FreeSpace(&(oy_after_bigram[idx]));
				FreeSpace(&(fy_after_bigram[idx]));
				FreeSpace(&(mcy_after_bigram[idx]));
				FreeSpace(&(cy_after_bigram[idx]));
				FreeSpace(&(my_after_bigram[idx]));
				FreeSpace(&(y_after_bigram[idx]));
			}

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(xMExp_before_bigram[idx]));
					FreeSpace(&(xExp_before_bigram[idx]));
					FreeSpace(&(xPoolIndex_before_bigram[idx]));
				}
				FreeSpace(&(xSum_before_bigram));
				FreeSpace(&(y_att_before_bigram));

				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(xMExp_middle_bigram[idx]));
					FreeSpace(&(xExp_middle_bigram[idx]));
					FreeSpace(&(xPoolIndex_middle_bigram[idx]));
				}
				FreeSpace(&(xSum_middle_bigram));
				FreeSpace(&(y_att_middle_bigram));

				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(xMExp_after_bigram[idx]));
					FreeSpace(&(xExp_after_bigram[idx]));
					FreeSpace(&(xPoolIndex_after_bigram[idx]));
				}
				FreeSpace(&(xSum_after_bigram));
				FreeSpace(&(y_att_after_bigram));
			}
		}

		// pos channel
		if(bPos) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				FreeSpace(&(input_entityFormer_pos[idx]));
				FreeSpace(&(iy_entityFormer_pos[idx]));
				FreeSpace(&(oy_entityFormer_pos[idx]));
				FreeSpace(&(fy_entityFormer_pos[idx]));
				FreeSpace(&(mcy_entityFormer_pos[idx]));
				FreeSpace(&(cy_entityFormer_pos[idx]));
				FreeSpace(&(my_entityFormer_pos[idx]));
				FreeSpace(&(y_entityFormer_pos[idx]));
			}

			for (int idx = 0; idx < enLatterSize; idx++) {
				FreeSpace(&(input_entityLatter_pos[idx]));
				FreeSpace(&(iy_entityLatter_pos[idx]));
				FreeSpace(&(oy_entityLatter_pos[idx]));
				FreeSpace(&(fy_entityLatter_pos[idx]));
				FreeSpace(&(mcy_entityLatter_pos[idx]));
				FreeSpace(&(cy_entityLatter_pos[idx]));
				FreeSpace(&(my_entityLatter_pos[idx]));
				FreeSpace(&(y_entityLatter_pos[idx]));
			}

			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(input_before_pos[idx]));
				FreeSpace(&(iy_before_pos[idx]));
				FreeSpace(&(oy_before_pos[idx]));
				FreeSpace(&(fy_before_pos[idx]));
				FreeSpace(&(mcy_before_pos[idx]));
				FreeSpace(&(cy_before_pos[idx]));
				FreeSpace(&(my_before_pos[idx]));
				FreeSpace(&(y_before_pos[idx]));
			}
			for (int idx = 0; idx < middleSize; idx++) {
				FreeSpace(&(input_middle_pos[idx]));
				FreeSpace(&(iy_middle_pos[idx]));
				FreeSpace(&(oy_middle_pos[idx]));
				FreeSpace(&(fy_middle_pos[idx]));
				FreeSpace(&(mcy_middle_pos[idx]));
				FreeSpace(&(cy_middle_pos[idx]));
				FreeSpace(&(my_middle_pos[idx]));
				FreeSpace(&(y_middle_pos[idx]));
			}
			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(input_after_pos[idx]));
				FreeSpace(&(iy_after_pos[idx]));
				FreeSpace(&(oy_after_pos[idx]));
				FreeSpace(&(fy_after_pos[idx]));
				FreeSpace(&(mcy_after_pos[idx]));
				FreeSpace(&(cy_after_pos[idx]));
				FreeSpace(&(my_after_pos[idx]));
				FreeSpace(&(y_after_pos[idx]));
			}

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(xMExp_before_pos[idx]));
					FreeSpace(&(xExp_before_pos[idx]));
					FreeSpace(&(xPoolIndex_before_pos[idx]));
				}
				FreeSpace(&(xSum_before_pos));
				FreeSpace(&(y_att_before_pos));

				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(xMExp_middle_pos[idx]));
					FreeSpace(&(xExp_middle_pos[idx]));
					FreeSpace(&(xPoolIndex_middle_pos[idx]));
				}
				FreeSpace(&(xSum_middle_pos));
				FreeSpace(&(y_att_middle_pos));

				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(xMExp_after_pos[idx]));
					FreeSpace(&(xExp_after_pos[idx]));
					FreeSpace(&(xPoolIndex_after_pos[idx]));
				}
				FreeSpace(&(xSum_after_pos));
				FreeSpace(&(y_att_after_pos));
			}
		}

		// sst channel
		if(bSst) {
			for (int idx = 0; idx < enFormerSize; idx++) {
				FreeSpace(&(input_entityFormer_sst[idx]));
				FreeSpace(&(iy_entityFormer_sst[idx]));
				FreeSpace(&(oy_entityFormer_sst[idx]));
				FreeSpace(&(fy_entityFormer_sst[idx]));
				FreeSpace(&(mcy_entityFormer_sst[idx]));
				FreeSpace(&(cy_entityFormer_sst[idx]));
				FreeSpace(&(my_entityFormer_sst[idx]));
				FreeSpace(&(y_entityFormer_sst[idx]));
			}

			for (int idx = 0; idx < enLatterSize; idx++) {
				FreeSpace(&(input_entityLatter_sst[idx]));
				FreeSpace(&(iy_entityLatter_sst[idx]));
				FreeSpace(&(oy_entityLatter_sst[idx]));
				FreeSpace(&(fy_entityLatter_sst[idx]));
				FreeSpace(&(mcy_entityLatter_sst[idx]));
				FreeSpace(&(cy_entityLatter_sst[idx]));
				FreeSpace(&(my_entityLatter_sst[idx]));
				FreeSpace(&(y_entityLatter_sst[idx]));
			}

			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(input_before_sst[idx]));
				FreeSpace(&(iy_before_sst[idx]));
				FreeSpace(&(oy_before_sst[idx]));
				FreeSpace(&(fy_before_sst[idx]));
				FreeSpace(&(mcy_before_sst[idx]));
				FreeSpace(&(cy_before_sst[idx]));
				FreeSpace(&(my_before_sst[idx]));
				FreeSpace(&(y_before_sst[idx]));
			}
			for (int idx = 0; idx < middleSize; idx++) {
				FreeSpace(&(input_middle_sst[idx]));
				FreeSpace(&(iy_middle_sst[idx]));
				FreeSpace(&(oy_middle_sst[idx]));
				FreeSpace(&(fy_middle_sst[idx]));
				FreeSpace(&(mcy_middle_sst[idx]));
				FreeSpace(&(cy_middle_sst[idx]));
				FreeSpace(&(my_middle_sst[idx]));
				FreeSpace(&(y_middle_sst[idx]));
			}
			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(input_after_sst[idx]));
				FreeSpace(&(iy_after_sst[idx]));
				FreeSpace(&(oy_after_sst[idx]));
				FreeSpace(&(fy_after_sst[idx]));
				FreeSpace(&(mcy_after_sst[idx]));
				FreeSpace(&(cy_after_sst[idx]));
				FreeSpace(&(my_after_sst[idx]));
				FreeSpace(&(y_after_sst[idx]));
			}

			if(options.attention) {
				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(xMExp_before_sst[idx]));
					FreeSpace(&(xExp_before_sst[idx]));
					FreeSpace(&(xPoolIndex_before_sst[idx]));
				}
				FreeSpace(&(xSum_before_sst));
				FreeSpace(&(y_att_before_sst));

				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(xMExp_middle_sst[idx]));
					FreeSpace(&(xExp_middle_sst[idx]));
					FreeSpace(&(xPoolIndex_middle_sst[idx]));
				}
				FreeSpace(&(xSum_middle_sst));
				FreeSpace(&(y_att_middle_sst));

				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(xMExp_after_sst[idx]));
					FreeSpace(&(xExp_after_sst[idx]));
					FreeSpace(&(xPoolIndex_after_sst[idx]));
				}
				FreeSpace(&(xSum_after_sst));
				FreeSpace(&(y_att_after_sst));
			}
		}

	}

	dtype process(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;
		int offset = 0;
		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];
			int enFormerSize = example.m_entityFormer.size();
			int enLatterSize = example.m_entityLatter.size();
			int middleSize = example.m_middle.size();
			int beforeSize = example.m_before.size();
			int afterSize = example.m_after.size();

			// word channel
			vector<Tensor<xpu, 2, dtype> > input_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > input_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > input_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mask_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > input_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > mask_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > input_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > mask_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_after(afterSize);
			if(bWord) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					input_entityFormer[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityFormer[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityFormer[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_words.GetEmb(example.m_entityFormer[idx], input_entityFormer[idx]);
					dropoutcol(mask_entityFormer[idx], options.dropProb);
					input_entityFormer[idx] = input_entityFormer[idx] * mask_entityFormer[idx];
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					input_entityLatter[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityLatter[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityLatter[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_words.GetEmb(example.m_entityLatter[idx], input_entityLatter[idx]);
					dropoutcol(mask_entityLatter[idx], options.dropProb);
					input_entityLatter[idx] = input_entityLatter[idx] * mask_entityLatter[idx];
				}
				for (int idx = 0; idx < beforeSize; idx++) {
					input_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_words.GetEmb(example.m_before[idx], input_before[idx]);
					dropoutcol(mask_before[idx], options.dropProb);
					input_before[idx] = input_before[idx] * mask_before[idx];
				}
				for (int idx = 0; idx < middleSize; idx++) {
					input_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_words.GetEmb(example.m_middle[idx], input_middle[idx]);
					dropoutcol(mask_middle[idx], options.dropProb);
					input_middle[idx] = input_middle[idx] * mask_middle[idx];
				}
				for (int idx = 0; idx < afterSize; idx++) {
					input_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_words.GetEmb(example.m_after[idx], input_after[idx]);
					dropoutcol(mask_after[idx], options.dropProb);
					input_after[idx] = input_after[idx] * mask_after[idx];
				}
			}


			// wordnet channel
			vector<Tensor<xpu, 2, dtype> > input_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > input_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > input_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mask_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > input_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > mask_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > input_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > mask_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_after_wordnet(afterSize);
			if(bWordnet) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					input_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_wordnet.GetEmb(example.m_entityFormer_wordnet[idx], input_entityFormer_wordnet[idx]);
					dropoutcol(mask_entityFormer_wordnet[idx], options.dropProb);
					input_entityFormer_wordnet[idx] = input_entityFormer_wordnet[idx] * mask_entityFormer_wordnet[idx];
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					input_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_wordnet.GetEmb(example.m_entityLatter_wordnet[idx], input_entityLatter_wordnet[idx]);
					dropoutcol(mask_entityLatter_wordnet[idx], options.dropProb);
					input_entityLatter_wordnet[idx] = input_entityLatter_wordnet[idx] * mask_entityLatter_wordnet[idx];
				}
				for (int idx = 0; idx < beforeSize; idx++) {
					input_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_wordnet.GetEmb(example.m_before_wordnet[idx], input_before_wordnet[idx]);
					dropoutcol(mask_before_wordnet[idx], options.dropProb);
					input_before_wordnet[idx] = input_before_wordnet[idx] * mask_before_wordnet[idx];
				}
				for (int idx = 0; idx < middleSize; idx++) {
					input_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_wordnet.GetEmb(example.m_middle_wordnet[idx], input_middle_wordnet[idx]);
					dropoutcol(mask_middle_wordnet[idx], options.dropProb);
					input_middle_wordnet[idx] = input_middle_wordnet[idx] * mask_middle_wordnet[idx];
				}
				for (int idx = 0; idx < afterSize; idx++) {
					input_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_wordnet.GetEmb(example.m_after_wordnet[idx], input_after_wordnet[idx]);
					dropoutcol(mask_after_wordnet[idx], options.dropProb);
					input_after_wordnet[idx] = input_after_wordnet[idx] * mask_after_wordnet[idx];
				}
			}

			// brown channel
			vector<Tensor<xpu, 2, dtype> > input_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > input_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > input_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mask_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > input_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > mask_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > input_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > mask_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_after_brown(afterSize);
			if(bBrown) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					input_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_brown.GetEmb(example.m_entityFormer_brown[idx], input_entityFormer_brown[idx]);
					dropoutcol(mask_entityFormer_brown[idx], options.dropProb);
					input_entityFormer_brown[idx] = input_entityFormer_brown[idx] * mask_entityFormer_brown[idx];
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					input_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_brown.GetEmb(example.m_entityLatter_brown[idx], input_entityLatter_brown[idx]);
					dropoutcol(mask_entityLatter_brown[idx], options.dropProb);
					input_entityLatter_brown[idx] = input_entityLatter_brown[idx] * mask_entityLatter_brown[idx];
				}
				for (int idx = 0; idx < beforeSize; idx++) {
					input_before_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_before_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_before_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_brown.GetEmb(example.m_before_brown[idx], input_before_brown[idx]);
					dropoutcol(mask_before_brown[idx], options.dropProb);
					input_before_brown[idx] = input_before_brown[idx] * mask_before_brown[idx];
				}
				for (int idx = 0; idx < middleSize; idx++) {
					input_middle_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_middle_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_middle_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_brown.GetEmb(example.m_middle_brown[idx], input_middle_brown[idx]);
					dropoutcol(mask_middle_brown[idx], options.dropProb);
					input_middle_brown[idx] = input_middle_brown[idx] * mask_middle_brown[idx];
				}
				for (int idx = 0; idx < afterSize; idx++) {
					input_after_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_after_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_after_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_brown.GetEmb(example.m_after_brown[idx], input_after_brown[idx]);
					dropoutcol(mask_after_brown[idx], options.dropProb);
					input_after_brown[idx] = input_after_brown[idx] * mask_after_brown[idx];
				}
			}


			// bigram channel
			vector<Tensor<xpu, 2, dtype> > input_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > input_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > input_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mask_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > input_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > mask_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > input_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > mask_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_after_bigram(afterSize);
			if(bBigram) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					input_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_bigram.GetEmb(example.m_entityFormer_bigram[idx], input_entityFormer_bigram[idx]);
					dropoutcol(mask_entityFormer_bigram[idx], options.dropProb);
					input_entityFormer_bigram[idx] = input_entityFormer_bigram[idx] * mask_entityFormer_bigram[idx];
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					input_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_bigram.GetEmb(example.m_entityLatter_bigram[idx], input_entityLatter_bigram[idx]);
					dropoutcol(mask_entityLatter_bigram[idx], options.dropProb);
					input_entityLatter_bigram[idx] = input_entityLatter_bigram[idx] * mask_entityLatter_bigram[idx];
				}
				for (int idx = 0; idx < beforeSize; idx++) {
					input_before_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_before_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_before_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_bigram.GetEmb(example.m_before_bigram[idx], input_before_bigram[idx]);
					dropoutcol(mask_before_bigram[idx], options.dropProb);
					input_before_bigram[idx] = input_before_bigram[idx] * mask_before_bigram[idx];
				}
				for (int idx = 0; idx < middleSize; idx++) {
					input_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_bigram.GetEmb(example.m_middle_bigram[idx], input_middle_bigram[idx]);
					dropoutcol(mask_middle_bigram[idx], options.dropProb);
					input_middle_bigram[idx] = input_middle_bigram[idx] * mask_middle_bigram[idx];
				}
				for (int idx = 0; idx < afterSize; idx++) {
					input_after_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_after_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_after_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_bigram.GetEmb(example.m_after_bigram[idx], input_after_bigram[idx]);
					dropoutcol(mask_after_bigram[idx], options.dropProb);
					input_after_bigram[idx] = input_after_bigram[idx] * mask_after_bigram[idx];
				}
			}

			// pos channel
			vector<Tensor<xpu, 2, dtype> > input_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > input_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > input_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mask_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > input_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > mask_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > input_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > mask_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_after_pos(afterSize);
			if(bPos) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					input_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_pos.GetEmb(example.m_entityFormer_pos[idx], input_entityFormer_pos[idx]);
					dropoutcol(mask_entityFormer_pos[idx], options.dropProb);
					input_entityFormer_pos[idx] = input_entityFormer_pos[idx] * mask_entityFormer_pos[idx];
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					input_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_pos.GetEmb(example.m_entityLatter_pos[idx], input_entityLatter_pos[idx]);
					dropoutcol(mask_entityLatter_pos[idx], options.dropProb);
					input_entityLatter_pos[idx] = input_entityLatter_pos[idx] * mask_entityLatter_pos[idx];
				}
				for (int idx = 0; idx < beforeSize; idx++) {
					input_before_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_before_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_before_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_pos.GetEmb(example.m_before_pos[idx], input_before_pos[idx]);
					dropoutcol(mask_before_pos[idx], options.dropProb);
					input_before_pos[idx] = input_before_pos[idx] * mask_before_pos[idx];
				}
				for (int idx = 0; idx < middleSize; idx++) {
					input_middle_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_middle_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_middle_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_pos.GetEmb(example.m_middle_pos[idx], input_middle_pos[idx]);
					dropoutcol(mask_middle_pos[idx], options.dropProb);
					input_middle_pos[idx] = input_middle_pos[idx] * mask_middle_pos[idx];
				}
				for (int idx = 0; idx < afterSize; idx++) {
					input_after_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_after_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_after_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_pos.GetEmb(example.m_after_pos[idx], input_after_pos[idx]);
					dropoutcol(mask_after_pos[idx], options.dropProb);
					input_after_pos[idx] = input_after_pos[idx] * mask_after_pos[idx];
				}
			}


			// sst channel
			vector<Tensor<xpu, 2, dtype> > input_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > input_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > input_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mask_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > input_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > mask_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > input_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > mask_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_after_sst(afterSize);
			if(bSst) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					input_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_sst.GetEmb(example.m_entityFormer_sst[idx], input_entityFormer_sst[idx]);
					dropoutcol(mask_entityFormer_sst[idx], options.dropProb);
					input_entityFormer_sst[idx] = input_entityFormer_sst[idx] * mask_entityFormer_sst[idx];
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					input_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_sst.GetEmb(example.m_entityLatter_sst[idx], input_entityLatter_sst[idx]);
					dropoutcol(mask_entityLatter_sst[idx], options.dropProb);
					input_entityLatter_sst[idx] = input_entityLatter_sst[idx] * mask_entityLatter_sst[idx];
				}
				for (int idx = 0; idx < beforeSize; idx++) {
					input_before_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_before_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_before_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_sst.GetEmb(example.m_before_sst[idx], input_before_sst[idx]);
					dropoutcol(mask_before_sst[idx], options.dropProb);
					input_before_sst[idx] = input_before_sst[idx] * mask_before_sst[idx];
				}
				for (int idx = 0; idx < middleSize; idx++) {
					input_middle_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_middle_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_middle_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_sst.GetEmb(example.m_middle_sst[idx], input_middle_sst[idx]);
					dropoutcol(mask_middle_sst[idx], options.dropProb);
					input_middle_sst[idx] = input_middle_sst[idx] * mask_middle_sst[idx];
				}
				for (int idx = 0; idx < afterSize; idx++) {
					input_after_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_after_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_after_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_sst.GetEmb(example.m_after_sst[idx], input_after_sst[idx]);
					dropoutcol(mask_after_sst[idx], options.dropProb);
					input_after_sst[idx] = input_after_sst[idx] * mask_after_sst[idx];
				}
			}


			Tensor<xpu, 2, dtype> hidden = NewTensor<xpu>(Shape2(1, options.hiddenSize), d_zero);
			Tensor<xpu, 2, dtype> hiddenLoss = NewTensor<xpu>(Shape2(1, options.hiddenSize), d_zero);

			Tensor<xpu, 2, dtype> output = NewTensor<xpu>(Shape2(1, _outputSize), d_zero);
			Tensor<xpu, 2, dtype> outputLoss = NewTensor<xpu>(Shape2(1, _outputSize), d_zero);

			// word channel
			vector<Tensor<xpu, 2, dtype> > iy_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > my_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > y_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > iy_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > my_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > y_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > iy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > oy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > fy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mcy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > cy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > my_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > y_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > loss_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > iy_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > oy_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > fy_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > mcy_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > cy_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > my_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > y_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > loss_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > iy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > oy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > fy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > cy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > my_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > y_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > loss_after(afterSize);

			vector<Tensor<xpu, 2, dtype> > xMExp_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xExp_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_before(beforeSize);
			Tensor<xpu, 2, dtype> xSum_before;
			Tensor<xpu, 2, dtype> y_att_before;
			Tensor<xpu, 2, dtype> ly_att_before;
			vector<Tensor<xpu, 2, dtype> > xMExp_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > xExp_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle(middleSize);
			Tensor<xpu, 2, dtype> xSum_middle;
			Tensor<xpu, 2, dtype> y_att_middle;
			Tensor<xpu, 2, dtype> ly_att_middle;
			vector<Tensor<xpu, 2, dtype> > xMExp_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > xExp_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_after(afterSize);
			Tensor<xpu, 2, dtype> xSum_after;
			Tensor<xpu, 2, dtype> y_att_after;
			Tensor<xpu, 2, dtype> ly_att_after;
			if(bWord) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					iy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				}
				unit_entityFormer.ComputeForwardScore(input_entityFormer, iy_entityFormer, oy_entityFormer,
						fy_entityFormer, mcy_entityFormer,cy_entityFormer, my_entityFormer, y_entityFormer);
				for (int idx = 0; idx < enLatterSize; idx++) {
					iy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);;
				}
				unit_entityLatter.ComputeForwardScore(input_entityLatter, iy_entityLatter, oy_entityLatter,
						fy_entityLatter, mcy_entityLatter,cy_entityLatter, my_entityLatter, y_entityLatter);
				for (int idx = 0; idx < beforeSize; idx++) {
					iy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);;
				}
				unit_before.ComputeForwardScore(input_before, iy_before, oy_before,
						fy_before, mcy_before,cy_before, my_before, y_before);
				for (int idx = 0; idx < middleSize; idx++) {
					iy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);;
				}
				unit_middle.ComputeForwardScore(input_middle, iy_middle, oy_middle,
						fy_middle, mcy_middle,cy_middle, my_middle, y_middle);
				for (int idx = 0; idx < afterSize; idx++) {
					iy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);;
				}
				unit_after.ComputeForwardScore(input_after, iy_after, oy_after,
						fy_after, mcy_after,cy_after, my_after, y_after);

				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						xMExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_before.ComputeForwardScore(y_before, y_before,
							xMExp_before, xExp_before, xSum_before,
							xPoolIndex_before, y_att_before);

					for (int idx = 0; idx < middleSize; idx++) {
						xMExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_middle.ComputeForwardScore(y_middle, y_middle,
							xMExp_middle, xExp_middle, xSum_middle,
							xPoolIndex_middle, y_att_middle);

					for (int idx = 0; idx < afterSize; idx++) {
						xMExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_after.ComputeForwardScore(y_after, y_after,
							xMExp_after, xExp_after, xSum_after,
							xPoolIndex_after, y_att_after);
				}

			}

			// wordnet channel
			vector<Tensor<xpu, 2, dtype> > iy_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > my_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > y_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityFormer_wordnet(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > iy_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > my_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > y_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityLatter_wordnet(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > iy_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > oy_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > fy_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mcy_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > cy_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > my_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > y_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > loss_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > iy_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > oy_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > fy_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > mcy_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > cy_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > my_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > y_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > loss_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > iy_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > oy_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > fy_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > cy_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > my_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > y_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > loss_after_wordnet(afterSize);

			vector<Tensor<xpu, 2, dtype> > xMExp_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xExp_before_wordnet(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_before_wordnet(beforeSize);
			Tensor<xpu, 2, dtype> xSum_before_wordnet;
			Tensor<xpu, 2, dtype> y_att_before_wordnet;
			Tensor<xpu, 2, dtype> ly_att_before_wordnet;
			vector<Tensor<xpu, 2, dtype> > xMExp_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > xExp_middle_wordnet(middleSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle_wordnet(middleSize);
			Tensor<xpu, 2, dtype> xSum_middle_wordnet;
			Tensor<xpu, 2, dtype> y_att_middle_wordnet;
			Tensor<xpu, 2, dtype> ly_att_middle_wordnet;
			vector<Tensor<xpu, 2, dtype> > xMExp_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > xExp_after_wordnet(afterSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_after_wordnet(afterSize);
			Tensor<xpu, 2, dtype> xSum_after_wordnet;
			Tensor<xpu, 2, dtype> y_att_after_wordnet;
			Tensor<xpu, 2, dtype> ly_att_after_wordnet;
			if(bWordnet) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					iy_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityFormer_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				}
				unit_entityFormer_wordnet.ComputeForwardScore(input_entityFormer_wordnet, iy_entityFormer_wordnet
						, oy_entityFormer_wordnet, fy_entityFormer_wordnet, mcy_entityFormer_wordnet,
						cy_entityFormer_wordnet, my_entityFormer_wordnet, y_entityFormer_wordnet);
				for (int idx = 0; idx < enLatterSize; idx++) {
					iy_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityLatter_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);;
				}
				unit_entityLatter_wordnet.ComputeForwardScore(input_entityLatter_wordnet, iy_entityLatter_wordnet,
						oy_entityLatter_wordnet,
						fy_entityLatter_wordnet, mcy_entityLatter_wordnet,cy_entityLatter_wordnet,
						my_entityLatter_wordnet, y_entityLatter_wordnet);
				for (int idx = 0; idx < beforeSize; idx++) {
					iy_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_before_wordnet.ComputeForwardScore(input_before_wordnet, iy_before_wordnet, oy_before_wordnet,
						fy_before_wordnet, mcy_before_wordnet,cy_before_wordnet, my_before_wordnet, y_before_wordnet);
				for (int idx = 0; idx < middleSize; idx++) {
					iy_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_middle_wordnet.ComputeForwardScore(input_middle_wordnet, iy_middle_wordnet, oy_middle_wordnet,
						fy_middle_wordnet, mcy_middle_wordnet,cy_middle_wordnet, my_middle_wordnet, y_middle_wordnet);
				for (int idx = 0; idx < afterSize; idx++) {
					iy_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_after_wordnet.ComputeForwardScore(input_after_wordnet, iy_after_wordnet, oy_after_wordnet,
						fy_after_wordnet, mcy_after_wordnet,cy_after_wordnet, my_after_wordnet, y_after_wordnet);

				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						xMExp_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_before_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_before_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_before_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_before_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_before_wordnet.ComputeForwardScore(y_before_wordnet, y_before_wordnet,
							xMExp_before_wordnet, xExp_before_wordnet, xSum_before_wordnet,
							xPoolIndex_before_wordnet, y_att_before_wordnet);

					for (int idx = 0; idx < middleSize; idx++) {
						xMExp_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_middle_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_middle_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_middle_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_middle_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_middle_wordnet.ComputeForwardScore(y_middle_wordnet, y_middle_wordnet,
							xMExp_middle_wordnet, xExp_middle_wordnet, xSum_middle_wordnet,
							xPoolIndex_middle_wordnet, y_att_middle_wordnet);

					for (int idx = 0; idx < afterSize; idx++) {
						xMExp_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_after_wordnet[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_after_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_after_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_after_wordnet = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_after_wordnet.ComputeForwardScore(y_after_wordnet, y_after_wordnet,
							xMExp_after_wordnet, xExp_after_wordnet, xSum_after_wordnet,
							xPoolIndex_after_wordnet, y_att_after_wordnet);
				}
			}


			// brown channel
			vector<Tensor<xpu, 2, dtype> > iy_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > my_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > y_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityFormer_brown(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > iy_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > my_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > y_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityLatter_brown(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > iy_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > oy_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > fy_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mcy_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > cy_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > my_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > y_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > loss_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > iy_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > oy_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > fy_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > mcy_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > cy_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > my_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > y_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > loss_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > iy_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > oy_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > fy_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > cy_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > my_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > y_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > loss_after_brown(afterSize);

			vector<Tensor<xpu, 2, dtype> > xMExp_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xExp_before_brown(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_before_brown(beforeSize);
			Tensor<xpu, 2, dtype> xSum_before_brown;
			Tensor<xpu, 2, dtype> y_att_before_brown;
			Tensor<xpu, 2, dtype> ly_att_before_brown;
			vector<Tensor<xpu, 2, dtype> > xMExp_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > xExp_middle_brown(middleSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle_brown(middleSize);
			Tensor<xpu, 2, dtype> xSum_middle_brown;
			Tensor<xpu, 2, dtype> y_att_middle_brown;
			Tensor<xpu, 2, dtype> ly_att_middle_brown;
			vector<Tensor<xpu, 2, dtype> > xMExp_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > xExp_after_brown(afterSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_after_brown(afterSize);
			Tensor<xpu, 2, dtype> xSum_after_brown;
			Tensor<xpu, 2, dtype> y_att_after_brown;
			Tensor<xpu, 2, dtype> ly_att_after_brown;
			if(bBrown) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					iy_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityFormer_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				}
				unit_entityFormer_brown.ComputeForwardScore(input_entityFormer_brown, iy_entityFormer_brown
						, oy_entityFormer_brown, fy_entityFormer_brown, mcy_entityFormer_brown,
						cy_entityFormer_brown, my_entityFormer_brown, y_entityFormer_brown);
				for (int idx = 0; idx < enLatterSize; idx++) {
					iy_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityLatter_brown[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);;
				}
				unit_entityLatter_brown.ComputeForwardScore(input_entityLatter_brown, iy_entityLatter_brown,
						oy_entityLatter_brown,
						fy_entityLatter_brown, mcy_entityLatter_brown,cy_entityLatter_brown,
						my_entityLatter_brown, y_entityLatter_brown);
				for (int idx = 0; idx < beforeSize; idx++) {
					iy_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_before_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_before_brown.ComputeForwardScore(input_before_brown, iy_before_brown, oy_before_brown,
						fy_before_brown, mcy_before_brown,cy_before_brown, my_before_brown, y_before_brown);
				for (int idx = 0; idx < middleSize; idx++) {
					iy_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_middle_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_middle_brown.ComputeForwardScore(input_middle_brown, iy_middle_brown, oy_middle_brown,
						fy_middle_brown, mcy_middle_brown,cy_middle_brown, my_middle_brown, y_middle_brown);
				for (int idx = 0; idx < afterSize; idx++) {
					iy_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_after_brown[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_after_brown.ComputeForwardScore(input_after_brown, iy_after_brown, oy_after_brown,
						fy_after_brown, mcy_after_brown,cy_after_brown, my_after_brown, y_after_brown);

				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						xMExp_before_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_before_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_before_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_before_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_before_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_before_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_before_brown.ComputeForwardScore(y_before_brown, y_before_brown,
							xMExp_before_brown, xExp_before_brown, xSum_before_brown,
							xPoolIndex_before_brown, y_att_before_brown);

					for (int idx = 0; idx < middleSize; idx++) {
						xMExp_middle_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_middle_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_middle_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_middle_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_middle_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_middle_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_middle_brown.ComputeForwardScore(y_middle_brown, y_middle_brown,
							xMExp_middle_brown, xExp_middle_brown, xSum_middle_brown,
							xPoolIndex_middle_brown, y_att_middle_brown);

					for (int idx = 0; idx < afterSize; idx++) {
						xMExp_after_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_after_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_after_brown[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_after_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_after_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_after_brown = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_after_brown.ComputeForwardScore(y_after_brown, y_after_brown,
							xMExp_after_brown, xExp_after_brown, xSum_after_brown,
							xPoolIndex_after_brown, y_att_after_brown);
				}
			}


			// bigram channel
			vector<Tensor<xpu, 2, dtype> > iy_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > my_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > y_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityFormer_bigram(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > iy_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > my_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > y_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityLatter_bigram(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > iy_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > oy_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > fy_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mcy_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > cy_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > my_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > y_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > loss_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > iy_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > oy_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > fy_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > mcy_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > cy_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > my_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > y_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > loss_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > iy_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > oy_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > fy_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > cy_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > my_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > y_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > loss_after_bigram(afterSize);

			vector<Tensor<xpu, 2, dtype> > xMExp_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xExp_before_bigram(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_before_bigram(beforeSize);
			Tensor<xpu, 2, dtype> xSum_before_bigram;
			Tensor<xpu, 2, dtype> y_att_before_bigram;
			Tensor<xpu, 2, dtype> ly_att_before_bigram;
			vector<Tensor<xpu, 2, dtype> > xMExp_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > xExp_middle_bigram(middleSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle_bigram(middleSize);
			Tensor<xpu, 2, dtype> xSum_middle_bigram;
			Tensor<xpu, 2, dtype> y_att_middle_bigram;
			Tensor<xpu, 2, dtype> ly_att_middle_bigram;
			vector<Tensor<xpu, 2, dtype> > xMExp_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > xExp_after_bigram(afterSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_after_bigram(afterSize);
			Tensor<xpu, 2, dtype> xSum_after_bigram;
			Tensor<xpu, 2, dtype> y_att_after_bigram;
			Tensor<xpu, 2, dtype> ly_att_after_bigram;
			if(bBigram) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					iy_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityFormer_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				}
				unit_entityFormer_bigram.ComputeForwardScore(input_entityFormer_bigram, iy_entityFormer_bigram
						, oy_entityFormer_bigram, fy_entityFormer_bigram, mcy_entityFormer_bigram,
						cy_entityFormer_bigram, my_entityFormer_bigram, y_entityFormer_bigram);
				for (int idx = 0; idx < enLatterSize; idx++) {
					iy_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityLatter_bigram[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);;
				}
				unit_entityLatter_bigram.ComputeForwardScore(input_entityLatter_bigram, iy_entityLatter_bigram,
						oy_entityLatter_bigram,
						fy_entityLatter_bigram, mcy_entityLatter_bigram,cy_entityLatter_bigram,
						my_entityLatter_bigram, y_entityLatter_bigram);
				for (int idx = 0; idx < beforeSize; idx++) {
					iy_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_before_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_before_bigram.ComputeForwardScore(input_before_bigram, iy_before_bigram, oy_before_bigram,
						fy_before_bigram, mcy_before_bigram,cy_before_bigram, my_before_bigram, y_before_bigram);
				for (int idx = 0; idx < middleSize; idx++) {
					iy_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_middle_bigram.ComputeForwardScore(input_middle_bigram, iy_middle_bigram, oy_middle_bigram,
						fy_middle_bigram, mcy_middle_bigram,cy_middle_bigram, my_middle_bigram, y_middle_bigram);
				for (int idx = 0; idx < afterSize; idx++) {
					iy_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_after_bigram[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_after_bigram.ComputeForwardScore(input_after_bigram, iy_after_bigram, oy_after_bigram,
						fy_after_bigram, mcy_after_bigram,cy_after_bigram, my_after_bigram, y_after_bigram);

				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						xMExp_before_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_before_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_before_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_before_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_before_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_before_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_before_bigram.ComputeForwardScore(y_before_bigram, y_before_bigram,
							xMExp_before_bigram, xExp_before_bigram, xSum_before_bigram,
							xPoolIndex_before_bigram, y_att_before_bigram);

					for (int idx = 0; idx < middleSize; idx++) {
						xMExp_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_middle_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_middle_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_middle_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_middle_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_middle_bigram.ComputeForwardScore(y_middle_bigram, y_middle_bigram,
							xMExp_middle_bigram, xExp_middle_bigram, xSum_middle_bigram,
							xPoolIndex_middle_bigram, y_att_middle_bigram);

					for (int idx = 0; idx < afterSize; idx++) {
						xMExp_after_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_after_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_after_bigram[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_after_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_after_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_after_bigram = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_after_bigram.ComputeForwardScore(y_after_bigram, y_after_bigram,
							xMExp_after_bigram, xExp_after_bigram, xSum_after_bigram,
							xPoolIndex_after_bigram, y_att_after_bigram);
				}
			}


			// pos channel
			vector<Tensor<xpu, 2, dtype> > iy_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > my_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > y_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityFormer_pos(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > iy_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > my_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > y_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityLatter_pos(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > iy_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > oy_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > fy_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mcy_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > cy_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > my_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > y_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > loss_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > iy_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > oy_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > fy_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > mcy_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > cy_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > my_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > y_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > loss_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > iy_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > oy_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > fy_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > cy_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > my_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > y_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > loss_after_pos(afterSize);

			vector<Tensor<xpu, 2, dtype> > xMExp_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xExp_before_pos(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_before_pos(beforeSize);
			Tensor<xpu, 2, dtype> xSum_before_pos;
			Tensor<xpu, 2, dtype> y_att_before_pos;
			Tensor<xpu, 2, dtype> ly_att_before_pos;
			vector<Tensor<xpu, 2, dtype> > xMExp_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > xExp_middle_pos(middleSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle_pos(middleSize);
			Tensor<xpu, 2, dtype> xSum_middle_pos;
			Tensor<xpu, 2, dtype> y_att_middle_pos;
			Tensor<xpu, 2, dtype> ly_att_middle_pos;
			vector<Tensor<xpu, 2, dtype> > xMExp_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > xExp_after_pos(afterSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_after_pos(afterSize);
			Tensor<xpu, 2, dtype> xSum_after_pos;
			Tensor<xpu, 2, dtype> y_att_after_pos;
			Tensor<xpu, 2, dtype> ly_att_after_pos;
			if(bPos) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					iy_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityFormer_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				}
				unit_entityFormer_pos.ComputeForwardScore(input_entityFormer_pos, iy_entityFormer_pos
						, oy_entityFormer_pos, fy_entityFormer_pos, mcy_entityFormer_pos,
						cy_entityFormer_pos, my_entityFormer_pos, y_entityFormer_pos);
				for (int idx = 0; idx < enLatterSize; idx++) {
					iy_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityLatter_pos[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);;
				}
				unit_entityLatter_pos.ComputeForwardScore(input_entityLatter_pos, iy_entityLatter_pos,
						oy_entityLatter_pos,
						fy_entityLatter_pos, mcy_entityLatter_pos,cy_entityLatter_pos,
						my_entityLatter_pos, y_entityLatter_pos);
				for (int idx = 0; idx < beforeSize; idx++) {
					iy_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_before_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_before_pos.ComputeForwardScore(input_before_pos, iy_before_pos, oy_before_pos,
						fy_before_pos, mcy_before_pos,cy_before_pos, my_before_pos, y_before_pos);
				for (int idx = 0; idx < middleSize; idx++) {
					iy_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_middle_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_middle_pos.ComputeForwardScore(input_middle_pos, iy_middle_pos, oy_middle_pos,
						fy_middle_pos, mcy_middle_pos,cy_middle_pos, my_middle_pos, y_middle_pos);
				for (int idx = 0; idx < afterSize; idx++) {
					iy_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_after_pos[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_after_pos.ComputeForwardScore(input_after_pos, iy_after_pos, oy_after_pos,
						fy_after_pos, mcy_after_pos,cy_after_pos, my_after_pos, y_after_pos);

				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						xMExp_before_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_before_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_before_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_before_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_before_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_before_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_before_pos.ComputeForwardScore(y_before_pos, y_before_pos,
							xMExp_before_pos, xExp_before_pos, xSum_before_pos,
							xPoolIndex_before_pos, y_att_before_pos);

					for (int idx = 0; idx < middleSize; idx++) {
						xMExp_middle_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_middle_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_middle_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_middle_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_middle_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_middle_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_middle_pos.ComputeForwardScore(y_middle_pos, y_middle_pos,
							xMExp_middle_pos, xExp_middle_pos, xSum_middle_pos,
							xPoolIndex_middle_pos, y_att_middle_pos);

					for (int idx = 0; idx < afterSize; idx++) {
						xMExp_after_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_after_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_after_pos[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_after_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_after_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_after_pos = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_after_pos.ComputeForwardScore(y_after_pos, y_after_pos,
							xMExp_after_pos, xExp_after_pos, xSum_after_pos,
							xPoolIndex_after_pos, y_att_after_pos);
				}
			}

			// sst channel
			vector<Tensor<xpu, 2, dtype> > iy_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > my_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > y_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityFormer_sst(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > iy_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > my_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > y_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityLatter_sst(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > iy_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > oy_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > fy_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mcy_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > cy_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > my_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > y_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > loss_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > iy_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > oy_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > fy_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > mcy_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > cy_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > my_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > y_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > loss_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > iy_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > oy_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > fy_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > cy_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > my_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > y_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > loss_after_sst(afterSize);

			vector<Tensor<xpu, 2, dtype> > xMExp_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xExp_before_sst(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_before_sst(beforeSize);
			Tensor<xpu, 2, dtype> xSum_before_sst;
			Tensor<xpu, 2, dtype> y_att_before_sst;
			Tensor<xpu, 2, dtype> ly_att_before_sst;
			vector<Tensor<xpu, 2, dtype> > xMExp_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > xExp_middle_sst(middleSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle_sst(middleSize);
			Tensor<xpu, 2, dtype> xSum_middle_sst;
			Tensor<xpu, 2, dtype> y_att_middle_sst;
			Tensor<xpu, 2, dtype> ly_att_middle_sst;
			vector<Tensor<xpu, 2, dtype> > xMExp_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > xExp_after_sst(afterSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_after_sst(afterSize);
			Tensor<xpu, 2, dtype> xSum_after_sst;
			Tensor<xpu, 2, dtype> y_att_after_sst;
			Tensor<xpu, 2, dtype> ly_att_after_sst;
			if(bSst) {
				for (int idx = 0; idx < enFormerSize; idx++) {
					iy_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityFormer_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				}
				unit_entityFormer_sst.ComputeForwardScore(input_entityFormer_sst, iy_entityFormer_sst
						, oy_entityFormer_sst, fy_entityFormer_sst, mcy_entityFormer_sst,
						cy_entityFormer_sst, my_entityFormer_sst, y_entityFormer_sst);
				for (int idx = 0; idx < enLatterSize; idx++) {
					iy_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					oy_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					fy_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					mcy_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					cy_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					my_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					y_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
					loss_entityLatter_sst[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);;
				}
				unit_entityLatter_sst.ComputeForwardScore(input_entityLatter_sst, iy_entityLatter_sst,
						oy_entityLatter_sst,
						fy_entityLatter_sst, mcy_entityLatter_sst,cy_entityLatter_sst,
						my_entityLatter_sst, y_entityLatter_sst);
				for (int idx = 0; idx < beforeSize; idx++) {
					iy_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_before_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_before_sst.ComputeForwardScore(input_before_sst, iy_before_sst, oy_before_sst,
						fy_before_sst, mcy_before_sst,cy_before_sst, my_before_sst, y_before_sst);
				for (int idx = 0; idx < middleSize; idx++) {
					iy_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_middle_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_middle_sst.ComputeForwardScore(input_middle_sst, iy_middle_sst, oy_middle_sst,
						fy_middle_sst, mcy_middle_sst,cy_middle_sst, my_middle_sst, y_middle_sst);
				for (int idx = 0; idx < afterSize; idx++) {
					iy_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					oy_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					fy_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					mcy_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					cy_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					my_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					y_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
					loss_after_sst[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				unit_after_sst.ComputeForwardScore(input_after_sst, iy_after_sst, oy_after_sst,
						fy_after_sst, mcy_after_sst,cy_after_sst, my_after_sst, y_after_sst);

				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						xMExp_before_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_before_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_before_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_before_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_before_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_before_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_before_sst.ComputeForwardScore(y_before_sst, y_before_sst,
							xMExp_before_sst, xExp_before_sst, xSum_before_sst,
							xPoolIndex_before_sst, y_att_before_sst);

					for (int idx = 0; idx < middleSize; idx++) {
						xMExp_middle_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_middle_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_middle_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_middle_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_middle_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_middle_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_middle_sst.ComputeForwardScore(y_middle_sst, y_middle_sst,
							xMExp_middle_sst, xExp_middle_sst, xSum_middle_sst,
							xPoolIndex_middle_sst, y_att_middle_sst);

					for (int idx = 0; idx < afterSize; idx++) {
						xMExp_after_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xExp_after_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
						xPoolIndex_after_sst[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					}
					xSum_after_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					y_att_after_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					ly_att_after_sst = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					unit_att_after_sst.ComputeForwardScore(y_after_sst, y_after_sst,
							xMExp_after_sst, xExp_after_sst, xSum_after_sst,
							xPoolIndex_after_sst, y_att_after_sst);
				}
			}


			vector<Tensor<xpu, 2, dtype> > v_hidden_input;

			// word channel
			if(bWord) {

				if(options.attention) {
					v_hidden_input.push_back(y_att_before);
				} else
					v_hidden_input.push_back(y_before[beforeSize-1]);

				v_hidden_input.push_back(y_entityFormer[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_middle);
				} else
					v_hidden_input.push_back(y_middle[middleSize-1]);

				v_hidden_input.push_back(y_entityLatter[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_after);
				} else
					v_hidden_input.push_back(y_after[afterSize-1]);
			}

			// wordnet channel
			if(bWordnet) {

				if(options.attention) {
					v_hidden_input.push_back(y_att_before_wordnet);
				} else
					v_hidden_input.push_back(y_before_wordnet[beforeSize-1]);

				v_hidden_input.push_back(y_entityFormer_wordnet[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_middle_wordnet);
				} else
					v_hidden_input.push_back(y_middle_wordnet[middleSize-1]);

				v_hidden_input.push_back(y_entityLatter_wordnet[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_after_wordnet);
				} else
					v_hidden_input.push_back(y_after_wordnet[afterSize-1]);
			}


			// brown channel
			if(bBrown) {

				if(options.attention) {
					v_hidden_input.push_back(y_att_before_brown);
				} else
					v_hidden_input.push_back(y_before_brown[beforeSize-1]);

				v_hidden_input.push_back(y_entityFormer_brown[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_middle_brown);
				} else
					v_hidden_input.push_back(y_middle_brown[middleSize-1]);

				v_hidden_input.push_back(y_entityLatter_brown[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_after_brown);
				} else
					v_hidden_input.push_back(y_after_brown[afterSize-1]);
			}


			// bigram channel
			if(bBigram) {

				if(options.attention) {
					v_hidden_input.push_back(y_att_before_bigram);
				} else
					v_hidden_input.push_back(y_before_bigram[beforeSize-1]);

				v_hidden_input.push_back(y_entityFormer_bigram[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_middle_bigram);
				} else
					v_hidden_input.push_back(y_middle_bigram[middleSize-1]);

				v_hidden_input.push_back(y_entityLatter_bigram[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_after_bigram);
				} else
					v_hidden_input.push_back(y_after_bigram[afterSize-1]);
			}


			// pos channel
			if(bPos) {

				if(options.attention) {
					v_hidden_input.push_back(y_att_before_pos);
				} else
					v_hidden_input.push_back(y_before_pos[beforeSize-1]);

				v_hidden_input.push_back(y_entityFormer_pos[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_middle_pos);
				} else
					v_hidden_input.push_back(y_middle_pos[middleSize-1]);

				v_hidden_input.push_back(y_entityLatter_pos[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_after_pos);
				} else
					v_hidden_input.push_back(y_after_pos[afterSize-1]);
			}


			// sst channel
			if(bSst) {

				if(options.attention) {
					v_hidden_input.push_back(y_att_before_sst);
				} else
					v_hidden_input.push_back(y_before_sst[beforeSize-1]);

				v_hidden_input.push_back(y_entityFormer_sst[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_middle_sst);
				} else
					v_hidden_input.push_back(y_middle_sst[middleSize-1]);

				v_hidden_input.push_back(y_entityLatter_sst[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input.push_back(y_att_after_sst);
				} else
					v_hidden_input.push_back(y_after_sst[afterSize-1]);
			}



			Tensor<xpu, 2, dtype> hidden_input = NewTensor<xpu>(Shape2(1, _hidden_input_size), d_zero);
			concat(v_hidden_input, hidden_input);


			hidden_layer.ComputeForwardScore(hidden_input, hidden);

			// hidden -> output
			output_layer.ComputeForwardScore(hidden, output);

			// get delta for each output
			cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

			// loss backward propagation
			// output
			output_layer.ComputeBackwardLoss(hidden, output, outputLoss, hiddenLoss);

			// hidden
			Tensor<xpu, 2, dtype> hidden_input_loss = NewTensor<xpu>(Shape2(1, _hidden_input_size), d_zero);
			hidden_layer.ComputeBackwardLoss(hidden_input, hidden, hiddenLoss, hidden_input_loss);


			vector<Tensor<xpu, 2, dtype> > v_hidden_input_loss;

			// word channel
			if(bWord) {

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_before);
				} else
					v_hidden_input_loss.push_back(loss_before[beforeSize-1]);

				v_hidden_input_loss.push_back(loss_entityFormer[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_middle);
				} else
					v_hidden_input_loss.push_back(loss_middle[middleSize-1]);

				v_hidden_input_loss.push_back(loss_entityLatter[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_after);
				} else
					v_hidden_input_loss.push_back(loss_after[afterSize-1]);
			}


			// wordnet channel
			if(bWordnet) {

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_before_wordnet);
				} else
					v_hidden_input_loss.push_back(loss_before_wordnet[beforeSize-1]);

				v_hidden_input_loss.push_back(loss_entityFormer_wordnet[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_middle_wordnet);
				} else
					v_hidden_input_loss.push_back(loss_middle_wordnet[middleSize-1]);

				v_hidden_input_loss.push_back(loss_entityLatter_wordnet[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_after_wordnet);
				} else
					v_hidden_input_loss.push_back(loss_after_wordnet[afterSize-1]);
			}


			// brown channel
			if(bBrown) {

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_before_brown);
				} else
					v_hidden_input_loss.push_back(loss_before_brown[beforeSize-1]);

				v_hidden_input_loss.push_back(loss_entityFormer_brown[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_middle_brown);
				} else
					v_hidden_input_loss.push_back(loss_middle_brown[middleSize-1]);

				v_hidden_input_loss.push_back(loss_entityLatter_brown[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_after_brown);
				} else
					v_hidden_input_loss.push_back(loss_after_brown[afterSize-1]);

			}

			// bigram channel
			if(bBigram) {

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_before_bigram);
				} else
					v_hidden_input_loss.push_back(loss_before_bigram[beforeSize-1]);

				v_hidden_input_loss.push_back(loss_entityFormer_bigram[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_middle_bigram);
				} else
					v_hidden_input_loss.push_back(loss_middle_bigram[middleSize-1]);

				v_hidden_input_loss.push_back(loss_entityLatter_bigram[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_after_bigram);
				} else
					v_hidden_input_loss.push_back(loss_after_bigram[afterSize-1]);
			}


			// pos channel
			if(bPos) {

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_before_pos);
				} else
					v_hidden_input_loss.push_back(loss_before_pos[beforeSize-1]);

				v_hidden_input_loss.push_back(loss_entityFormer_pos[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_middle_pos);
				} else
					v_hidden_input_loss.push_back(loss_middle_pos[middleSize-1]);

				v_hidden_input_loss.push_back(loss_entityLatter_pos[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_after_pos);
				} else
					v_hidden_input_loss.push_back(loss_after_pos[afterSize-1]);
			}

			// sst channel
			if(bSst) {

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_before_sst);
				} else
					v_hidden_input_loss.push_back(loss_before_sst[beforeSize-1]);

				v_hidden_input_loss.push_back(loss_entityFormer_sst[enFormerSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_middle_sst);
				} else
					v_hidden_input_loss.push_back(loss_middle_sst[middleSize-1]);

				v_hidden_input_loss.push_back(loss_entityLatter_sst[enLatterSize-1]);

				if(options.attention) {
					v_hidden_input_loss.push_back(ly_att_after_sst);
				} else
					v_hidden_input_loss.push_back(loss_after_sst[afterSize-1]);
			}


			unconcat(v_hidden_input_loss, hidden_input_loss);


			// word channel
			if(bWord) {
				if(options.attention) {
					unit_att_before.ComputeBackwardLoss(y_before, y_before,
							xMExp_before, xExp_before, xSum_before,
							xPoolIndex_before, y_att_before,
							ly_att_before, loss_before, loss_before);

					unit_att_middle.ComputeBackwardLoss(y_middle, y_middle,
							xMExp_middle, xExp_middle, xSum_middle,
							xPoolIndex_middle, y_att_middle,
							ly_att_middle, loss_middle, loss_middle);

					unit_att_after.ComputeBackwardLoss(y_after, y_after,
							xMExp_after, xExp_after, xSum_after,
							xPoolIndex_after, y_att_after,
							ly_att_after, loss_after, loss_after);
				}
				unit_before.ComputeBackwardLoss(input_before, iy_before, oy_before,
								      fy_before, mcy_before, cy_before, my_before,
								      y_before, loss_before, inputLoss_before);
				unit_middle.ComputeBackwardLoss(input_middle, iy_middle, oy_middle,
								      fy_middle, mcy_middle, cy_middle, my_middle,
								      y_middle, loss_middle, inputLoss_middle);
				unit_after.ComputeBackwardLoss(input_after, iy_after, oy_after,
								      fy_after, mcy_after, cy_after, my_after,
								      y_after, loss_after, inputLoss_after);

				unit_entityFormer.ComputeBackwardLoss(input_entityFormer, iy_entityFormer, oy_entityFormer,
					      fy_entityFormer, mcy_entityFormer, cy_entityFormer, my_entityFormer,
					      y_entityFormer, loss_entityFormer, inputLoss_entityFormer);

				unit_entityLatter.ComputeBackwardLoss(input_entityLatter, iy_entityLatter, oy_entityLatter,
								      fy_entityLatter, mcy_entityLatter, cy_entityLatter, my_entityLatter,
								      y_entityLatter, loss_entityLatter, inputLoss_entityLatter);

			}


			// wordnet channel
			if(bWordnet) {
				if(options.attention) {
					unit_att_before_wordnet.ComputeBackwardLoss(y_before_wordnet, y_before_wordnet,
							xMExp_before_wordnet, xExp_before_wordnet, xSum_before_wordnet,
							xPoolIndex_before_wordnet, y_att_before_wordnet,
							ly_att_before_wordnet, loss_before_wordnet, loss_before_wordnet);

					unit_att_middle_wordnet.ComputeBackwardLoss(y_middle_wordnet, y_middle_wordnet,
							xMExp_middle_wordnet, xExp_middle_wordnet, xSum_middle_wordnet,
							xPoolIndex_middle_wordnet, y_att_middle_wordnet,
							ly_att_middle_wordnet,loss_middle_wordnet, loss_middle_wordnet);

					unit_att_after_wordnet.ComputeBackwardLoss(y_after_wordnet, y_after_wordnet,
							xMExp_after_wordnet, xExp_after_wordnet, xSum_after_wordnet,
							xPoolIndex_after_wordnet, y_att_after_wordnet,
							ly_att_after_wordnet, loss_after_wordnet, loss_after_wordnet);
				}
				unit_before_wordnet.ComputeBackwardLoss(input_before_wordnet, iy_before_wordnet, oy_before_wordnet,
								      fy_before_wordnet, mcy_before_wordnet, cy_before_wordnet, my_before_wordnet,
								      y_before_wordnet, loss_before_wordnet, inputLoss_before_wordnet);
				unit_middle_wordnet.ComputeBackwardLoss(input_middle_wordnet, iy_middle_wordnet, oy_middle_wordnet,
								      fy_middle_wordnet, mcy_middle_wordnet, cy_middle_wordnet, my_middle_wordnet,
								      y_middle_wordnet, loss_middle_wordnet, inputLoss_middle_wordnet);
				unit_after_wordnet.ComputeBackwardLoss(input_after_wordnet, iy_after_wordnet, oy_after_wordnet,
								      fy_after_wordnet, mcy_after_wordnet, cy_after_wordnet, my_after_wordnet,
								      y_after_wordnet, loss_after_wordnet, inputLoss_after_wordnet);
				unit_entityFormer_wordnet.ComputeBackwardLoss(input_entityFormer_wordnet, iy_entityFormer_wordnet, oy_entityFormer_wordnet,
								      fy_entityFormer_wordnet, mcy_entityFormer_wordnet, cy_entityFormer_wordnet, my_entityFormer_wordnet,
								      y_entityFormer_wordnet, loss_entityFormer_wordnet, inputLoss_entityFormer_wordnet);
				unit_entityLatter_wordnet.ComputeBackwardLoss(input_entityLatter_wordnet, iy_entityLatter_wordnet, oy_entityLatter_wordnet,
								      fy_entityLatter_wordnet, mcy_entityLatter_wordnet, cy_entityLatter_wordnet, my_entityLatter_wordnet,
								      y_entityLatter_wordnet, loss_entityLatter_wordnet, inputLoss_entityLatter_wordnet);

			}

			// brown channel
			if(bBrown) {
				if(options.attention) {
					unit_att_before_brown.ComputeBackwardLoss(y_before_brown, y_before_brown,
							xMExp_before_brown, xExp_before_brown, xSum_before_brown,
							xPoolIndex_before_brown, y_att_before_brown,
							ly_att_before_brown, loss_before_brown, loss_before_brown);

					unit_att_middle_brown.ComputeBackwardLoss(y_middle_brown, y_middle_brown,
							xMExp_middle_brown, xExp_middle_brown, xSum_middle_brown,
							xPoolIndex_middle_brown, y_att_middle_brown,
							ly_att_middle_brown, loss_middle_brown, loss_middle_brown);

					unit_att_after_brown.ComputeBackwardLoss(y_after_brown, y_after_brown,
							xMExp_after_brown, xExp_after_brown, xSum_after_brown,
							xPoolIndex_after_brown, y_att_after_brown,
							ly_att_after_brown, loss_after_brown, loss_after_brown);
				}
				unit_before_brown.ComputeBackwardLoss(input_before_brown, iy_before_brown, oy_before_brown,
									  fy_before_brown, mcy_before_brown, cy_before_brown, my_before_brown,
									  y_before_brown, loss_before_brown, inputLoss_before_brown);
				unit_middle_brown.ComputeBackwardLoss(input_middle_brown, iy_middle_brown, oy_middle_brown,
									  fy_middle_brown, mcy_middle_brown, cy_middle_brown, my_middle_brown,
									  y_middle_brown, loss_middle_brown, inputLoss_middle_brown);
				unit_after_brown.ComputeBackwardLoss(input_after_brown, iy_after_brown, oy_after_brown,
									  fy_after_brown, mcy_after_brown, cy_after_brown, my_after_brown,
									  y_after_brown, loss_after_brown, inputLoss_after_brown);
				unit_entityFormer_brown.ComputeBackwardLoss(input_entityFormer_brown, iy_entityFormer_brown, oy_entityFormer_brown,
									  fy_entityFormer_brown, mcy_entityFormer_brown, cy_entityFormer_brown, my_entityFormer_brown,
									  y_entityFormer_brown, loss_entityFormer_brown, inputLoss_entityFormer_brown);
				unit_entityLatter_brown.ComputeBackwardLoss(input_entityLatter_brown, iy_entityLatter_brown, oy_entityLatter_brown,
									  fy_entityLatter_brown, mcy_entityLatter_brown, cy_entityLatter_brown, my_entityLatter_brown,
									  y_entityLatter_brown, loss_entityLatter_brown, inputLoss_entityLatter_brown);

			}

			// bigram channel
			if(bBigram) {
				if(options.attention) {
					unit_att_before_bigram.ComputeBackwardLoss(y_before_bigram, y_before_bigram,
							xMExp_before_bigram, xExp_before_bigram, xSum_before_bigram,
							xPoolIndex_before_bigram, y_att_before_bigram,
							ly_att_before_bigram, loss_before_bigram, loss_before_bigram);

					unit_att_middle_bigram.ComputeBackwardLoss(y_middle_bigram, y_middle_bigram,
							xMExp_middle_bigram, xExp_middle_bigram, xSum_middle_bigram,
							xPoolIndex_middle_bigram, y_att_middle_bigram,
							ly_att_middle_bigram, loss_middle_bigram, loss_middle_bigram);

					unit_att_after_bigram.ComputeBackwardLoss(y_after_bigram, y_after_bigram,
							xMExp_after_bigram, xExp_after_bigram, xSum_after_bigram,
							xPoolIndex_after_bigram, y_att_after_bigram,
							ly_att_after_bigram, loss_after_bigram, loss_after_bigram);
				}
				unit_before_bigram.ComputeBackwardLoss(input_before_bigram, iy_before_bigram, oy_before_bigram,
									  fy_before_bigram, mcy_before_bigram, cy_before_bigram, my_before_bigram,
									  y_before_bigram, loss_before_bigram, inputLoss_before_bigram);
				unit_middle_bigram.ComputeBackwardLoss(input_middle_bigram, iy_middle_bigram, oy_middle_bigram,
									  fy_middle_bigram, mcy_middle_bigram, cy_middle_bigram, my_middle_bigram,
									  y_middle_bigram, loss_middle_bigram, inputLoss_middle_bigram);
				unit_after_bigram.ComputeBackwardLoss(input_after_bigram, iy_after_bigram, oy_after_bigram,
									  fy_after_bigram, mcy_after_bigram, cy_after_bigram, my_after_bigram,
									  y_after_bigram, loss_after_bigram, inputLoss_after_bigram);
				unit_entityFormer_bigram.ComputeBackwardLoss(input_entityFormer_bigram, iy_entityFormer_bigram, oy_entityFormer_bigram,
									  fy_entityFormer_bigram, mcy_entityFormer_bigram, cy_entityFormer_bigram, my_entityFormer_bigram,
									  y_entityFormer_bigram, loss_entityFormer_bigram, inputLoss_entityFormer_bigram);
				unit_entityLatter_bigram.ComputeBackwardLoss(input_entityLatter_bigram, iy_entityLatter_bigram, oy_entityLatter_bigram,
									  fy_entityLatter_bigram, mcy_entityLatter_bigram, cy_entityLatter_bigram, my_entityLatter_bigram,
									  y_entityLatter_bigram, loss_entityLatter_bigram, inputLoss_entityLatter_bigram);

			}

			// pos channel
			if(bPos) {
				if(options.attention) {
					unit_att_before_pos.ComputeBackwardLoss(y_before_pos, y_before_pos,
							xMExp_before_pos, xExp_before_pos, xSum_before_pos,
							xPoolIndex_before_pos, y_att_before_pos,
							ly_att_before_pos, loss_before_pos, loss_before_pos);

					unit_att_middle_pos.ComputeBackwardLoss(y_middle_pos, y_middle_pos,
							xMExp_middle_pos, xExp_middle_pos, xSum_middle_pos,
							xPoolIndex_middle_pos, y_att_middle_pos,
							ly_att_middle_pos, loss_middle_pos, loss_middle_pos);

					unit_att_after_pos.ComputeBackwardLoss(y_after_pos, y_after_pos,
							xMExp_after_pos, xExp_after_pos, xSum_after_pos,
							xPoolIndex_after_pos, y_att_after_pos,
							ly_att_after_pos, loss_after_pos, loss_after_pos);
				}
				unit_before_pos.ComputeBackwardLoss(input_before_pos, iy_before_pos, oy_before_pos,
									  fy_before_pos, mcy_before_pos, cy_before_pos, my_before_pos,
									  y_before_pos, loss_before_pos, inputLoss_before_pos);
				unit_middle_pos.ComputeBackwardLoss(input_middle_pos, iy_middle_pos, oy_middle_pos,
									  fy_middle_pos, mcy_middle_pos, cy_middle_pos, my_middle_pos,
									  y_middle_pos, loss_middle_pos, inputLoss_middle_pos);
				unit_after_pos.ComputeBackwardLoss(input_after_pos, iy_after_pos, oy_after_pos,
									  fy_after_pos, mcy_after_pos, cy_after_pos, my_after_pos,
									  y_after_pos, loss_after_pos, inputLoss_after_pos);
				unit_entityFormer_pos.ComputeBackwardLoss(input_entityFormer_pos, iy_entityFormer_pos, oy_entityFormer_pos,
									  fy_entityFormer_pos, mcy_entityFormer_pos, cy_entityFormer_pos, my_entityFormer_pos,
									  y_entityFormer_pos, loss_entityFormer_pos, inputLoss_entityFormer_pos);
				unit_entityLatter_pos.ComputeBackwardLoss(input_entityLatter_pos, iy_entityLatter_pos, oy_entityLatter_pos,
									  fy_entityLatter_pos, mcy_entityLatter_pos, cy_entityLatter_pos, my_entityLatter_pos,
									  y_entityLatter_pos, loss_entityLatter_pos, inputLoss_entityLatter_pos);

			}

			// sst channel
			if(bSst) {
				if(options.attention) {
					unit_att_before_sst.ComputeBackwardLoss(y_before_sst, y_before_sst,
							xMExp_before_sst, xExp_before_sst, xSum_before_sst,
							xPoolIndex_before_sst, y_att_before_sst,
							ly_att_before_sst, loss_before_sst, loss_before_sst);

					unit_att_middle_sst.ComputeBackwardLoss(y_middle_sst, y_middle_sst,
							xMExp_middle_sst, xExp_middle_sst, xSum_middle_sst,
							xPoolIndex_middle_sst, y_att_middle_sst,
							ly_att_middle_sst, loss_middle_sst, loss_middle_sst);

					unit_att_after_sst.ComputeBackwardLoss(y_after_sst, y_after_sst,
							xMExp_after_sst, xExp_after_sst, xSum_after_sst,
							xPoolIndex_after_sst, y_att_after_sst,
							ly_att_after_sst, loss_after_sst, loss_after_sst);
				}

				unit_before_sst.ComputeBackwardLoss(input_before_sst, iy_before_sst, oy_before_sst,
									  fy_before_sst, mcy_before_sst, cy_before_sst, my_before_sst,
									  y_before_sst, loss_before_sst, inputLoss_before_sst);
				unit_middle_sst.ComputeBackwardLoss(input_middle_sst, iy_middle_sst, oy_middle_sst,
									  fy_middle_sst, mcy_middle_sst, cy_middle_sst, my_middle_sst,
									  y_middle_sst, loss_middle_sst, inputLoss_middle_sst);
				unit_after_sst.ComputeBackwardLoss(input_after_sst, iy_after_sst, oy_after_sst,
									  fy_after_sst, mcy_after_sst, cy_after_sst, my_after_sst,
									  y_after_sst, loss_after_sst, inputLoss_after_sst);
				unit_entityFormer_sst.ComputeBackwardLoss(input_entityFormer_sst, iy_entityFormer_sst, oy_entityFormer_sst,
									  fy_entityFormer_sst, mcy_entityFormer_sst, cy_entityFormer_sst, my_entityFormer_sst,
									  y_entityFormer_sst, loss_entityFormer_sst, inputLoss_entityFormer_sst);
				unit_entityLatter_sst.ComputeBackwardLoss(input_entityLatter_sst, iy_entityLatter_sst, oy_entityLatter_sst,
									  fy_entityLatter_sst, mcy_entityLatter_sst, cy_entityLatter_sst, my_entityLatter_sst,
									  y_entityLatter_sst, loss_entityLatter_sst, inputLoss_entityLatter_sst);

			}


			// word channel
			if (bWord && _words.bEmbFineTune()) {
				for (int idx = 0; idx < beforeSize; idx++) {
					inputLoss_before[idx] = inputLoss_before[idx] * mask_before[idx];
					_words.EmbLoss(example.m_before[idx], inputLoss_before[idx]);
				}
				for (int idx = 0; idx < enFormerSize; idx++) {
					inputLoss_entityFormer[idx] = inputLoss_entityFormer[idx] * mask_entityFormer[idx];
					_words.EmbLoss(example.m_entityFormer[idx], inputLoss_entityFormer[idx]);
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					inputLoss_entityLatter[idx] = inputLoss_entityLatter[idx] * mask_entityLatter[idx];
					_words.EmbLoss(example.m_entityLatter[idx], inputLoss_entityLatter[idx]);
				}
				for (int idx = 0; idx < middleSize; idx++) {
					inputLoss_middle[idx] = inputLoss_middle[idx] * mask_middle[idx];
					_words.EmbLoss(example.m_middle[idx], inputLoss_middle[idx]);
				}
				for (int idx = 0; idx < afterSize; idx++) {
					inputLoss_after[idx] = inputLoss_after[idx] * mask_after[idx];
					_words.EmbLoss(example.m_after[idx], inputLoss_after[idx]);
				}
			}

			// wordnet channel
			if (bWordnet && _wordnet.bEmbFineTune()) {
				for (int idx = 0; idx < beforeSize; idx++) {
					inputLoss_before_wordnet[idx] = inputLoss_before_wordnet[idx] * mask_before_wordnet[idx];
					_wordnet.EmbLoss(example.m_before_wordnet[idx], inputLoss_before_wordnet[idx]);
				}
				for (int idx = 0; idx < middleSize; idx++) {
					inputLoss_middle_wordnet[idx] = inputLoss_middle_wordnet[idx] * mask_middle_wordnet[idx];
					_wordnet.EmbLoss(example.m_middle_wordnet[idx], inputLoss_middle_wordnet[idx]);
				}
				for (int idx = 0; idx < afterSize; idx++) {
					inputLoss_after_wordnet[idx] = inputLoss_after_wordnet[idx] * mask_after_wordnet[idx];
					_wordnet.EmbLoss(example.m_after_wordnet[idx], inputLoss_after_wordnet[idx]);
				}
				for (int idx = 0; idx < enFormerSize; idx++) {
					inputLoss_entityFormer_wordnet[idx] = inputLoss_entityFormer_wordnet[idx] * mask_entityFormer_wordnet[idx];
					_wordnet.EmbLoss(example.m_entityFormer_wordnet[idx], inputLoss_entityFormer_wordnet[idx]);
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					inputLoss_entityLatter_wordnet[idx] = inputLoss_entityLatter_wordnet[idx] * mask_entityLatter_wordnet[idx];
					_wordnet.EmbLoss(example.m_entityLatter_wordnet[idx], inputLoss_entityLatter_wordnet[idx]);
				}
			}

			// brown channel
			if (bBrown && _brown.bEmbFineTune()) {
				for (int idx = 0; idx < beforeSize; idx++) {
					inputLoss_before_brown[idx] = inputLoss_before_brown[idx] * mask_before_brown[idx];
					_brown.EmbLoss(example.m_before_brown[idx], inputLoss_before_brown[idx]);
				}
				for (int idx = 0; idx < middleSize; idx++) {
					inputLoss_middle_brown[idx] = inputLoss_middle_brown[idx] * mask_middle_brown[idx];
					_brown.EmbLoss(example.m_middle_brown[idx], inputLoss_middle_brown[idx]);
				}
				for (int idx = 0; idx < afterSize; idx++) {
					inputLoss_after_brown[idx] = inputLoss_after_brown[idx] * mask_after_brown[idx];
					_brown.EmbLoss(example.m_after_brown[idx], inputLoss_after_brown[idx]);
				}
				for (int idx = 0; idx < enFormerSize; idx++) {
					inputLoss_entityFormer_brown[idx] = inputLoss_entityFormer_brown[idx] * mask_entityFormer_brown[idx];
					_brown.EmbLoss(example.m_entityFormer_brown[idx], inputLoss_entityFormer_brown[idx]);
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					inputLoss_entityLatter_brown[idx] = inputLoss_entityLatter_brown[idx] * mask_entityLatter_brown[idx];
					_brown.EmbLoss(example.m_entityLatter_brown[idx], inputLoss_entityLatter_brown[idx]);
				}
			}

			// bigram channel
			if (bBigram && _bigram.bEmbFineTune()) {
				for (int idx = 0; idx < beforeSize; idx++) {
					inputLoss_before_bigram[idx] = inputLoss_before_bigram[idx] * mask_before_bigram[idx];
					_bigram.EmbLoss(example.m_before_bigram[idx], inputLoss_before_bigram[idx]);
				}
				for (int idx = 0; idx < middleSize; idx++) {
					inputLoss_middle_bigram[idx] = inputLoss_middle_bigram[idx] * mask_middle_bigram[idx];
					_bigram.EmbLoss(example.m_middle_bigram[idx], inputLoss_middle_bigram[idx]);
				}
				for (int idx = 0; idx < afterSize; idx++) {
					inputLoss_after_bigram[idx] = inputLoss_after_bigram[idx] * mask_after_bigram[idx];
					_bigram.EmbLoss(example.m_after_bigram[idx], inputLoss_after_bigram[idx]);
				}
				for (int idx = 0; idx < enFormerSize; idx++) {
					inputLoss_entityFormer_bigram[idx] = inputLoss_entityFormer_bigram[idx] * mask_entityFormer_bigram[idx];
					_bigram.EmbLoss(example.m_entityFormer_bigram[idx], inputLoss_entityFormer_bigram[idx]);
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					inputLoss_entityLatter_bigram[idx] = inputLoss_entityLatter_bigram[idx] * mask_entityLatter_bigram[idx];
					_bigram.EmbLoss(example.m_entityLatter_bigram[idx], inputLoss_entityLatter_bigram[idx]);
				}
			}

			// pos channel
			if (bPos && _pos.bEmbFineTune()) {
				for (int idx = 0; idx < beforeSize; idx++) {
					inputLoss_before_pos[idx] = inputLoss_before_pos[idx] * mask_before_pos[idx];
					_pos.EmbLoss(example.m_before_pos[idx], inputLoss_before_pos[idx]);
				}
				for (int idx = 0; idx < middleSize; idx++) {
					inputLoss_middle_pos[idx] = inputLoss_middle_pos[idx] * mask_middle_pos[idx];
					_pos.EmbLoss(example.m_middle_pos[idx], inputLoss_middle_pos[idx]);
				}
				for (int idx = 0; idx < afterSize; idx++) {
					inputLoss_after_pos[idx] = inputLoss_after_pos[idx] * mask_after_pos[idx];
					_pos.EmbLoss(example.m_after_pos[idx], inputLoss_after_pos[idx]);
				}
				for (int idx = 0; idx < enFormerSize; idx++) {
					inputLoss_entityFormer_pos[idx] = inputLoss_entityFormer_pos[idx] * mask_entityFormer_pos[idx];
					_pos.EmbLoss(example.m_entityFormer_pos[idx], inputLoss_entityFormer_pos[idx]);
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					inputLoss_entityLatter_pos[idx] = inputLoss_entityLatter_pos[idx] * mask_entityLatter_pos[idx];
					_pos.EmbLoss(example.m_entityLatter_pos[idx], inputLoss_entityLatter_pos[idx]);
				}
			}

			// sst channel
			if (bSst && _sst.bEmbFineTune()) {
				for (int idx = 0; idx < beforeSize; idx++) {
					inputLoss_before_sst[idx] = inputLoss_before_sst[idx] * mask_before_sst[idx];
					_sst.EmbLoss(example.m_before_sst[idx], inputLoss_before_sst[idx]);
				}
				for (int idx = 0; idx < middleSize; idx++) {
					inputLoss_middle_sst[idx] = inputLoss_middle_sst[idx] * mask_middle_sst[idx];
					_sst.EmbLoss(example.m_middle_sst[idx], inputLoss_middle_sst[idx]);
				}
				for (int idx = 0; idx < afterSize; idx++) {
					inputLoss_after_sst[idx] = inputLoss_after_sst[idx] * mask_after_sst[idx];
					_sst.EmbLoss(example.m_after_sst[idx], inputLoss_after_sst[idx]);
				}
				for (int idx = 0; idx < enFormerSize; idx++) {
					inputLoss_entityFormer_sst[idx] = inputLoss_entityFormer_sst[idx] * mask_entityFormer_sst[idx];
					_sst.EmbLoss(example.m_entityFormer_sst[idx], inputLoss_entityFormer_sst[idx]);
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					inputLoss_entityLatter_sst[idx] = inputLoss_entityLatter_sst[idx] * mask_entityLatter_sst[idx];
					_sst.EmbLoss(example.m_entityLatter_sst[idx], inputLoss_entityLatter_sst[idx]);
				}
			}

			// word channel
			if(bWord) {
				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						FreeSpace(&(xMExp_before[idx]));
						FreeSpace(&(xExp_before[idx]));
						FreeSpace(&(xPoolIndex_before[idx]));
					}
					FreeSpace(&(xSum_before));
					FreeSpace(&(y_att_before));
					FreeSpace(&(ly_att_before));

					for (int idx = 0; idx < middleSize; idx++) {
						FreeSpace(&(xMExp_middle[idx]));
						FreeSpace(&(xExp_middle[idx]));
						FreeSpace(&(xPoolIndex_middle[idx]));
					}
					FreeSpace(&(xSum_middle));
					FreeSpace(&(y_att_middle));
					FreeSpace(&(ly_att_middle));

					for (int idx = 0; idx < afterSize; idx++) {
						FreeSpace(&(xMExp_after[idx]));
						FreeSpace(&(xExp_after[idx]));
						FreeSpace(&(xPoolIndex_after[idx]));
					}
					FreeSpace(&(xSum_after));
					FreeSpace(&(y_att_after));
					FreeSpace(&(ly_att_after));
				}
				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(input_before[idx]));
					FreeSpace(&(mask_before[idx]));
					FreeSpace(&(inputLoss_before[idx]));
					FreeSpace(&(iy_before[idx]));
					FreeSpace(&(oy_before[idx]));
					FreeSpace(&(fy_before[idx]));
					FreeSpace(&(mcy_before[idx]));
					FreeSpace(&(cy_before[idx]));
					FreeSpace(&(my_before[idx]));
					FreeSpace(&(y_before[idx]));
					FreeSpace(&(loss_before[idx]));
				}
				for (int idx = 0; idx < enFormerSize; idx++) {
					FreeSpace(&(input_entityFormer[idx]));
					FreeSpace(&(mask_entityFormer[idx]));
					FreeSpace(&(inputLoss_entityFormer[idx]));
					FreeSpace(&(iy_entityFormer[idx]));
					FreeSpace(&(oy_entityFormer[idx]));
					FreeSpace(&(fy_entityFormer[idx]));
					FreeSpace(&(mcy_entityFormer[idx]));
					FreeSpace(&(cy_entityFormer[idx]));
					FreeSpace(&(my_entityFormer[idx]));
					FreeSpace(&(y_entityFormer[idx]));
					FreeSpace(&(loss_entityFormer[idx]));
				}

				for (int idx = 0; idx < enLatterSize; idx++) {
					FreeSpace(&(input_entityLatter[idx]));
					FreeSpace(&(mask_entityLatter[idx]));
					FreeSpace(&(inputLoss_entityLatter[idx]));
					FreeSpace(&(iy_entityLatter[idx]));
					FreeSpace(&(oy_entityLatter[idx]));
					FreeSpace(&(fy_entityLatter[idx]));
					FreeSpace(&(mcy_entityLatter[idx]));
					FreeSpace(&(cy_entityLatter[idx]));
					FreeSpace(&(my_entityLatter[idx]));
					FreeSpace(&(y_entityLatter[idx]));
					FreeSpace(&(loss_entityLatter[idx]));
				}

				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(input_middle[idx]));
					FreeSpace(&(mask_middle[idx]));
					FreeSpace(&(inputLoss_middle[idx]));
					FreeSpace(&(iy_middle[idx]));
					FreeSpace(&(oy_middle[idx]));
					FreeSpace(&(fy_middle[idx]));
					FreeSpace(&(mcy_middle[idx]));
					FreeSpace(&(cy_middle[idx]));
					FreeSpace(&(my_middle[idx]));
					FreeSpace(&(y_middle[idx]));
					FreeSpace(&(loss_middle[idx]));
				}

				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(input_after[idx]));
					FreeSpace(&(mask_after[idx]));
					FreeSpace(&(inputLoss_after[idx]));
					FreeSpace(&(iy_after[idx]));
					FreeSpace(&(oy_after[idx]));
					FreeSpace(&(fy_after[idx]));
					FreeSpace(&(mcy_after[idx]));
					FreeSpace(&(cy_after[idx]));
					FreeSpace(&(my_after[idx]));
					FreeSpace(&(y_after[idx]));
					FreeSpace(&(loss_after[idx]));
				}
			}



			// wordnet channel
			if(bWordnet) {
				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						FreeSpace(&(xMExp_before_wordnet[idx]));
						FreeSpace(&(xExp_before_wordnet[idx]));
						FreeSpace(&(xPoolIndex_before_wordnet[idx]));
					}
					FreeSpace(&(xSum_before_wordnet));
					FreeSpace(&(y_att_before_wordnet));
					FreeSpace(&(ly_att_before_wordnet));

					for (int idx = 0; idx < middleSize; idx++) {
						FreeSpace(&(xMExp_middle_wordnet[idx]));
						FreeSpace(&(xExp_middle_wordnet[idx]));
						FreeSpace(&(xPoolIndex_middle_wordnet[idx]));
					}
					FreeSpace(&(xSum_middle_wordnet));
					FreeSpace(&(y_att_middle_wordnet));
					FreeSpace(&(ly_att_middle_wordnet));

					for (int idx = 0; idx < afterSize; idx++) {
						FreeSpace(&(xMExp_after_wordnet[idx]));
						FreeSpace(&(xExp_after_wordnet[idx]));
						FreeSpace(&(xPoolIndex_after_wordnet[idx]));
					}
					FreeSpace(&(xSum_after_wordnet));
					FreeSpace(&(y_att_after_wordnet));
					FreeSpace(&(ly_att_after_wordnet));
				}

				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(input_before_wordnet[idx]));
					FreeSpace(&(mask_before_wordnet[idx]));
					FreeSpace(&(inputLoss_before_wordnet[idx]));
					FreeSpace(&(iy_before_wordnet[idx]));
					FreeSpace(&(oy_before_wordnet[idx]));
					FreeSpace(&(fy_before_wordnet[idx]));
					FreeSpace(&(mcy_before_wordnet[idx]));
					FreeSpace(&(cy_before_wordnet[idx]));
					FreeSpace(&(my_before_wordnet[idx]));
					FreeSpace(&(y_before_wordnet[idx]));
					FreeSpace(&(loss_before_wordnet[idx]));
				}
				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(input_middle_wordnet[idx]));
					FreeSpace(&(mask_middle_wordnet[idx]));
					FreeSpace(&(inputLoss_middle_wordnet[idx]));
					FreeSpace(&(iy_middle_wordnet[idx]));
					FreeSpace(&(oy_middle_wordnet[idx]));
					FreeSpace(&(fy_middle_wordnet[idx]));
					FreeSpace(&(mcy_middle_wordnet[idx]));
					FreeSpace(&(cy_middle_wordnet[idx]));
					FreeSpace(&(my_middle_wordnet[idx]));
					FreeSpace(&(y_middle_wordnet[idx]));
					FreeSpace(&(loss_middle_wordnet[idx]));
				}
				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(input_after_wordnet[idx]));
					FreeSpace(&(mask_after_wordnet[idx]));
					FreeSpace(&(inputLoss_after_wordnet[idx]));
					FreeSpace(&(iy_after_wordnet[idx]));
					FreeSpace(&(oy_after_wordnet[idx]));
					FreeSpace(&(fy_after_wordnet[idx]));
					FreeSpace(&(mcy_after_wordnet[idx]));
					FreeSpace(&(cy_after_wordnet[idx]));
					FreeSpace(&(my_after_wordnet[idx]));
					FreeSpace(&(y_after_wordnet[idx]));
					FreeSpace(&(loss_after_wordnet[idx]));
				}

				for (int idx = 0; idx < enFormerSize; idx++) {
					FreeSpace(&(input_entityFormer_wordnet[idx]));
					FreeSpace(&(mask_entityFormer_wordnet[idx]));
					FreeSpace(&(inputLoss_entityFormer_wordnet[idx]));
					FreeSpace(&(iy_entityFormer_wordnet[idx]));
					FreeSpace(&(oy_entityFormer_wordnet[idx]));
					FreeSpace(&(fy_entityFormer_wordnet[idx]));
					FreeSpace(&(mcy_entityFormer_wordnet[idx]));
					FreeSpace(&(cy_entityFormer_wordnet[idx]));
					FreeSpace(&(my_entityFormer_wordnet[idx]));
					FreeSpace(&(y_entityFormer_wordnet[idx]));
					FreeSpace(&(loss_entityFormer_wordnet[idx]));
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					FreeSpace(&(input_entityLatter_wordnet[idx]));
					FreeSpace(&(mask_entityLatter_wordnet[idx]));
					FreeSpace(&(inputLoss_entityLatter_wordnet[idx]));
					FreeSpace(&(iy_entityLatter_wordnet[idx]));
					FreeSpace(&(oy_entityLatter_wordnet[idx]));
					FreeSpace(&(fy_entityLatter_wordnet[idx]));
					FreeSpace(&(mcy_entityLatter_wordnet[idx]));
					FreeSpace(&(cy_entityLatter_wordnet[idx]));
					FreeSpace(&(my_entityLatter_wordnet[idx]));
					FreeSpace(&(y_entityLatter_wordnet[idx]));
					FreeSpace(&(loss_entityLatter_wordnet[idx]));
				}

			}


			// brown channel
			if(bBrown) {
				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						FreeSpace(&(xMExp_before_brown[idx]));
						FreeSpace(&(xExp_before_brown[idx]));
						FreeSpace(&(xPoolIndex_before_brown[idx]));
					}
					FreeSpace(&(xSum_before_brown));
					FreeSpace(&(y_att_before_brown));
					FreeSpace(&(ly_att_before_brown));

					for (int idx = 0; idx < middleSize; idx++) {
						FreeSpace(&(xMExp_middle_brown[idx]));
						FreeSpace(&(xExp_middle_brown[idx]));
						FreeSpace(&(xPoolIndex_middle_brown[idx]));
					}
					FreeSpace(&(xSum_middle_brown));
					FreeSpace(&(y_att_middle_brown));
					FreeSpace(&(ly_att_middle_brown));

					for (int idx = 0; idx < afterSize; idx++) {
						FreeSpace(&(xMExp_after_brown[idx]));
						FreeSpace(&(xExp_after_brown[idx]));
						FreeSpace(&(xPoolIndex_after_brown[idx]));
					}
					FreeSpace(&(xSum_after_brown));
					FreeSpace(&(y_att_after_brown));
					FreeSpace(&(ly_att_after_brown));
				}

				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(input_before_brown[idx]));
					FreeSpace(&(mask_before_brown[idx]));
					FreeSpace(&(inputLoss_before_brown[idx]));
					FreeSpace(&(iy_before_brown[idx]));
					FreeSpace(&(oy_before_brown[idx]));
					FreeSpace(&(fy_before_brown[idx]));
					FreeSpace(&(mcy_before_brown[idx]));
					FreeSpace(&(cy_before_brown[idx]));
					FreeSpace(&(my_before_brown[idx]));
					FreeSpace(&(y_before_brown[idx]));
					FreeSpace(&(loss_before_brown[idx]));
				}
				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(input_middle_brown[idx]));
					FreeSpace(&(mask_middle_brown[idx]));
					FreeSpace(&(inputLoss_middle_brown[idx]));
					FreeSpace(&(iy_middle_brown[idx]));
					FreeSpace(&(oy_middle_brown[idx]));
					FreeSpace(&(fy_middle_brown[idx]));
					FreeSpace(&(mcy_middle_brown[idx]));
					FreeSpace(&(cy_middle_brown[idx]));
					FreeSpace(&(my_middle_brown[idx]));
					FreeSpace(&(y_middle_brown[idx]));
					FreeSpace(&(loss_middle_brown[idx]));
				}
				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(input_after_brown[idx]));
					FreeSpace(&(mask_after_brown[idx]));
					FreeSpace(&(inputLoss_after_brown[idx]));
					FreeSpace(&(iy_after_brown[idx]));
					FreeSpace(&(oy_after_brown[idx]));
					FreeSpace(&(fy_after_brown[idx]));
					FreeSpace(&(mcy_after_brown[idx]));
					FreeSpace(&(cy_after_brown[idx]));
					FreeSpace(&(my_after_brown[idx]));
					FreeSpace(&(y_after_brown[idx]));
					FreeSpace(&(loss_after_brown[idx]));
				}

				for (int idx = 0; idx < enFormerSize; idx++) {
					FreeSpace(&(input_entityFormer_brown[idx]));
					FreeSpace(&(mask_entityFormer_brown[idx]));
					FreeSpace(&(inputLoss_entityFormer_brown[idx]));
					FreeSpace(&(iy_entityFormer_brown[idx]));
					FreeSpace(&(oy_entityFormer_brown[idx]));
					FreeSpace(&(fy_entityFormer_brown[idx]));
					FreeSpace(&(mcy_entityFormer_brown[idx]));
					FreeSpace(&(cy_entityFormer_brown[idx]));
					FreeSpace(&(my_entityFormer_brown[idx]));
					FreeSpace(&(y_entityFormer_brown[idx]));
					FreeSpace(&(loss_entityFormer_brown[idx]));
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					FreeSpace(&(input_entityLatter_brown[idx]));
					FreeSpace(&(mask_entityLatter_brown[idx]));
					FreeSpace(&(inputLoss_entityLatter_brown[idx]));
					FreeSpace(&(iy_entityLatter_brown[idx]));
					FreeSpace(&(oy_entityLatter_brown[idx]));
					FreeSpace(&(fy_entityLatter_brown[idx]));
					FreeSpace(&(mcy_entityLatter_brown[idx]));
					FreeSpace(&(cy_entityLatter_brown[idx]));
					FreeSpace(&(my_entityLatter_brown[idx]));
					FreeSpace(&(y_entityLatter_brown[idx]));
					FreeSpace(&(loss_entityLatter_brown[idx]));
				}
			}



			// bigram channel
			if(bBigram) {
				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						FreeSpace(&(xMExp_before_bigram[idx]));
						FreeSpace(&(xExp_before_bigram[idx]));
						FreeSpace(&(xPoolIndex_before_bigram[idx]));
					}
					FreeSpace(&(xSum_before_bigram));
					FreeSpace(&(y_att_before_bigram));
					FreeSpace(&(ly_att_before_bigram));

					for (int idx = 0; idx < middleSize; idx++) {
						FreeSpace(&(xMExp_middle_bigram[idx]));
						FreeSpace(&(xExp_middle_bigram[idx]));
						FreeSpace(&(xPoolIndex_middle_bigram[idx]));
					}
					FreeSpace(&(xSum_middle_bigram));
					FreeSpace(&(y_att_middle_bigram));
					FreeSpace(&(ly_att_middle_bigram));

					for (int idx = 0; idx < afterSize; idx++) {
						FreeSpace(&(xMExp_after_bigram[idx]));
						FreeSpace(&(xExp_after_bigram[idx]));
						FreeSpace(&(xPoolIndex_after_bigram[idx]));
					}
					FreeSpace(&(xSum_after_bigram));
					FreeSpace(&(y_att_after_bigram));
					FreeSpace(&(ly_att_after_bigram));
				}

				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(input_before_bigram[idx]));
					FreeSpace(&(mask_before_bigram[idx]));
					FreeSpace(&(inputLoss_before_bigram[idx]));
					FreeSpace(&(iy_before_bigram[idx]));
					FreeSpace(&(oy_before_bigram[idx]));
					FreeSpace(&(fy_before_bigram[idx]));
					FreeSpace(&(mcy_before_bigram[idx]));
					FreeSpace(&(cy_before_bigram[idx]));
					FreeSpace(&(my_before_bigram[idx]));
					FreeSpace(&(y_before_bigram[idx]));
					FreeSpace(&(loss_before_bigram[idx]));
				}
				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(input_middle_bigram[idx]));
					FreeSpace(&(mask_middle_bigram[idx]));
					FreeSpace(&(inputLoss_middle_bigram[idx]));
					FreeSpace(&(iy_middle_bigram[idx]));
					FreeSpace(&(oy_middle_bigram[idx]));
					FreeSpace(&(fy_middle_bigram[idx]));
					FreeSpace(&(mcy_middle_bigram[idx]));
					FreeSpace(&(cy_middle_bigram[idx]));
					FreeSpace(&(my_middle_bigram[idx]));
					FreeSpace(&(y_middle_bigram[idx]));
					FreeSpace(&(loss_middle_bigram[idx]));
				}
				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(input_after_bigram[idx]));
					FreeSpace(&(mask_after_bigram[idx]));
					FreeSpace(&(inputLoss_after_bigram[idx]));
					FreeSpace(&(iy_after_bigram[idx]));
					FreeSpace(&(oy_after_bigram[idx]));
					FreeSpace(&(fy_after_bigram[idx]));
					FreeSpace(&(mcy_after_bigram[idx]));
					FreeSpace(&(cy_after_bigram[idx]));
					FreeSpace(&(my_after_bigram[idx]));
					FreeSpace(&(y_after_bigram[idx]));
					FreeSpace(&(loss_after_bigram[idx]));
				}

				for (int idx = 0; idx < enFormerSize; idx++) {
					FreeSpace(&(input_entityFormer_bigram[idx]));
					FreeSpace(&(mask_entityFormer_bigram[idx]));
					FreeSpace(&(inputLoss_entityFormer_bigram[idx]));
					FreeSpace(&(iy_entityFormer_bigram[idx]));
					FreeSpace(&(oy_entityFormer_bigram[idx]));
					FreeSpace(&(fy_entityFormer_bigram[idx]));
					FreeSpace(&(mcy_entityFormer_bigram[idx]));
					FreeSpace(&(cy_entityFormer_bigram[idx]));
					FreeSpace(&(my_entityFormer_bigram[idx]));
					FreeSpace(&(y_entityFormer_bigram[idx]));
					FreeSpace(&(loss_entityFormer_bigram[idx]));
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					FreeSpace(&(input_entityLatter_bigram[idx]));
					FreeSpace(&(mask_entityLatter_bigram[idx]));
					FreeSpace(&(inputLoss_entityLatter_bigram[idx]));
					FreeSpace(&(iy_entityLatter_bigram[idx]));
					FreeSpace(&(oy_entityLatter_bigram[idx]));
					FreeSpace(&(fy_entityLatter_bigram[idx]));
					FreeSpace(&(mcy_entityLatter_bigram[idx]));
					FreeSpace(&(cy_entityLatter_bigram[idx]));
					FreeSpace(&(my_entityLatter_bigram[idx]));
					FreeSpace(&(y_entityLatter_bigram[idx]));
					FreeSpace(&(loss_entityLatter_bigram[idx]));
				}
			}


			// pos channel
			if(bPos) {
				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						FreeSpace(&(xMExp_before_pos[idx]));
						FreeSpace(&(xExp_before_pos[idx]));
						FreeSpace(&(xPoolIndex_before_pos[idx]));
					}
					FreeSpace(&(xSum_before_pos));
					FreeSpace(&(y_att_before_pos));
					FreeSpace(&(ly_att_before_pos));

					for (int idx = 0; idx < middleSize; idx++) {
						FreeSpace(&(xMExp_middle_pos[idx]));
						FreeSpace(&(xExp_middle_pos[idx]));
						FreeSpace(&(xPoolIndex_middle_pos[idx]));
					}
					FreeSpace(&(xSum_middle_pos));
					FreeSpace(&(y_att_middle_pos));
					FreeSpace(&(ly_att_middle_pos));

					for (int idx = 0; idx < afterSize; idx++) {
						FreeSpace(&(xMExp_after_pos[idx]));
						FreeSpace(&(xExp_after_pos[idx]));
						FreeSpace(&(xPoolIndex_after_pos[idx]));
					}
					FreeSpace(&(xSum_after_pos));
					FreeSpace(&(y_att_after_pos));
					FreeSpace(&(ly_att_after_pos));
				}

				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(input_before_pos[idx]));
					FreeSpace(&(mask_before_pos[idx]));
					FreeSpace(&(inputLoss_before_pos[idx]));
					FreeSpace(&(iy_before_pos[idx]));
					FreeSpace(&(oy_before_pos[idx]));
					FreeSpace(&(fy_before_pos[idx]));
					FreeSpace(&(mcy_before_pos[idx]));
					FreeSpace(&(cy_before_pos[idx]));
					FreeSpace(&(my_before_pos[idx]));
					FreeSpace(&(y_before_pos[idx]));
					FreeSpace(&(loss_before_pos[idx]));
				}
				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(input_middle_pos[idx]));
					FreeSpace(&(mask_middle_pos[idx]));
					FreeSpace(&(inputLoss_middle_pos[idx]));
					FreeSpace(&(iy_middle_pos[idx]));
					FreeSpace(&(oy_middle_pos[idx]));
					FreeSpace(&(fy_middle_pos[idx]));
					FreeSpace(&(mcy_middle_pos[idx]));
					FreeSpace(&(cy_middle_pos[idx]));
					FreeSpace(&(my_middle_pos[idx]));
					FreeSpace(&(y_middle_pos[idx]));
					FreeSpace(&(loss_middle_pos[idx]));
				}
				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(input_after_pos[idx]));
					FreeSpace(&(mask_after_pos[idx]));
					FreeSpace(&(inputLoss_after_pos[idx]));
					FreeSpace(&(iy_after_pos[idx]));
					FreeSpace(&(oy_after_pos[idx]));
					FreeSpace(&(fy_after_pos[idx]));
					FreeSpace(&(mcy_after_pos[idx]));
					FreeSpace(&(cy_after_pos[idx]));
					FreeSpace(&(my_after_pos[idx]));
					FreeSpace(&(y_after_pos[idx]));
					FreeSpace(&(loss_after_pos[idx]));
				}

				for (int idx = 0; idx < enFormerSize; idx++) {
					FreeSpace(&(input_entityFormer_pos[idx]));
					FreeSpace(&(mask_entityFormer_pos[idx]));
					FreeSpace(&(inputLoss_entityFormer_pos[idx]));
					FreeSpace(&(iy_entityFormer_pos[idx]));
					FreeSpace(&(oy_entityFormer_pos[idx]));
					FreeSpace(&(fy_entityFormer_pos[idx]));
					FreeSpace(&(mcy_entityFormer_pos[idx]));
					FreeSpace(&(cy_entityFormer_pos[idx]));
					FreeSpace(&(my_entityFormer_pos[idx]));
					FreeSpace(&(y_entityFormer_pos[idx]));
					FreeSpace(&(loss_entityFormer_pos[idx]));
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					FreeSpace(&(input_entityLatter_pos[idx]));
					FreeSpace(&(mask_entityLatter_pos[idx]));
					FreeSpace(&(inputLoss_entityLatter_pos[idx]));
					FreeSpace(&(iy_entityLatter_pos[idx]));
					FreeSpace(&(oy_entityLatter_pos[idx]));
					FreeSpace(&(fy_entityLatter_pos[idx]));
					FreeSpace(&(mcy_entityLatter_pos[idx]));
					FreeSpace(&(cy_entityLatter_pos[idx]));
					FreeSpace(&(my_entityLatter_pos[idx]));
					FreeSpace(&(y_entityLatter_pos[idx]));
					FreeSpace(&(loss_entityLatter_pos[idx]));
				}
			}


			// sst channel
			if(bSst) {
				if(options.attention) {
					for (int idx = 0; idx < beforeSize; idx++) {
						FreeSpace(&(xMExp_before_sst[idx]));
						FreeSpace(&(xExp_before_sst[idx]));
						FreeSpace(&(xPoolIndex_before_sst[idx]));
					}
					FreeSpace(&(xSum_before_sst));
					FreeSpace(&(y_att_before_sst));
					FreeSpace(&(ly_att_before_sst));

					for (int idx = 0; idx < middleSize; idx++) {
						FreeSpace(&(xMExp_middle_sst[idx]));
						FreeSpace(&(xExp_middle_sst[idx]));
						FreeSpace(&(xPoolIndex_middle_sst[idx]));
					}
					FreeSpace(&(xSum_middle_sst));
					FreeSpace(&(y_att_middle_sst));
					FreeSpace(&(ly_att_middle_sst));

					for (int idx = 0; idx < afterSize; idx++) {
						FreeSpace(&(xMExp_after_sst[idx]));
						FreeSpace(&(xExp_after_sst[idx]));
						FreeSpace(&(xPoolIndex_after_sst[idx]));
					}
					FreeSpace(&(xSum_after_sst));
					FreeSpace(&(y_att_after_sst));
					FreeSpace(&(ly_att_after_sst));
				}

				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(input_before_sst[idx]));
					FreeSpace(&(mask_before_sst[idx]));
					FreeSpace(&(inputLoss_before_sst[idx]));
					FreeSpace(&(iy_before_sst[idx]));
					FreeSpace(&(oy_before_sst[idx]));
					FreeSpace(&(fy_before_sst[idx]));
					FreeSpace(&(mcy_before_sst[idx]));
					FreeSpace(&(cy_before_sst[idx]));
					FreeSpace(&(my_before_sst[idx]));
					FreeSpace(&(y_before_sst[idx]));
					FreeSpace(&(loss_before_sst[idx]));
				}
				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(input_middle_sst[idx]));
					FreeSpace(&(mask_middle_sst[idx]));
					FreeSpace(&(inputLoss_middle_sst[idx]));
					FreeSpace(&(iy_middle_sst[idx]));
					FreeSpace(&(oy_middle_sst[idx]));
					FreeSpace(&(fy_middle_sst[idx]));
					FreeSpace(&(mcy_middle_sst[idx]));
					FreeSpace(&(cy_middle_sst[idx]));
					FreeSpace(&(my_middle_sst[idx]));
					FreeSpace(&(y_middle_sst[idx]));
					FreeSpace(&(loss_middle_sst[idx]));
				}
				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(input_after_sst[idx]));
					FreeSpace(&(mask_after_sst[idx]));
					FreeSpace(&(inputLoss_after_sst[idx]));
					FreeSpace(&(iy_after_sst[idx]));
					FreeSpace(&(oy_after_sst[idx]));
					FreeSpace(&(fy_after_sst[idx]));
					FreeSpace(&(mcy_after_sst[idx]));
					FreeSpace(&(cy_after_sst[idx]));
					FreeSpace(&(my_after_sst[idx]));
					FreeSpace(&(y_after_sst[idx]));
					FreeSpace(&(loss_after_sst[idx]));
				}

				for (int idx = 0; idx < enFormerSize; idx++) {
					FreeSpace(&(input_entityFormer_sst[idx]));
					FreeSpace(&(mask_entityFormer_sst[idx]));
					FreeSpace(&(inputLoss_entityFormer_sst[idx]));
					FreeSpace(&(iy_entityFormer_sst[idx]));
					FreeSpace(&(oy_entityFormer_sst[idx]));
					FreeSpace(&(fy_entityFormer_sst[idx]));
					FreeSpace(&(mcy_entityFormer_sst[idx]));
					FreeSpace(&(cy_entityFormer_sst[idx]));
					FreeSpace(&(my_entityFormer_sst[idx]));
					FreeSpace(&(y_entityFormer_sst[idx]));
					FreeSpace(&(loss_entityFormer_sst[idx]));
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					FreeSpace(&(input_entityLatter_sst[idx]));
					FreeSpace(&(mask_entityLatter_sst[idx]));
					FreeSpace(&(inputLoss_entityLatter_sst[idx]));
					FreeSpace(&(iy_entityLatter_sst[idx]));
					FreeSpace(&(oy_entityLatter_sst[idx]));
					FreeSpace(&(fy_entityLatter_sst[idx]));
					FreeSpace(&(mcy_entityLatter_sst[idx]));
					FreeSpace(&(cy_entityLatter_sst[idx]));
					FreeSpace(&(my_entityLatter_sst[idx]));
					FreeSpace(&(y_entityLatter_sst[idx]));
					FreeSpace(&(loss_entityLatter_sst[idx]));
				}
			}


			FreeSpace(&hidden_input);
			FreeSpace(&hidden_input_loss);

			FreeSpace(&hidden);
			FreeSpace(&hiddenLoss);

			FreeSpace(&output);
			FreeSpace(&outputLoss);

		} // end for example_num


		return cost;
	}

	void updateParams() {
		if(bWord) {
			unit_entityFormer.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_entityLatter.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_before.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_middle.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_after.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			if(options.attention) {
				unit_att_before.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_middle.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_after.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			}
			_words.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		}

		if(bWordnet) {
			unit_entityFormer_wordnet.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_entityLatter_wordnet.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_before_wordnet.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_middle_wordnet.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_after_wordnet.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			if(options.attention) {
				unit_att_before_wordnet.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_middle_wordnet.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_after_wordnet.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			}
			_wordnet.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		}


		if(bBrown) {
			unit_entityFormer_brown.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_entityLatter_brown.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_before_brown.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_middle_brown.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_after_brown.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			if(options.attention) {
				unit_att_before_brown.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_middle_brown.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_after_brown.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			}
			_brown.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		}

		if(bBigram) {
			unit_entityFormer_bigram.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_entityLatter_bigram.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_before_bigram.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_middle_bigram.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_after_bigram.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			if(options.attention) {
				unit_att_before_bigram.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_middle_bigram.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_after_bigram.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			}
			_bigram.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		}

		if(bPos) {
			unit_entityFormer_pos.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_entityLatter_pos.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_before_pos.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_middle_pos.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_after_pos.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			if(options.attention) {
				unit_att_before_pos.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_middle_pos.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_after_pos.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			}
			_pos.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		}

		if(bSst) {
			unit_entityFormer_sst.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_entityLatter_sst.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_before_sst.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_middle_sst.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_after_sst.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			if(options.attention) {
				unit_att_before_sst.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_middle_sst.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
				unit_att_after_sst.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			}
			_sst.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		}


		hidden_layer.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);

		output_layer.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);

	}

};



#endif /* CLASSIFIER_LSTM_H_ */
