/*
 * Classifier.h
 *
 *  Created on: Dec 28, 2015
 *      Author: fox
 */

#ifndef CLASSIFIER_DEP_H_
#define CLASSIFIER_DEP_H_

#include <iostream>

#include <assert.h>

#include "Example.h"
#include "Metric.h"
#include "N3L.h"
#include "Options.h"
#include "utils.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class Classifier_dep {
public:
	Options options;


	int _wordDim;
	int _outputSize;
	int _hidden_input_size;

	// input
	// word channel
	LookupTable<xpu> _words;
	LSTM<xpu> unit_before;
	LSTM<xpu> unit_after;



	// hidden
	UniLayer<xpu> hidden_layer;

	// output
	UniLayer<xpu> output_layer;
	// softmax loss corresponds no class in n3l

	Metric _eval;

	bool bWord = false;


	Classifier_dep(const Options& options):options(options) {

	}


	void release() {

		output_layer.release();

		hidden_layer.release();


		if(bWord) {
			unit_before.release();
			unit_after.release();
			_words.release();
		}

	}


	void init(int outputsize, const NRMat<dtype>& wordEmb, const NRMat<dtype>& wordnetEmb, const NRMat<dtype>& brownEmb,
			const NRMat<dtype>& bigramEmb, const NRMat<dtype>& posEmb, const NRMat<dtype>& sstEmb,
			int sparseFeatureSize) {
		int model = options.channelMode;
		if((model & 1) == 1)
			bWord = true;

		assert(bWord==true);

	    _wordDim = wordEmb.ncols();
	    _outputSize = outputsize;



	    if(bWord) {
			_words.initial(wordEmb);
			_words.setEmbFineTune(options.wordEmbFineTune);
			unit_before.initial(options.context_embsize, _wordDim, 10);
			unit_after.initial(options.context_embsize, _wordDim, 20);
	    }

	    _hidden_input_size = 0;
	    int lstmSize = options.context_embsize*2;

	    if(bWord) {
	    	_hidden_input_size += lstmSize;
	    }


	    hidden_layer.initial(options.hiddenSize, _hidden_input_size, true, 30, 0);

	    output_layer.initial(_outputSize, options.hiddenSize, true, 40, 2);

	    cout<<"classifier dep initial"<<endl;
	}

	void predict(const Example& example, vector<double>& scores) {
		int beforeSize = example.m_before.size();
		int afterSize = example.m_after.size();


		// word channel
		vector<Tensor<xpu, 2, dtype> > input_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > input_after(afterSize);
		if(bWord) {
			for (int idx = 0; idx < beforeSize; idx++) {
				input_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_words.GetEmb(example.m_before[idx], input_before[idx]);
			}
			for (int idx = 0; idx < afterSize; idx++) {
				input_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				_words.GetEmb(example.m_after[idx], input_after[idx]);
			}
		}


		Tensor<xpu, 2, dtype> hidden = NewTensor<xpu>(Shape2(1, options.hiddenSize), d_zero);
		Tensor<xpu, 2, dtype> output = NewTensor<xpu>(Shape2(1, _outputSize), d_zero);

		// word channel
		vector<Tensor<xpu, 2, dtype> > iy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > oy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > fy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > mcy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > cy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > my_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > y_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > iy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > oy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > fy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > cy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > my_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > y_after(afterSize);

		if(bWord) {
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

		}


		vector<Tensor<xpu, 2, dtype> > v_hidden_input;

		// word channel
		Tensor<xpu, 2, dtype> y_before_pool;
		vector<Tensor<xpu, 2, dtype> > y_before_poolIndex(beforeSize);
		Tensor<xpu, 2, dtype> y_after_pool;
		vector<Tensor<xpu, 2, dtype> > y_after_poolIndex(afterSize);
		if(bWord) {
			y_before_pool = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			for (int idx = 0; idx < beforeSize; idx++) {
				y_before_poolIndex[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}
			y_after_pool = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			for (int idx = 0; idx < afterSize; idx++) {
				y_after_poolIndex[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			}

			maxpool_forward(y_before, y_before_pool, y_before_poolIndex);
			maxpool_forward(y_after, y_after_pool, y_after_poolIndex);

			v_hidden_input.push_back(y_before_pool);
			v_hidden_input.push_back(y_after_pool);
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

			FreeSpace(&(y_before_pool));
			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(y_before_poolIndex[idx]));
			}
			FreeSpace(&(y_after_pool));
			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(y_after_poolIndex[idx]));
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
			int beforeSize = example.m_before.size();
			int afterSize = example.m_after.size();

			// word channel
			vector<Tensor<xpu, 2, dtype> > input_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mask_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > input_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > mask_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_after(afterSize);
			if(bWord) {

				for (int idx = 0; idx < beforeSize; idx++) {
					input_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					mask_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					inputLoss_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					srand(iter * example_num + count + idx);
					_words.GetEmb(example.m_before[idx], input_before[idx]);
					dropoutcol(mask_before[idx], options.dropProb);
					input_before[idx] = input_before[idx] * mask_before[idx];
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



			Tensor<xpu, 2, dtype> hidden = NewTensor<xpu>(Shape2(1, options.hiddenSize), d_zero);
			Tensor<xpu, 2, dtype> hiddenLoss = NewTensor<xpu>(Shape2(1, options.hiddenSize), d_zero);

			Tensor<xpu, 2, dtype> output = NewTensor<xpu>(Shape2(1, _outputSize), d_zero);


			// word channel
			vector<Tensor<xpu, 2, dtype> > iy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > oy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > fy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mcy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > cy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > my_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > y_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > loss_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > iy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > oy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > fy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > cy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > my_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > y_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > loss_after(afterSize);

			if(bWord) {
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

			}


			vector<Tensor<xpu, 2, dtype> > v_hidden_input;

			// word channel
			Tensor<xpu, 2, dtype> y_before_pool;
			vector<Tensor<xpu, 2, dtype> > y_before_poolIndex(beforeSize);
			Tensor<xpu, 2, dtype> oly_before_pool;
			Tensor<xpu, 2, dtype> y_after_pool;
			vector<Tensor<xpu, 2, dtype> > y_after_poolIndex(afterSize);
			Tensor<xpu, 2, dtype> oly_after_pool;
			if(bWord) {
				y_before_pool = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oly_before_pool = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				for (int idx = 0; idx < beforeSize; idx++) {
					y_before_poolIndex[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}
				y_after_pool = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oly_after_pool = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				for (int idx = 0; idx < afterSize; idx++) {
					y_after_poolIndex[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				}

				maxpool_forward(y_before, y_before_pool, y_before_poolIndex);
				maxpool_forward(y_after, y_after_pool, y_after_poolIndex);

				v_hidden_input.push_back(y_before_pool);
				v_hidden_input.push_back(y_after_pool);
			}


			Tensor<xpu, 2, dtype> hidden_input = NewTensor<xpu>(Shape2(1, _hidden_input_size), d_zero);
			concat(v_hidden_input, hidden_input);


			hidden_layer.ComputeForwardScore(hidden_input, hidden);

			// hidden -> output
			output_layer.ComputeForwardScore(hidden, output);

			// combine output
			Tensor<xpu, 2, dtype> combine_output = NewTensor<xpu>(Shape2(1, _outputSize), d_zero);
			Tensor<xpu, 2, dtype> combine_output_loss = NewTensor<xpu>(Shape2(1, _outputSize), d_zero);
			for(int idx=0;idx<combine_output.size(1);idx++) {
				combine_output[0][idx] = output[0][idx];

			}

			// get delta for each output
			cost += softmax_loss(combine_output, example.m_labels, combine_output_loss, _eval, example_num);

			// loss backward propagation
			// output
			output_layer.ComputeBackwardLoss(hidden, output, combine_output_loss, hiddenLoss);

			// hidden
			Tensor<xpu, 2, dtype> hidden_input_loss = NewTensor<xpu>(Shape2(1, _hidden_input_size), d_zero);
			hidden_layer.ComputeBackwardLoss(hidden_input, hidden, hiddenLoss, hidden_input_loss);

			vector<Tensor<xpu, 2, dtype> > v_hidden_input_loss;

			// word channel
			if(bWord) {

				v_hidden_input_loss.push_back(oly_before_pool);
				v_hidden_input_loss.push_back(oly_after_pool);
			}

			unconcat(v_hidden_input_loss, hidden_input_loss);


			// word channel
			if(bWord) {
				pool_backward(oly_before_pool, y_before_poolIndex, loss_before);
				pool_backward(oly_after_pool, y_after_poolIndex, loss_after);


				unit_before.ComputeBackwardLoss(input_before, iy_before, oy_before,
								      fy_before, mcy_before, cy_before, my_before,
								      y_before, loss_before, inputLoss_before);
				unit_after.ComputeBackwardLoss(input_after, iy_after, oy_after,
								      fy_after, mcy_after, cy_after, my_after,
								      y_after, loss_after, inputLoss_after);

			}


			// word channel
			if (bWord && _words.bEmbFineTune()) {
				for (int idx = 0; idx < beforeSize; idx++) {
					inputLoss_before[idx] = inputLoss_before[idx] * mask_before[idx];
					_words.EmbLoss(example.m_before[idx], inputLoss_before[idx]);
				}
				for (int idx = 0; idx < afterSize; idx++) {
					inputLoss_after[idx] = inputLoss_after[idx] * mask_after[idx];
					_words.EmbLoss(example.m_after[idx], inputLoss_after[idx]);
				}
			}


			// word channel
			if(bWord) {
				FreeSpace(&(y_before_pool));
				FreeSpace(&(oly_before_pool));
				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(y_before_poolIndex[idx]));
				}

				FreeSpace(&(y_after_pool));
				FreeSpace(&(oly_after_pool));
				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(y_after_poolIndex[idx]));
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


			FreeSpace(&hidden_input);
			FreeSpace(&hidden_input_loss);

			FreeSpace(&hidden);
			FreeSpace(&hiddenLoss);

			FreeSpace(&output);


		} // end for example_num


		return cost;
	}

	void updateParams() {
		if(bWord) {
			unit_before.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_after.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);

			_words.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		}



		hidden_layer.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);

		output_layer.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);


	}

};



#endif /* CLASSIFIER_DEP_H_ */
