

#ifndef SRC_CLASSIFIER_7_H_
#define SRC_CLASSIFIER_7_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "N3L.h"
#include "SemiDepRCRNN.h"
#include "SemiDepRCGRNN.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class Classifier_7 {
public:
	Classifier_7() {
    _dropOut = 0.5;
  }
  ~Classifier_7() {

  }

public:
  LookupTable<xpu> _words;

  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;

  int _outputsize_rr;
  int _outputsize_rs;
  int _inputsize;


  //SemiDepRCRNN<xpu> _rcrnn;
  SemiDepRCGRNN<xpu> _rcrnn;
  UniLayer<xpu> _olayer_linear;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  Options options;


public:

  inline void init(const NRMat<dtype>& wordEmb, Options options) {
	  this->options = options;
    _wordcontext = options.wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _labelSize = MAX_RELATION;

    _outputsize_rr = options.rnnHiddenSize;
    _outputsize_rs = options.hiddenSize;
    _inputsize = _wordwindow * _wordDim;

    _words.initial(wordEmb);

    _rcrnn.initial(_outputsize_rr, _outputsize_rs, _inputsize, true, 10);
    _olayer_linear.initial(_labelSize, _outputsize_rr, false, 20, 2);

    cout<<"do rcrnn in the dependency tree"<<endl;
  }

  inline void release() {
    _words.release();
    _rcrnn.release();
    _olayer_linear.release();
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

      Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, dtype> input, inputLoss;

      Tensor<xpu, 3, dtype> rcrnn_hidden, rcrnn_hiddenLoss, rcrnn_rsy, rcrnn_rsp;
	  Tensor<xpu, 3, dtype> mry, ry, uy, cy;

      vector< Tensor<xpu, 3, dtype> > rcrnn_v_rsc;

      Tensor<xpu, 2, dtype> pool, poolLoss;
      Tensor<xpu, 3, dtype> poolIndex;

      Tensor<xpu, 2, dtype> output, outputLoss;


      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 1.0);

      input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);
      inputLoss = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

      rcrnn_hidden = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);
      rcrnn_hiddenLoss = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);

      rcrnn_rsy = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rs), 0.0);
      rcrnn_rsp = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rs), 0.0);

	  mry = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);
	  ry = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);
	  uy = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);
	  cy = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);


      pool = NewTensor<xpu>(Shape2(1, _outputsize_rr), 0.0);
      poolLoss = NewTensor<xpu>(Shape2(1, _outputsize_rr), 0.0);
      poolIndex = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);

      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);


      // forward
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        srand(iter * example_num + count * seq_size + idx);
        const vector<int>& words = feature.words;
       _words.GetEmb(words[0], wordprime[idx]);

        //dropout
        dropoutcol(wordprimeMask[idx], _dropOut);
        wordprime[idx] = wordprime[idx] * wordprimeMask[idx];
      }

      windowlized(wordprime, input, _wordcontext);

      _rcrnn.ComputeForwardScore(input, example.dep, example.depType,
    		  rcrnn_rsy, rcrnn_v_rsc, rcrnn_rsp,
    		  mry, ry, uy, cy,
    		  rcrnn_hidden);

      maxpool_forward(rcrnn_hidden, pool, poolIndex);

      _olayer_linear.ComputeForwardScore(pool, output);

      // backward
      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      _olayer_linear.ComputeBackwardLoss(pool, output, outputLoss, poolLoss);


      pool_backward(poolLoss, poolIndex, rcrnn_hiddenLoss);


      _rcrnn.ComputeBackwardLoss(input, example.dep, example.depType,
    		  rcrnn_rsy, rcrnn_v_rsc, rcrnn_rsp,
    		  mry, ry, uy, cy,
		  rcrnn_hidden, rcrnn_hiddenLoss, inputLoss);

	  
      windowlized_backward(wordprimeLoss, inputLoss, _wordcontext);

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

      FreeSpace(&input);
      FreeSpace(&inputLoss);

      FreeSpace(&rcrnn_hidden);
      FreeSpace(&rcrnn_hiddenLoss);
      FreeSpace(&rcrnn_rsy);
      FreeSpace(&rcrnn_rsp);

      FreeSpace(&mry);
      FreeSpace(&ry);
      FreeSpace(&uy);
      FreeSpace(&cy);

      for(int i=0;i<rcrnn_v_rsc.size();i++)
    	  FreeSpace(&(rcrnn_v_rsc[i]));

		FreeSpace(&(pool));
		FreeSpace(&(poolLoss));
		FreeSpace(&(poolIndex));

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
    if(seq_size==0)
  	  return OTHER_LABEL;
    int offset = 0;

    Tensor<xpu, 3, dtype> wordprime;
    Tensor<xpu, 3, dtype> input;

    Tensor<xpu, 3, dtype> rcrnn_hidden, rcrnn_rsy, rcrnn_rsp;
	Tensor<xpu, 3, dtype> mry, ry, uy, cy;

    vector< Tensor<xpu, 3, dtype> > rcrnn_v_rsc;

    Tensor<xpu, 2, dtype> pool;
    Tensor<xpu, 3, dtype> poolIndex;

    Tensor<xpu, 2, dtype> output;

    //initialize
    wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);

    input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

    rcrnn_hidden = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);

    rcrnn_rsy = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rs), 0.0);
    rcrnn_rsp = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rs), 0.0);

	  mry = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);
	  ry = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);
	  uy = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);
	  cy = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);

    pool = NewTensor<xpu>(Shape2(1, _outputsize_rr), 0.0);
    poolIndex = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);

    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

    //forward
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      const vector<int>& words = feature.words;
      _words.GetEmb(words[0], wordprime[idx]);
    }


    windowlized(wordprime, input, _wordcontext);

    _rcrnn.ComputeForwardScore(input, example.dep, example.depType,
    		rcrnn_rsy, rcrnn_v_rsc, rcrnn_rsp,
  		  mry, ry, uy, cy,
  		  rcrnn_hidden);

    maxpool_forward(rcrnn_hidden, pool, poolIndex);

    _olayer_linear.ComputeForwardScore(pool, output);


    int optLabel = softmax_predict(output, results);

    //release
    FreeSpace(&wordprime);

    FreeSpace(&input);

    FreeSpace(&rcrnn_hidden);
    FreeSpace(&rcrnn_rsy);
    FreeSpace(&rcrnn_rsp);

    FreeSpace(&mry);
    FreeSpace(&ry);
    FreeSpace(&uy);
    FreeSpace(&cy);


    for(int i=0;i<rcrnn_v_rsc.size();i++)
  	  FreeSpace(&(rcrnn_v_rsc[i]));

		FreeSpace(&(pool));
		FreeSpace(&(poolIndex));

    FreeSpace(&output);

    return optLabel;
  }


  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rcrnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);


    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void checkgrads(const vector<Example>& examples, int iter) {

    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);

    checkgrad(examples, _rcrnn._recursive_p._W, _rcrnn._recursive_p._gradW, "_rcrnn._recursive_p._W", iter);
    checkgrad(examples, _rcrnn._recursive_r_other._W, _rcrnn._recursive_r_other._gradW, "_rcrnn._recursive_r_other._W", iter);
    for(int i=0;i<_rcrnn._recursive_r.size();i++) {
    	stringstream ss;
    	ss<<"_rcrnn._recursive_r["<<i<<"]._W";
    	checkgrad(examples, _rcrnn._recursive_r[i]._W, _rcrnn._recursive_r[i]._gradW, ss.str(), iter);
    }
    checkgrad(examples, _rcrnn._b, _rcrnn._gradb, "_rcrnn._b", iter);

    checkgrad(examples, _rcrnn._recurrent._rnn._WL, _rcrnn._recurrent._rnn._gradWL, "_rcrnn._recurrent._rnn._WL", iter);
    checkgrad(examples, _rcrnn._recurrent._rnn._WR, _rcrnn._recurrent._rnn._gradWR, "_rcrnn._recurrent._rnn._WR", iter);
    checkgrad(examples, _rcrnn._recurrent._rnn._b, _rcrnn._recurrent._rnn._gradb, "_rcrnn._recurrent._rnn._b", iter);

    checkgrad(examples, _rcrnn._recurrent._rnn_update._WL, _rcrnn._recurrent._rnn_update._gradWL, "_rcrnn._recurrent._rnn_update._WL", iter);
    checkgrad(examples, _rcrnn._recurrent._rnn_update._WR, _rcrnn._recurrent._rnn_update._gradWR, "_rcrnn._recurrent._rnn_update._WR", iter);
    checkgrad(examples, _rcrnn._recurrent._rnn_update._b, _rcrnn._recurrent._rnn_update._gradb, "_rcrnn._recurrent._rnn_update._b", iter);

    checkgrad(examples, _rcrnn._recurrent._rnn_reset._WL, _rcrnn._recurrent._rnn_reset._gradWL, "_rcrnn._recurrent._rnn_reset._WL", iter);
    checkgrad(examples, _rcrnn._recurrent._rnn_reset._WR, _rcrnn._recurrent._rnn_reset._gradWR, "_rcrnn._recurrent._rnn_reset._WR", iter);
    checkgrad(examples, _rcrnn._recurrent._rnn_reset._b, _rcrnn._recurrent._rnn_reset._gradb, "_rcrnn._recurrent._rnn_reset._b", iter);


    checkgrad(examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);

  }

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

  dtype computeScore(const Example& example) {

		const vector<Feature>& features = example.m_features;
	    int seq_size = features.size();
	    if(seq_size==0)
	  	  return OTHER_LABEL;
	    int offset = 0;

	    Tensor<xpu, 3, dtype> wordprime;
	    Tensor<xpu, 3, dtype> input;

	    Tensor<xpu, 3, dtype> rcrnn_hidden, rcrnn_rsy, rcrnn_rsp;
		  Tensor<xpu, 3, dtype> mry, ry, uy, cy;

	    vector< Tensor<xpu, 3, dtype> > rcrnn_v_rsc;

	    Tensor<xpu, 2, dtype> pool;
	    Tensor<xpu, 3, dtype> poolIndex;

	    Tensor<xpu, 2, dtype> output;

	    //initialize
	    wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);

	    input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

	    rcrnn_hidden = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);

	    rcrnn_rsy = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rs), 0.0);
	    rcrnn_rsp = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rs), 0.0);

		  mry = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);
		  ry = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);
		  uy = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);
		  cy = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);

	    pool = NewTensor<xpu>(Shape2(1, _outputsize_rr), 0.0);
	    poolIndex = NewTensor<xpu>(Shape3(seq_size, 1, _outputsize_rr), 0.0);

	    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

	    //forward
	    for (int idx = 0; idx < seq_size; idx++) {
	      const Feature& feature = features[idx];
	      const vector<int>& words = feature.words;
	      _words.GetEmb(words[0], wordprime[idx]);
	    }


	    windowlized(wordprime, input, _wordcontext);

	      _rcrnn.ComputeForwardScore(input, example.dep, example.depType,
	    		  rcrnn_rsy, rcrnn_v_rsc, rcrnn_rsp,
	    		  mry, ry, uy, cy,
	    		  rcrnn_hidden);

	    maxpool_forward(rcrnn_hidden, pool, poolIndex);

	    _olayer_linear.ComputeForwardScore(pool, output);

	    dtype cost = softmax_cost(output, example.m_labels);

	    //release
	    FreeSpace(&wordprime);

	    FreeSpace(&input);

	    FreeSpace(&rcrnn_hidden);
	    FreeSpace(&rcrnn_rsy);
	    FreeSpace(&rcrnn_rsp);

	      FreeSpace(&mry);
	      FreeSpace(&ry);
	      FreeSpace(&uy);
	      FreeSpace(&cy);

	    for(int i=0;i<rcrnn_v_rsc.size();i++)
	  	  FreeSpace(&(rcrnn_v_rsc[i]));

			FreeSpace(&(pool));
			FreeSpace(&(poolIndex));

	    FreeSpace(&output);

	    return cost;

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

};

#endif /* SRC_PoolRNNClassifier_H_ */

