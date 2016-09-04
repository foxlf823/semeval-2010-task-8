
#ifndef SRC_SEMIDEPRECURSIVENN_H_
#define SRC_SEMIDEPRECURSIVENN_H_
#include "tensor.h"

#include "BiLayer.h"
#include "MyLib.h"
#include "Utiltensor.h"
#include <map>

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

/*
 * The Difference from RCRNN is that RNN is removed.
 */
template<typename xpu>
class SemiDepRecursiveNN {
public:
  UniLayer<xpu> _recursive_p;
  UniLayer<xpu> _recursive_r_other;
  vector< UniLayer<xpu> > _recursive_r;
  map<string, int> _map_r; // key is the dependency type, value is the index in _recursive_r

  Tensor<xpu, 2, dtype> _b;
  Tensor<xpu, 2, dtype> _gradb;
  Tensor<xpu, 2, dtype> _eg2b;

  int _outputsize_rs;
  int _inputsize;

public:
  SemiDepRecursiveNN() {
  }

  virtual ~SemiDepRecursiveNN() {
    // TODO Auto-generated destructor stub
  }

  inline void initial(int outputsize_rs, int inputsize, int seed = 0) {
	_outputsize_rs = outputsize_rs;
	_inputsize = inputsize;

	_recursive_p.initial(outputsize_rs, inputsize, false, seed+10, 2);

    dtype bound = sqrt(6.0 / (outputsize_rs + inputsize + 1));
    _b = NewTensor<xpu>(Shape2(1, outputsize_rs), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, outputsize_rs), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, outputsize_rs), d_zero);
    random(_b, -1.0 * bound, 1.0 * bound, seed + 20);

    // other
    _recursive_r_other.initial(outputsize_rs, inputsize, false, seed+40, 2);
    // 16 most common dependency types
    _map_r.insert(map<string, int>::value_type("det", 0));
    _map_r.insert(map<string, int>::value_type("prep", 1));
    _map_r.insert(map<string, int>::value_type("pobj", 2));
    _map_r.insert(map<string, int>::value_type("amod", 3));
    _map_r.insert(map<string, int>::value_type("nsubj", 4));
    _map_r.insert(map<string, int>::value_type("nn", 5));
    _map_r.insert(map<string, int>::value_type("dobj", 6));
    _map_r.insert(map<string, int>::value_type("conj", 7));
    _map_r.insert(map<string, int>::value_type("cc", 8));
    _map_r.insert(map<string, int>::value_type("advmod", 9));
    _map_r.insert(map<string, int>::value_type("aux", 10));
    _map_r.insert(map<string, int>::value_type("dep", 11));
    _map_r.insert(map<string, int>::value_type("auxpass", 12));
    _map_r.insert(map<string, int>::value_type("poss", 13));
    _map_r.insert(map<string, int>::value_type("nsubjpass", 14));
    _map_r.insert(map<string, int>::value_type("vmod", 15));
    for(int i=0;i<_map_r.size();i++) {
    	UniLayer<xpu> temp;
    	temp.initial(outputsize_rs, inputsize, false, seed+50+i*10, 2);
    	_recursive_r.push_back(temp);
    }



  }

  inline void release() {
	_recursive_p.release();

	_recursive_r_other.release();
	for(int i=0;i<_recursive_r.size();i++)
		_recursive_r[i].release();

    FreeSpace(&_b);
    FreeSpace(&_gradb);
    FreeSpace(&_eg2b);

  }


public:


  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, const vector<int> &dep,
		  const vector<string> &depType,
		  vector<Tensor<xpu, 3, dtype> > &v_rsc, Tensor<xpu, 3, dtype> rsp,
		  Tensor<xpu, 3, dtype> y) {
    rsp = 0.0;
	y = 0.0;
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if(seq_size != dep.size() || seq_size != y.size(0) || seq_size != rsp.size(0)) {
        std::cerr << "SemiDepRecursiveNN error: seq_size not equal" << std::endl;
        return;
    }

    if(x.size(3)!=_inputsize || rsp.size(3)!=_outputsize_rs) {
        std::cerr << "SemiDepRecursiveNN error: size(3) not equal" << std::endl;
        return;
    }


    for(int i=0;i<seq_size;i++) {
    	// find its all children, do recursive
    	vector<int> childrenIndex;
    	vector<string> childrenDepType;
    	for(int j=0;j<seq_size;j++) {
    		if(dep[j] < 0)
    			continue;
    		if(dep[j] == i) {
    			childrenIndex.push_back(j);
    			childrenDepType.push_back(depType[j]);
    		}
    	}

    	// Wp*p
    	_recursive_p.ComputeForwardScore(x[i], rsp[i]);
    	y[i] += rsp[i];

		// Wr*c
		Tensor<xpu, 3, dtype> rsc = NewTensor<xpu>(Shape3
				(childrenIndex.size(), 1, _outputsize_rs), d_zero);
		for(int j=0;j<childrenIndex.size();j++) {
			map<string, int>::iterator it = _map_r.find(childrenDepType[j]);
			if (it != _map_r.end()) {
				_recursive_r[it->second].ComputeForwardScore(x[childrenIndex[j]], rsc[j]);
			} else {
				_recursive_r_other.ComputeForwardScore(x[childrenIndex[j]], rsc[j]);
			}

			y[i] += rsc[j];
		}
		v_rsc.push_back(rsc);


    	// Wp*p+Wr*c+b
    	y[i] += _b;
    	// activation
        y[i] = F<nl_tanh>(y[i]);


    }


  }


  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, const vector<int> &dep,
		  const vector<string> &depType,
		  vector<Tensor<xpu, 3, dtype> > &v_rsc, Tensor<xpu, 3, dtype> rsp,
		  Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly,
		  Tensor<xpu, 3, dtype> lx, bool bclear = false) {
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if(seq_size != dep.size() || seq_size != y.size(0) ||
    	seq_size!=ly.size(0) || seq_size!=lx.size(0)) {
        std::cerr << "SemiDepRecursiveNN error: seq_size not equal" << std::endl;
        return;
    }

    if(x.size(3)!=_inputsize || lx.size(3)!=_inputsize ) {
        std::cerr << "SemiDepRecursiveNN error: size(3) not equal" << std::endl;
        return;
    }

    if (bclear) {
      lx = 0.0;
    }



    for(int i=0;i<seq_size;i++) {
    	vector<int> childrenIndex;
      	vector<string> childrenDepType;
    	for(int j=0;j<seq_size;j++) {
    		if(dep[j] < 0)
    			continue;
    		if(dep[j] == i) {
    			childrenIndex.push_back(j);
    			childrenDepType.push_back(depType[j]);
    		}
    	}

        Tensor<xpu, 2, dtype> deri_y = NewTensor<xpu>(Shape2(1, _outputsize_rs), d_zero);
        Tensor<xpu, 2, dtype> cl_y = NewTensor<xpu>(Shape2(1, _outputsize_rs), d_zero);
        deri_y = F<nl_dtanh>(y[i]);
        cl_y = ly[i] * deri_y;

        _gradb += cl_y;

        Tensor<xpu, 3, dtype> rsc = v_rsc[i];
        for(int j=0;j<childrenIndex.size();j++) {
    		Tensor<xpu, 2, dtype> lrsc = NewTensor<xpu>(Shape2(1, _outputsize_rs), d_zero);
    		lrsc += cl_y;

			map<string, int>::iterator it = _map_r.find(childrenDepType[j]);
			if (it != _map_r.end()) {
				_recursive_r[it->second].ComputeBackwardLoss(x[childrenIndex[j]], rsc[j], lrsc, lx[childrenIndex[j]]);
			} else {
				_recursive_r_other.ComputeBackwardLoss(x[childrenIndex[j]], rsc[j], lrsc, lx[childrenIndex[j]]);
			}

            FreeSpace(&lrsc);
        }

        Tensor<xpu, 2, dtype> lrsp = NewTensor<xpu>(Shape2(1, _outputsize_rs), d_zero);
        lrsp += cl_y;
    	_recursive_p.ComputeBackwardLoss(x[i], rsp[i], lrsp, lx[i]);


        FreeSpace(&deri_y);
        FreeSpace(&cl_y);
        FreeSpace(&lrsp);
    }


  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
	  _recursive_p.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

	  _recursive_r_other.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
	  for(int i=0;i<_recursive_r.size();i++)
		  _recursive_r[i].updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb * _gradb;
      _b = _b - _gradb * adaAlpha / F<nl_sqrt>(_eg2b + adaEps);

	  clearGrad();
  }

  inline void clearGrad() {
      _gradb = 0;
  }


};

#endif /* SRC_rcrnn_H_ */
