/*
 * NNsem10rc.h
 *
 *  Created on: Mar 9, 2016
 *      Author: fox
 */

#ifndef NNSEM10RC_DEP_H_
#define NNSEM10RC_DEP_H_

#include "N3Lhelper.h"
#include "N3L.h"
#include "Options.h"
#include "Tool.h"
#include "utils.h"
#include "EnglishPos.h"
#include "WordNet.h"
#include "Example.h"
#include "Sent.h"
#include "FoxUtil.h"
#include "Token.h"
#include "Word2Vec.h"
#include "Classifier.h"
#include "Classifier_attentionword.h"
#include "Classifier_attentionlstm.h"
#include "Classifier_pooling.h"
#include "Classifier_jointwordpool.h"
#include "Classifier_dep.h"
#include "Dependency.h"

using namespace nr;
using namespace std;
using namespace fox;

class NNsem10rc_dep {
public:
public:
	Options m_options;
	string nullkey;
	string unknownkey;
	string sentencekey;

	Alphabet m_wordAlphabet;
	Alphabet m_wordnetAlphabet;
	Alphabet m_brownAlphabet;
	Alphabet m_bigramAlphabet;
	Alphabet m_posAlphabet;
	Alphabet m_sstAlphabet;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else
//  Classifier<cpu> m_classifier;
//  Classifier_attentionword<cpu> m_classifier;
//  Classifier_attentionlstm<cpu> m_classifier;
//  Classifier_pooling<cpu> m_classifier;
  Classifier_dep<cpu> m_classifier;
#endif


	NNsem10rc_dep(const Options &options):m_options(options), m_classifier(options) {
		nullkey = "-#null#-";
		unknownkey = "-#unknown#-";
		sentencekey = "-#sent#-";
	}

	void train(const string& trainFile, const string& devFile, const string& testFile,
				Tool& tool, const string& trainNlpFile, const string& devNlpFile, const string& testNlpFile) {

		// load train data
		vector<Example> trainSet;
		parseSem10rcFile(trainFile, trainSet);
		vector<Example> devSet;
		if(!devFile.empty()) {
			parseSem10rcFile(devFile, devSet);
		}
		vector<Example> testSet;
		if(!testFile.empty()) {
			parseSem10rcFile(testFile, testSet);
		}

		// load nlp file
		loadNlpFile(trainNlpFile, trainSet);
		if(!devNlpFile.empty()) {
			loadNlpFile(devNlpFile, devSet);
		}
		if(!testNlpFile.empty()) {
			loadNlpFile(testNlpFile, testSet);
		}


		/*
		 * For all alphabets, unknownkey and nullkey should be 0 and 1.
		 */
		m_wordAlphabet.clear();
		m_wordAlphabet.from_string(unknownkey);
		m_wordAlphabet.from_string(nullkey);
		m_wordAlphabet.from_string(sentencekey);

		m_wordnetAlphabet.clear();
		m_wordnetAlphabet.from_string(unknownkey);
		m_wordnetAlphabet.from_string(nullkey);

		m_brownAlphabet.clear();
		m_brownAlphabet.from_string(unknownkey);
		m_brownAlphabet.from_string(nullkey);

		m_bigramAlphabet.clear();
		m_bigramAlphabet.from_string(unknownkey);
		m_bigramAlphabet.from_string(nullkey);

		m_posAlphabet.clear();
		m_posAlphabet.from_string(unknownkey);
		m_posAlphabet.from_string(nullkey);

		m_sstAlphabet.clear();
		m_sstAlphabet.from_string(unknownkey);
		m_sstAlphabet.from_string(nullkey);


		createAlphabet(trainSet, tool, true);

		if (!m_options.wordEmbFineTune) {
			// if not fine tune, use all the data to build alphabet
			if(!devSet.empty())
				createAlphabet(devSet, tool, false);
			if(!testSet.empty())
				createAlphabet(testSet, tool, false);
		}

		NRMat<dtype> wordEmb;
		if(m_options.wordEmbFineTune) {
			if(m_options.embFile.empty()) {
				cout<<"random emb"<<endl;

				randomInitNrmat(wordEmb, m_wordAlphabet, m_options.wordEmbSize);
			} else {
				cout<< "load pre-trained emb"<<endl;
				tool.w2v->loadFromBinFile(m_options.embFile, false, true);
				// format the words of pre-trained embeddings
				//formatWords(tool.w2v);
				double* emb = new double[m_wordAlphabet.size()*m_options.wordEmbSize];
				fox::initArray2((double *)emb, (int)m_wordAlphabet.size(), m_options.wordEmbSize, 0.0);
				vector<string> known;
				map<string, int> IDs;
				alphabet2vectormap(m_wordAlphabet, known, IDs);

				tool.w2v->getEmbedding((double*)emb, m_options.wordEmbSize, known, unknownkey, IDs);

				wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
				array2NRMat((double*) emb, m_wordAlphabet.size(), m_options.wordEmbSize, wordEmb);

				delete[] emb;
			}
		} else {
			if(m_options.embFile.empty()) {
				assert(0);
			} else {
				cout<< "load pre-trained emb"<<endl;
				tool.w2v->loadFromBinFile(m_options.embFile, false, true);
				// format the words of pre-trained embeddings
				//formatWords(tool.w2v);
				double* emb = new double[m_wordAlphabet.size()*m_options.wordEmbSize];
				fox::initArray2((double *)emb, (int)m_wordAlphabet.size(), m_options.wordEmbSize, 0.0);
				vector<string> known;
				map<string, int> IDs;
				alphabet2vectormap(m_wordAlphabet, known, IDs);

				tool.w2v->getEmbedding((double*)emb, m_options.wordEmbSize, known, unknownkey, IDs);

				wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
				array2NRMat((double*) emb, m_wordAlphabet.size(), m_options.wordEmbSize, wordEmb);

				delete[] emb;
			}
		}

		NRMat<dtype> wordnetEmb;
		randomInitNrmat(wordnetEmb, m_wordnetAlphabet, m_options.otherEmbSize);
		NRMat<dtype> brownEmb;
		randomInitNrmat(brownEmb, m_brownAlphabet, m_options.otherEmbSize);
		NRMat<dtype> bigramEmb;
		randomInitNrmat(bigramEmb, m_bigramAlphabet, m_options.otherEmbSize);
		NRMat<dtype> posEmb;
		randomInitNrmat(posEmb, m_posAlphabet, m_options.otherEmbSize);
		NRMat<dtype> sstEmb;
		randomInitNrmat(sstEmb, m_sstAlphabet, m_options.otherEmbSize);


		initialExamples(tool, trainSet);
		cout<<"Total train example number: "<<trainSet.size()<<endl;
		if(!trainSet.empty()) {
			initialExamples(tool, devSet);
			cout<<"Total dev example number: "<<devSet.size()<<endl;
		}
		vector<Example> testExamples;
		if(!testSet.empty()) {
			initialExamples(tool, testSet);
			cout<<"Total test example number: "<<testSet.size()<<endl;
		}

		m_classifier.init(MAX_RELATION, wordEmb, wordnetEmb,brownEmb, bigramEmb, posEmb, sstEmb, 0);

		int inputSize = trainSet.size();
		int batchBlock = inputSize / m_options.batchSize;
		if (inputSize % m_options.batchSize != 0)
			batchBlock++;

		std::vector<int> indexes;
		for (int i = 0; i < inputSize; ++i)
			indexes.push_back(i);

		static Metric eval, metric_dev;
		static vector<Example> subExamples;
		int devNum = devSet.size(), testNum = testExamples.size();

		dtype best = 0;
		vector<int> toBeOutput;

		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {
			if(m_options.verboseIter>0)
				cout << "##### Iteration " << iter << std::endl;

		    random_shuffle(indexes.begin(), indexes.end());
		    eval.reset();

		    // use all batches to train during an iteration
		    for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
				subExamples.clear();
				int start_pos = updateIter * m_options.batchSize;
				int end_pos = (updateIter + 1) * m_options.batchSize;
				if (end_pos > inputSize)
					end_pos = inputSize;

				for (int idy = start_pos; idy < end_pos; idy++) {
					subExamples.push_back(trainSet[indexes[idy]]);
				}

				int curUpdateIter = iter * batchBlock + updateIter;
				dtype cost = m_classifier.process(subExamples, curUpdateIter);

				eval.overall_label_count += m_classifier._eval.overall_label_count;
				eval.correct_label_count += m_classifier._eval.correct_label_count;

		      if (m_options.verboseIter>0 && (curUpdateIter + 1) % m_options.verboseIter == 0) {
		        //m_classifier.checkgrads(subExamples, curUpdateIter+1);
		        //std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
		        std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
		      }
		      m_classifier.updateParams();

		    }

		    // an iteration end, begin to evaluate
		    if (devSet.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words, m_options.wordEmbSize);

		    	}

		    	metric_dev.reset();
		    	toBeOutput.clear();

				for (int idx = 0; idx < devSet.size(); idx++) {
					int id = predict(devSet[idx]);
					int gold = relationName2ID(devSet[idx].relation.type, devSet[idx].relation.isE1toE2);

					//if(id!=18) {
						metric_dev.overall_label_count ++;
						if(gold==id)
							metric_dev.correct_label_count++;
					//}
					toBeOutput.push_back(id);
				}

				metric_dev.print();

				if (metric_dev.getAccuracy() > best) {
					cout << "Exceeds best performance of " << best << endl;
					best = metric_dev.getAccuracy();
					outputToSem10rc(toBeOutput, m_options.output);
					// if the current exceeds the best, we do the blind test on the test set
					// but don't evaluate and store the results for the official evaluation
					if (testSet.size() > 0) {
						toBeOutput.clear();

						for (int idx = 0; idx < testSet.size(); idx++) {
							int id = predict(testSet[idx]);

							toBeOutput.push_back(id);

						}

						outputToSem10rc(toBeOutput, m_options.output);
					}
				}



		    } // devExamples > 0

		} // for iter




		m_classifier.release();

	}

	void initialExamples(Tool& tool, vector<Example>& examples) {

		for(int egIdx=0;egIdx<examples.size();egIdx++) {

			// find all the entities in the current sentence
			Entity& latter = examples[egIdx].relation.entity2;

			Entity& former = examples[egIdx].relation.entity1;

			int id = relationName2ID(examples[egIdx].relation.type, examples[egIdx].relation.isE1toE2);
			for(int i=0;i<MAX_RELATION;i++) {
				if(i!=id)
					examples[egIdx].m_labels.push_back(0);
				else
					examples[egIdx].m_labels.push_back(1);
			}

			// get head word of the entity
			int headFormerIdx = getEntityHeadWord(former, examples[egIdx].sent);
			int headLatterIdx = getEntityHeadWord(latter, examples[egIdx].sent);
			if(headFormerIdx==-1 || headLatterIdx == -1)
				assert(0);

			// before corresponds to the shortest path from former to common ancestor
			// after corresponds to the shortest path from latter to common ancestor
			vector<int> sdpA;
			vector<int> sdpB;
			// we consider they are in the same sentence!!!!!!!!!!!!!!!!
			int common = fox::Dependency::getCommonAncestor(examples[egIdx].sent.tokens,
					headFormerIdx, headLatterIdx, sdpA, sdpB);

			if(common!=-2) {

				for(int sdpANodeIdx=0;sdpANodeIdx<sdpA.size();sdpANodeIdx++) {
					string word;
					if(sdpA[sdpANodeIdx]!=0)
						word = examples[egIdx].sent.tokens[sdpA[sdpANodeIdx]-1].word;
					else
						word = sentencekey;

					featureName2ID(m_wordAlphabet, feature_word(word), examples[egIdx].m_before);

				}

				for(int sdpBNodeIdx=0;sdpBNodeIdx<sdpB.size();sdpBNodeIdx++) {
					string word;
					if(sdpB[sdpBNodeIdx]!=0)
						word = examples[egIdx].sent.tokens[sdpB[sdpBNodeIdx]-1].word;
					else
						word = sentencekey;

					featureName2ID(m_wordAlphabet, feature_word(word), examples[egIdx].m_after);

				}

			}



			// for concise, we don't judge channel mode here, but it's ok since
			// classifier will not use unnecessary channel
			// in case that null
			if(examples[egIdx].m_before.size()==0) {
				examples[egIdx].m_before.push_back(m_wordAlphabet.from_string(nullkey));
			}
			if(examples[egIdx].m_after.size()==0) {
				examples[egIdx].m_after.push_back(m_wordAlphabet.from_string(nullkey));
			}


		}


	}

	void createAlphabet (const vector<Example>& examples, Tool& tool, bool isTrainSet) {

		hash_map<string, int> word_stat;
		hash_map<string, int> wordnet_stat;
		hash_map<string, int> brown_stat;
		hash_map<string, int> bigram_stat;
		hash_map<string, int> pos_stat;
		hash_map<string, int> sst_stat;

		for(int egIdx=0;egIdx<examples.size();egIdx++) {

			for(int tkIdx=0;tkIdx<examples[egIdx].sent.tokens.size();tkIdx++) {

				string curword = feature_word(examples[egIdx].sent.tokens[tkIdx]);
				word_stat[curword]++;

				if(isTrainSet && (m_options.channelMode & 2) == 2) {
					string wordnet = feature_wordnet(examples[egIdx].sent.tokens[tkIdx]);
					wordnet_stat[wordnet]++;
				}
				if(isTrainSet && (m_options.channelMode & 4) == 4) {
					string brown = feature_brown(examples[egIdx].sent.tokens[tkIdx], tool);
					brown_stat[brown]++;
				}
				if(isTrainSet && (m_options.channelMode & 8) == 8) {
					string bigram = feature_bigram(examples[egIdx].sent.tokens, tkIdx);
					bigram_stat[bigram]++;
				}
				if(isTrainSet && (m_options.channelMode & 16) == 16) {
					string pos = feature_pos(examples[egIdx].sent.tokens[tkIdx]);
					pos_stat[pos]++;
				}
				if(isTrainSet && (m_options.channelMode & 32) == 32) {
					string sst = feature_sst(examples[egIdx].sent.tokens[tkIdx]);
					sst_stat[sst]++;
				}




			}




		}

		stat2Alphabet(word_stat, m_wordAlphabet, "word");

		if(isTrainSet && (m_options.channelMode & 2) == 2) {
			stat2Alphabet(wordnet_stat, m_wordnetAlphabet, "wordnet");
		}
		if(isTrainSet && (m_options.channelMode & 4) == 4) {
			stat2Alphabet(brown_stat, m_brownAlphabet, "brown");
		}
		if(isTrainSet && (m_options.channelMode & 8) == 8) {
			stat2Alphabet(bigram_stat, m_bigramAlphabet, "bigram");
		}
		if(isTrainSet && (m_options.channelMode & 16) == 16) {
			stat2Alphabet(pos_stat, m_posAlphabet, "pos");
		}
		if(isTrainSet && (m_options.channelMode & 32) == 32) {
			stat2Alphabet(sst_stat, m_sstAlphabet, "sst");
		}
	}


	string feature_word(const Token& token) {
		return feature_word(token.word);
	}

	string feature_word(const string& word) {
		string ret = normalize_to_lowerwithdigit(word);
		return ret;
	}

	string feature_wordnet(const Token& token) {

		string lemmalow = fox::toLowercase(token.lemma);
		char buffer[64] = {0};
		sprintf(buffer, "%s", lemmalow.c_str());

		int pos = -1;
		EnglishPosType type = EnglishPos::getType(token.pos);
		if(type == FOX_NOUN)
			pos = WNNOUN;
		else if(type == FOX_VERB)
			pos = WNVERB;
		else if(type == FOX_ADJ)
			pos = WNADJ;
		else if(type == FOX_ADV)
			pos = WNADV;

		if(pos != -1) {
			string id = fox::getWnID(buffer, pos, 1);
			if(!id.empty())
				return id;
			else
				return unknownkey;
		} else
			return unknownkey;


	}

	string feature_brown(const Token& token, Tool& tool) {
		string brownID = tool.brown.get(fox::toLowercase(token.word));
		if(!brownID.empty())
			return brownID;
		else
			return unknownkey;
	}

	string feature_bigram(const vector<Token>& tokens, int idx) {
		string bigram;

		if(idx>0) {
			bigram = normalize_to_lowerwithdigit(tokens[idx-1].word+"_"+tokens[idx].word);
		} else {
			bigram = normalize_to_lowerwithdigit(nullkey+"_"+tokens[idx].word);
		}
		return bigram;
	}

	string feature_pos(const Token& token) {
		return token.pos;
	}

	string feature_sst(const Token& token) {
		int pos = token.sst.find("B-");
		if(pos!=-1) {
			return token.sst.substr(pos+2);
		} else {
			pos = token.sst.find("I-");
			if(pos!=-1) {
				return token.sst.substr(pos+2);
			} else
				return token.sst;
		}
		//return token.sst;


	}

	void stat2Alphabet(hash_map<string, int>& stat, Alphabet& alphabet, const string& label) {

		cout << label<<" num: " << stat.size() << endl;
		alphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator feat_iter;
		for (feat_iter = stat.begin(); feat_iter != stat.end(); feat_iter++) {
			// if not fine tune, add all the words; if fine tune, add the words considering wordCutOff
			// in order to train unknown
			if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
			  alphabet.from_string(feat_iter->first);
			}
		}
		cout << "alphabet "<< label<<" num: " << alphabet.size() << endl;
		alphabet.set_fixed_flag(true);

	}

	void randomInitNrmat(NRMat<dtype>& nrmat, Alphabet& alphabet, int embSize) {
		double* emb = new double[alphabet.size()*embSize];
		fox::initArray2((double *)emb, (int)alphabet.size(), embSize, 0.0);

		vector<string> known;
		map<string, int> IDs;
		alphabet2vectormap(alphabet, known, IDs);

		fox::randomInitEmb((double*)emb, embSize, known, unknownkey,
				IDs, false, m_options.initRange);

		nrmat.resize(alphabet.size(), embSize);
		array2NRMat((double*) emb, alphabet.size(), embSize, nrmat);

		delete[] emb;
	}

	void featureName2ID(Alphabet& alphabet, const string& featureName, vector<int>& vfeatureID) {
		int id = alphabet.from_string(featureName);
		if(id >=0)
			vfeatureID.push_back(id);
		else
			vfeatureID.push_back(0); // assume unknownID is zero
	}

	template<typename xpu>
	void averageUnkownEmb(Alphabet& alphabet, LookupTable<xpu>& table, int embSize) {

		// unknown cannot be trained, use the average embedding
		int unknownID = alphabet.from_string(unknownkey);
		Tensor<cpu, 2, dtype> temp = NewTensor<cpu>(Shape2(1, embSize), d_zero);
		int number = table._nVSize-1;
		table._E[unknownID] = 0.0;
		for(int i=0;i<table._nVSize;i++) {
			if(i==unknownID)
				continue;
			table.GetEmb(i, temp);
			table._E[unknownID] += temp[0]/number;
		}

		FreeSpace(&temp);

	}


	// return the best choice
	int predict(const Example& example) {
		vector<double> scores(MAX_RELATION);
		m_classifier.predict(example, scores);

		int id = 0;
		double max = scores[0];
		for(int i=1;i<scores.size();i++) {
			if(scores[i]>max) {
				max = scores[i];
				id = i;
			}
		}

		return id;
	}

};




#endif /* NNSEM10RC_H_ */
