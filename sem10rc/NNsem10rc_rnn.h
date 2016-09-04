/*
 * NNsem10rc.h
 *
 *  Created on: Mar 9, 2016
 *      Author: fox
 */

#ifndef NNSEM10RC_RNN_H_
#define NNSEM10RC_RNN_H_

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
#include "Punctuation.h"
#include "Dependency.h"

#include "PoolLSTMClassifier.h"
#include "PoolLSTMAttClassifier.h"


using namespace nr;
using namespace std;
using namespace fox;

class NNsem10rc_rnn {
public:
public:
	Options m_options;
	string nullkey;
	string unknownkey;

	Alphabet m_wordAlphabet;

	Alphabet m_randomWordAlphabet;
	Alphabet m_characterAlphabet;

	Alphabet m_nerAlphabet;
	Alphabet m_posAlphabet;
	Alphabet m_sstAlphabet;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else

  PoolLSTMClassifier<cpu> m_classifier;
  //PoolLSTMAttClassifier<cpu> m_classifier;
#endif


	NNsem10rc_rnn(const Options &options):m_options(options)/*, m_classifier(options)*/ {
		nullkey = "-#null#-";
		unknownkey = "-#unknown#-";

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

		m_randomWordAlphabet.clear();
		m_randomWordAlphabet.from_string(unknownkey);
		m_randomWordAlphabet.from_string(nullkey);

		m_characterAlphabet.clear();
		m_characterAlphabet.from_string(unknownkey);
		m_characterAlphabet.from_string(nullkey);

		m_nerAlphabet.clear();
		m_nerAlphabet.from_string(unknownkey);
		m_nerAlphabet.from_string(nullkey);

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

				randomInitNrmat(wordEmb, m_wordAlphabet, m_options.wordEmbSize, 1000);
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

		NRMat<dtype> randomWordEmb;
		randomInitNrmat(randomWordEmb, m_randomWordAlphabet, m_options.entity_embsize, 1001);
		NRMat<dtype> characterEmb;
		randomInitNrmat(characterEmb, m_characterAlphabet, m_options.otherEmbSize, 1002);
		NRMat<dtype> nerEmb;
		randomInitNrmat(nerEmb, m_nerAlphabet, m_options.otherEmbSize, 1003);
		NRMat<dtype> posEmb;
		randomInitNrmat(posEmb, m_posAlphabet, m_options.otherEmbSize, 1004);
		NRMat<dtype> sstEmb;
		randomInitNrmat(sstEmb, m_sstAlphabet, m_options.otherEmbSize, 1005);




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


		  m_classifier.init(wordEmb, m_options);

		  m_classifier.resetRemove(m_options.removePool);
		  m_classifier.setDropValue(m_options.dropProb);
		  m_classifier.setWordEmbFinetune(m_options.wordEmbFineTune);

			m_classifier._randomWord.initial(randomWordEmb);
			m_classifier._randomWord.setEmbFineTune(true);

			m_classifier._character.initial(characterEmb);
			m_classifier._character.setEmbFineTune(true);

			m_classifier._ner.initial(nerEmb);
			m_classifier._ner.setEmbFineTune(true);

			m_classifier._pos.initial(posEmb);
			m_classifier._pos.setEmbFineTune(true);

			m_classifier._sst.initial(sstEmb);
			m_classifier._sst.setEmbFineTune(true);



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
		        //std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
		      }
		      m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);

		    }

		    // an iteration end, begin to evaluate
		    if (devSet.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words, m_options.wordEmbSize);


					if((m_options.channelMode & 2) == 2) {
			    		averageUnkownEmb(m_randomWordAlphabet, m_classifier._randomWord, m_options.entity_embsize);
					}
					if((m_options.channelMode & 4) == 4) {
						averageUnkownEmb(m_characterAlphabet, m_classifier._character, m_options.otherEmbSize);
					}
					if((m_options.channelMode & 8) == 8) {
						averageUnkownEmb(m_nerAlphabet, m_classifier._ner, m_options.otherEmbSize);
					}
					if((m_options.channelMode & 16) == 16) {
						averageUnkownEmb(m_posAlphabet, m_classifier._pos, m_options.otherEmbSize);
					}
					if((m_options.channelMode & 32) == 32) {
						averageUnkownEmb(m_sstAlphabet, m_classifier._sst, m_options.otherEmbSize);
					}
		    	}

		    	metric_dev.reset();
		    	toBeOutput.clear();

				for (int idx = 0; idx < devSet.size(); idx++) {
					int gold = relationName2ID(devSet[idx].relation.type, devSet[idx].relation.isE1toE2);
					vector<dtype> results;
					int predict = -1;

					if(m_options.lossFunction == 1) {
						m_classifier.predict(devSet[idx], results);
						int classSize = m_options.omitOther ? MAX_RELATION-1:MAX_RELATION;
						int optLabel = -1;
						for(int i=0;i<classSize;i++) {
							if(optLabel < 0 || results[i]>results[optLabel])
								optLabel = i;
						}
						if(m_options.omitOther && results[optLabel]<0)
							predict = OTHER_LABEL;
						else
							predict = optLabel;
					} else {
						predict = m_classifier.predict(devSet[idx], results);
					}


					//if(gold!=OTHER_LABEL) {
						metric_dev.overall_label_count ++;
						if(gold==predict)
							metric_dev.correct_label_count++;
					//}
					toBeOutput.push_back(predict);

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
							vector<dtype> results;
							int predict = -1;

							if(m_options.lossFunction == 1) {
								m_classifier.predict(devSet[idx], results);
								int classSize = m_options.omitOther ? MAX_RELATION-1:MAX_RELATION;
								int optLabel = -1;
								for(int i=0;i<classSize;i++) {
									if(optLabel < 0 || results[i]>results[optLabel])
										optLabel = i;
								}
								if(m_options.omitOther && results[optLabel]<0)
									predict = OTHER_LABEL;
								else
									predict = optLabel;
							} else {
								predict = m_classifier.predict(devSet[idx], results);
							}

							toBeOutput.push_back(predict);

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
			Example& eg = examples[egIdx];

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
			examples[egIdx].goldLabel = id;

			vector<int> originalDeps;

			int seqIdx = 0;
			for(int tkIdx=0;tkIdx<examples[egIdx].sent.tokens.size();tkIdx++) {
/*				if(fox::Punctuation::isEnglishPunc(examples[egIdx].sent.tokens[tkIdx].word[0])) {
					cout<<"";
					continue;
				}*/

				if(isTokenInEntity(examples[egIdx].sent.tokens[tkIdx], former)) {

					if(examples[egIdx].formerTkBegin == -1)
						examples[egIdx].formerTkBegin = seqIdx;

					if(examples[egIdx].formerTkEnd == -1)
						examples[egIdx].formerTkEnd = seqIdx;
					else if(examples[egIdx].formerTkEnd < seqIdx)
						examples[egIdx].formerTkEnd = seqIdx;

				} else if(isTokenInEntity(examples[egIdx].sent.tokens[tkIdx], latter)) {

					if(examples[egIdx].latterTkBegin == -1)
						examples[egIdx].latterTkBegin = seqIdx;

					if(examples[egIdx].latterTkEnd == -1)
						examples[egIdx].latterTkEnd = seqIdx;
					else if(examples[egIdx].latterTkEnd < seqIdx)
						examples[egIdx].latterTkEnd = seqIdx;

				}

				int originalDep = examples[egIdx].sent.tokens[tkIdx].depGov;
				string depType = examples[egIdx].sent.tokens[tkIdx].depType;
				examples[egIdx].dep.push_back(originalDep-1);
				examples[egIdx].depType.push_back(depType);
				originalDeps.push_back(originalDep);


				Feature feature;

				featureName2ID(m_wordAlphabet, feature_word(examples[egIdx].sent.tokens[tkIdx]), feature.words);

				if((m_options.channelMode & 2) == 2) {
					feature.randomWord = featureName2ID(m_randomWordAlphabet, feature_randomWord(examples[egIdx].sent.tokens[tkIdx]));
				}
				if((m_options.channelMode & 4) == 4) {
					featureName2ID(m_characterAlphabet, feature_character(examples[egIdx].sent.tokens[tkIdx]), feature.characters);
				}
				if((m_options.channelMode & 8) == 8) {
					feature.ner = featureName2ID(m_nerAlphabet, feature_ner(examples[egIdx].sent.tokens[tkIdx]));
				}
				if((m_options.channelMode & 16) == 16) {
					feature.pos = featureName2ID(m_posAlphabet, feature_pos(examples[egIdx].sent.tokens[tkIdx]));
				}
				if((m_options.channelMode & 32) == 32) {
					feature.sst = featureName2ID(m_sstAlphabet, feature_sst(examples[egIdx].sent.tokens[tkIdx]));
				}

				examples[egIdx].m_features.push_back(feature);

				seqIdx++;
			}


			assert(eg.formerTkBegin!=-1 && eg.formerTkEnd!=-1);
			assert(eg.latterTkBegin!=-1 && eg.latterTkEnd!=-1);

			// order by level from big to small
			vector<int> level;
			fox::Dependency::getLevelForEachNode(originalDeps, level);
			vector<int> level1;
			for(int i=0;i<level.size();i++) {

				if(level1.size() == 0) {
					level1.push_back(level[i]);
					eg.order.push_back(i);
				} else {
					int j=0;
					for(;j<level1.size();j++) {
						if(level[i]>level1[j]) {
							level1.insert(level1.begin()+j, level[i]);
							eg.order.insert(eg.order.begin()+j, i);
							break;
						}
					}
					if(j==level1.size()) {
						level1.push_back(level[i]);
						eg.order.push_back(i);
					}
				}
			}


			// fill the padding key, now only consider word channel and dep
/*			int seq_size = eg.m_features.size();
			if(eg.formerTkBegin==0) {
				int position = 0;
				int originalDep = -1;
				string depType = "null";
				examples[egIdx].dep.insert(examples[egIdx].dep.begin()+position, originalDep-1);
				examples[egIdx].depType.insert(examples[egIdx].depType.begin()+position, depType);

				Feature feature;
				featureName2ID(m_wordAlphabet, nullkey, feature.words);
				examples[egIdx].m_features.insert(examples[egIdx].m_features.begin()+position, feature);

			} else if(eg.latterTkEnd==seq_size-1) {
				int position = seq_size;
				int originalDep = -1;
				string depType = "null";
				examples[egIdx].dep.insert(examples[egIdx].dep.begin()+position, originalDep-1);
				examples[egIdx].depType.insert(examples[egIdx].depType.begin()+position, depType);

				Feature feature;
				featureName2ID(m_wordAlphabet, nullkey, feature.words);
				examples[egIdx].m_features.insert(examples[egIdx].m_features.begin()+position, feature);

			} else if(eg.formerTkEnd+1==eg.latterTkBegin) {
				int position = eg.latterTkBegin;
				int originalDep = -1;
				string depType = "null";
				examples[egIdx].dep.insert(examples[egIdx].dep.begin()+position, originalDep-1);
				examples[egIdx].depType.insert(examples[egIdx].depType.begin()+position, depType);

				Feature feature;
				featureName2ID(m_wordAlphabet, nullkey, feature.words);
				examples[egIdx].m_features.insert(examples[egIdx].m_features.begin()+position, feature);

			}*/



		}


	}

	void createAlphabet (const vector<Example>& examples, Tool& tool, bool isTrainSet) {

		hash_map<string, int> word_stat;
		hash_map<string, int> randomWord_stat;
		hash_map<string, int> character_stat;
		hash_map<string, int> ner_stat;
		hash_map<string, int> pos_stat;
		hash_map<string, int> sst_stat;

		for(int egIdx=0;egIdx<examples.size();egIdx++) {

			for(int tkIdx=0;tkIdx<examples[egIdx].sent.tokens.size();tkIdx++) {

				string curword = feature_word(examples[egIdx].sent.tokens[tkIdx]);
				word_stat[curword]++;

				if(isTrainSet && (m_options.channelMode & 2) == 2) {
					string ranWord = feature_randomWord(examples[egIdx].sent.tokens[tkIdx]);
					randomWord_stat[ranWord]++;
				}
				if(isTrainSet && (m_options.channelMode & 4) == 4) {
					vector<string> characters = feature_character(examples[egIdx].sent.tokens[tkIdx]);
					for(int i=0;i<characters.size();i++)
						character_stat[characters[i]]++;
				}
				if(isTrainSet && (m_options.channelMode & 8) == 8) {
					string ner = feature_ner(examples[egIdx].sent.tokens[tkIdx]);
					ner_stat[ner]++;
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
			stat2Alphabet(randomWord_stat, m_randomWordAlphabet, "random word");
		}
		if(isTrainSet && (m_options.channelMode & 4) == 4) {
			stat2Alphabet(character_stat, m_characterAlphabet, "character");
		}
		if(isTrainSet && (m_options.channelMode & 8) == 8) {
			stat2Alphabet(ner_stat, m_nerAlphabet, "ner");
		}
		if(isTrainSet && (m_options.channelMode & 16) == 16) {
			stat2Alphabet(pos_stat, m_posAlphabet, "pos");
		}
		if(isTrainSet && (m_options.channelMode & 32) == 32) {
			stat2Alphabet(sst_stat, m_sstAlphabet, "sst");
		}
	}


	string feature_word(const Token& token) {
		//string ret = normalize_to_lowerwithdigit(token.word);
		string ret = normalize_to_lowerwithdigit(token.lemma);

		return ret;
	}

	string feature_randomWord(const Token& token) {
		return feature_word(token);
	}

	vector<string> feature_character(const Token& token) {
		vector<string> ret;
		string word = feature_word(token);
		for(int i=0;i<word.length();i++)
			ret.push_back(word.substr(i, 1));
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

	string feature_ner(const Token& token) {
		return token.sst;
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

	void randomInitNrmat(NRMat<dtype>& nrmat, Alphabet& alphabet, int embSize, int seed) {
		double* emb = new double[alphabet.size()*embSize];
		fox::initArray2((double *)emb, (int)alphabet.size(), embSize, 0.0);

		vector<string> known;
		map<string, int> IDs;
		alphabet2vectormap(alphabet, known, IDs);

		fox::randomInitEmb((double*)emb, embSize, known, unknownkey,
				IDs, false, m_options.initRange, seed);

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

	void featureName2ID(Alphabet& alphabet, const vector<string>& featureName, vector<int>& vfeatureID) {
		for(int i=0;i<featureName.size();i++) {
			int id = alphabet.from_string(featureName[i]);
			if(id >=0)
				vfeatureID.push_back(id);
			else
				vfeatureID.push_back(0); // assume unknownID is zero
		}
	}

	int featureName2ID(Alphabet& alphabet, const string& featureName) {
		int id = alphabet.from_string(featureName);
		if(id >=0)
			return id;
		else
			return 0; // assume unknownID is zero
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



};




#endif /* NNSEM10RC_H_ */
