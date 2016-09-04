/*
 * Example.h
 *
 *  Created on: Mar 9, 2016
 *      Author: fox
 */

#ifndef EXAMPLE_sem_H_
#define EXAMPLE_sem_H_

#include <string>
#include "RelationEntity.h"
#include "Sent.h"
#include "Feature.h"

using namespace std;
using namespace fox;

class Example_sem {
public:
	string text;
	RelationEntity relation;
	Sent sent;
	vector<int> dep;
	vector<string> depType;
	vector<int> order;

	  vector<int> m_labels; // there are 9*2+1 relation types
	  int goldLabel;

	  vector<int> m_before;
	  vector<int> m_entityFormer;
	  vector<int> m_entityLatter;
	  vector<int> m_middle;
	  vector<int> m_after;

	  vector<int> m_before_wordnet;
	  vector<int> m_middle_wordnet;
	  vector<int> m_after_wordnet;
	  vector<int> m_entityFormer_wordnet;
	  vector<int> m_entityLatter_wordnet;

	  vector<int> m_before_brown;
	  vector<int> m_middle_brown;
	  vector<int> m_after_brown;
	  vector<int> m_entityFormer_brown;
	  vector<int> m_entityLatter_brown;

	  vector<int> m_before_bigram;
	  vector<int> m_middle_bigram;
	  vector<int> m_after_bigram;
	  vector<int> m_entityFormer_bigram;
	  vector<int> m_entityLatter_bigram;

	  vector<int> m_before_pos;
	  vector<int> m_middle_pos;
	  vector<int> m_after_pos;
	  vector<int> m_entityFormer_pos;
	  vector<int> m_entityLatter_pos;

	  vector<int> m_before_sst;
	  vector<int> m_middle_sst;
	  vector<int> m_after_sst;
	  vector<int> m_entityFormer_sst;
	  vector<int> m_entityLatter_sst;

	  vector<int> m_sparseFeature;

	  vector<Feature> m_features; // one feature corresponds to a word
	  int formerTkBegin; // the beginning of former
	  int formerTkEnd; // the end of former (include)
	  int latterTkBegin; // the beginning of latter
	  int latterTkEnd; // the end of latter (include)

	  Example_sem() {
		  formerTkBegin = -1;
		  formerTkEnd = -1;
		  latterTkBegin = -1;
		  latterTkEnd = -1;
	  }
};


#endif /* EXAMPLE_H_ */
