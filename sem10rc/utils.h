/*
 * utils.h
 *
 *  Created on: Dec 20, 2015
 *      Author: fox
 */

#ifndef UTILS_H_
#define UTILS_H_

/*
 * cdr.cpp
 *
 *  Created on: Dec 19, 2015
 *      Author: fox
 */

#include <stdio.h>
#include <vector>
#include "Word2Vec.h"
#include "Utf.h"
#include "Entity.h"
#include "Token.h"
#include "Example.h"
#include "FoxUtil.h"
#include <sys/types.h>
#include <sys/stat.h>
#include "Sent.h"

using namespace std;
using namespace fox;

#define MAX_RELATION 19
#define OTHER_LABEL 18

// Give an entity and a sentence, find the entity last word and return its token idx
int getEntityHeadWord(const Entity& entity, const fox::Sent& sent) {
	int entityEnd = entity.end2==-1 ? entity.end : entity.end2;
	for(int i=0;i<sent.tokens.size();i++) {
		if(entityEnd == sent.tokens[i].end)
			return i;
	}
	return -1;
}

int relationName2ID(const string& name, bool isE1toE2) {
	if(name == "Cause-Effect") {
		return isE1toE2 ? 0:1;
	} else if(name == "Instrument-Agency") {
		return isE1toE2 ? 2:3;
	} else if(name == "Product-Producer") {
		return isE1toE2 ? 4:5;
	} else if(name == "Content-Container") {
		return isE1toE2 ? 6:7;
	} else if(name == "Entity-Origin") {
		return isE1toE2 ? 8:9;
	} else if(name == "Entity-Destination") {
		return isE1toE2 ? 10:11;
	} else if(name == "Component-Whole") {
		return isE1toE2 ? 12:13;
	} else if(name == "Member-Collection") {
		return isE1toE2 ? 14:15;
	} else if(name == "Message-Topic") {
		return isE1toE2 ? 16:17;
	} else if(name == "Other")
		return OTHER_LABEL;
	else
		assert(0);
}

string relationID2Name(int id) {
	switch(id) {
	case 0:
		return "Cause-Effect(e1,e2)";
	case 1:
		return "Cause-Effect(e2,e1)";
	case 2:
		return "Instrument-Agency(e1,e2)";
	case 3:
		return "Instrument-Agency(e2,e1)";
	case 4:
		return "Product-Producer(e1,e2)";
	case 5:
		return "Product-Producer(e2,e1)";
	case 6:
		return "Content-Container(e1,e2)";
	case 7:
		return "Content-Container(e2,e1)";
	case 8:
		return "Entity-Origin(e1,e2)";
	case 9:
		return "Entity-Origin(e2,e1)";
	case 10:
		return "Entity-Destination(e1,e2)";
	case 11:
		return "Entity-Destination(e2,e1)";
	case 12:
		return "Component-Whole(e1,e2)";
	case 13:
		return "Component-Whole(e2,e1)";
	case 14:
		return "Member-Collection(e1,e2)";
	case 15:
		return "Member-Collection(e2,e1)";
	case 16:
		return "Message-Topic(e1,e2)";
	case 17:
		return "Message-Topic(e2,e1)";
	case 18:
		return "Other";
	default:
		assert(0);
		return "";
	}

}

void outputToSem10rc(const vector<int>& ids, const string& path) {
	ofstream m_outf;
	m_outf.open(path.c_str());
	int startSentID = 8001;
	for(int i=0;i<ids.size();i++) {

		m_outf << startSentID << "\t"<< relationID2Name(ids[i]) <<endl;

		startSentID++;
	}
	m_outf.close();
}

void loadNlpFile(const string& file, vector<Example>& examples) {
	ifstream ifs;
	ifs.open(file.c_str());

	string line;
	Sent sent;
	int count = 0;
	while(getline(ifs, line)) {
		if(line.empty()){
			// new line
			examples[count].sent = sent;
			sent.tokens.clear();
			count++;
		} else {
			vector<string> splitted;
			fox::split_bychar(line, splitted, '\t');
			Token token;
			token.word = splitted[0];
			token.begin = atoi(splitted[1].c_str());
			token.end = atoi(splitted[2].c_str());
			token.pos = splitted[3];
			token.lemma = splitted[4];
			token.depGov = atoi(splitted[5].c_str());
			token.depType = splitted[6];
			token.sst = splitted[7];
			sent.tokens.push_back(token);
		}



	}

	ifs.close();
}


bool isTokenBeforeEntity(const fox::Token& tok, const Entity& entity) {
	if(tok.begin<entity.begin)
		return true;
	else
		return false;
}

bool isTokenAfterEntity(const fox::Token& tok, const Entity& entity) {
	if(tok.end>entity.end)
		return true;
	else
		return false;

}

bool isTokenInEntity(const fox::Token& tok, const Entity& entity) {
	if(tok.begin>=entity.begin && tok.end<=entity.end)
		return true;
	else
		return false;

}



bool isTokenBetweenTwoEntities(const fox::Token& tok, const Entity& former, const Entity& latter) {
	if(tok.begin>=former.end && tok.end<=latter.begin)
		return true;
	else
		return false;
}


int parseSem10rcFile(const string& filePath, vector<Example>& examples)
{
	string labelE1Begin = "<e1>";
	string labelE1End = "</e1>";
	string labelE2Begin = "<e2>";
	string labelE2End = "</e2>";
	string rlabel1 = "(";
	string rlabel2 = ",";

	ifstream ifs;
	ifs.open(filePath.c_str());

	string line;
	Example e;
	int count = 0;
	while(getline(ifs, line)) {

		if(line.empty() || line=="\r") {
			// save the old
			if(!e.text.empty()) {
				examples.push_back(e);
			}
			// a new example is going to start
			count = 0;
		} else if(count==0) {
			// text and entity
			// remove colon
			string temp1 = line.substr(line.find("\t\"")+2);
			string temp = temp1.substr(0, temp1.length()-2);
			// remove annotation label
			int posE1Begin = temp.find(labelE1Begin);
			int posE1End = temp.find(labelE1End);


			e.relation.entity1.text = temp.substr(posE1Begin+labelE1Begin.length(), posE1End-posE1Begin-labelE1Begin.length());
			e.relation.entity1.begin = posE1Begin;
			e.relation.entity1.end = e.relation.entity1.begin+e.relation.entity1.text.length();

			int posE2Begin = temp.find(labelE2Begin);
			int posE2End = temp.find(labelE2End);

			e.relation.entity2.text = temp.substr(posE2Begin+labelE2Begin.length(), posE2End-posE2Begin-labelE2Begin.length());
			e.relation.entity2.begin = posE2Begin-labelE2Begin.length()-labelE2End.length();
			e.relation.entity2.end = e.relation.entity2.begin+e.relation.entity2.text.length();

			string temp2 = temp.substr(0, posE1Begin)+e.relation.entity1.text+
					temp.substr(posE1End+labelE1End.length(), posE2Begin-posE1End-labelE1End.length())+e.relation.entity2.text+
					temp.substr(posE2End+labelE2End.length());

			e.text = temp2;
			//e.text = "#"; // no need to keep the whole sentence

			count++;
		} else if(count==1) {
			// relation
			if(line.find(rlabel1) != -1) {
				string type = line.substr(0, line.find(rlabel1));
				string first = line.substr(line.find(rlabel1)+1, line.find(rlabel2)-line.find(rlabel1)-1);

				if(first == "e1")
					e.relation.isE1toE2 = true;
				else
					e.relation.isE1toE2 = false;

				e.relation.type = type;
			} else {
				e.relation.type = "Other";
			}

			count++;
		}
	}
	ifs.close();



    return 0;

}





#endif /* UTILS_H_ */
