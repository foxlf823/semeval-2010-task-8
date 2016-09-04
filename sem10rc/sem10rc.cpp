/*
 * sem10rc.cpp
 *
 *  Created on: Mar 9, 2016
 *      Author: fox
 */
#include <iostream>
#include <string>
#include "Argument_helper.h"
#include "Options.h"
#include "Tool.h"
#include "N3L.h"
#include "WordNet.h"
#include "NNsem10rc_rnn.h"


using namespace std;


int main(int argc, char **argv)
{
#if USE_CUDA==1
  InitTensorEngine();
#else
  InitTensorEngine<cpu>();
#endif



	string optionFile;
	string trainFile;
	string devFile;
	string testFile;
	string trainNlpFile;
	string devNlpFile;
	string testNlpFile;
	string output;

	dsr::Argument_helper ah;
	ah.new_named_string("train", "", "", "", trainFile);
	ah.new_named_string("dev", "", "", "", devFile);
	ah.new_named_string("test", "", "", "", testFile);
	ah.new_named_string("option", "", "", "", optionFile);
	ah.new_named_string("trainnlp", "", "", "", trainNlpFile);
	ah.new_named_string("devnlp", "", "", "", devNlpFile);
	ah.new_named_string("testnlp", "", "", "", testNlpFile);
	ah.new_named_string("output", "", "", "", output);
	ah.process(argc, argv);
	cout<<"train file: " <<trainFile <<endl;
	cout<<"dev file: "<<devFile<<endl;
	cout<<"test file: "<<testFile<<endl;
	cout<<"trainnlp file: "<<trainNlpFile<<endl;
	cout<<"devnlp file: "<<devNlpFile<<endl;
	cout<<"testnlp file: "<<testNlpFile<<endl;
	cout<<"option file: "<<optionFile<<endl;

	Options options;
	options.load(optionFile);

	if(!output.empty())
		options.output = output;

	options.showOptions();


/*	if((options.channelMode & 2) == 2) {
		if(wninit()) {
			cout<<"warning: can't init wordnet"<<endl;
			exit(0);
		}
	}*/

	Tool tool(options);


	NNsem10rc_rnn nnrc(options);

	nnrc.train(trainFile, devFile, testFile, tool, trainNlpFile, devNlpFile, testNlpFile);

#if USE_CUDA==1
  ShutdownTensorEngine();
#else
  ShutdownTensorEngine<cpu>();
#endif

	return 0;
}


