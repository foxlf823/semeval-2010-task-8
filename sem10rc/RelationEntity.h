/*
 * RelationEntity.h
 *
 *  Created on: Mar 9, 2016
 *      Author: fox
 */

#ifndef RELATIONENTITY_H_
#define RELATIONENTITY_H_

#include <string>

using namespace std;

class RelationEntity {

public:
	  string type;
	  Entity entity1;
	  Entity entity2;
	  bool isE1toE2;

	  Entity& getFormer() {
		return entity1;;
	}

	  Entity& getLatter() {
		return entity2;
	}



};



#endif /* RELATIONENTITY_H_ */
