#ifndef Connection_h
#define Connection_h


#include "NeuralNode.h"
#include <stdlib.h>
#include <stdio.h>


using namespace std;

class NeuralNode;

class Connection{

	public:

		Connection(double weight, NeuralNode* inputNode, NeuralNode* outputNode);
		~Connection();

		double getLearningRate();
		NeuralNode* getInputNode();
		NeuralNode* getOutputNode();
		double getWeight();

		void setLearningRate(double theLearningRate);
		void setInputNode(NeuralNode* theInputNode);
		void setOutputNode(NeuralNode* theOutputNode);
		void setWeight(double theWeight);
		double trainWeight();

	private:

		double m_rate;
		double m_weight;
		NeuralNode* m_inputNode;
		NeuralNode* m_outputNode;


};


#endif
