#include "Connection.h"

using namespace std;

Connection::Connection(double weight, NeuralNode* inputNode, NeuralNode* outputNode){
	setLearningRate(0.1);
	setWeight(m_weight);
	setInputNode(inputNode);
	setOutputNode(outputNode);
	return;
}


double Connection::getLearningRate(){
	return m_rate;
}

double Connection::getWeight(){
	return m_weight;
}

NeuralNode* Connection::getInputNode(){
	return m_inputNode;
}

NeuralNode* Connection::getOutputNode(){
	return m_outputNode;
}

void Connection::setLearningRate(double theLearningRate){
	m_rate = theLearningRate;
}

void Connection::setWeight(double theWeight){
	m_weight = theWeight;
}

void Connection::setInputNode(NeuralNode* theInputNode){
	m_inputNode = theInputNode;
}

void Connection::setOutputNode(NeuralNode* theOutputNode){
	m_outputNode = theOutputNode;
}

double Connection::trainWeight(){
	double theSumOfInputs = m_inputNode->getResponse();
	double theDelta = m_outputNode->getDelta();
	double updateWeight = -1.0*m_rate*theSumOfInputs*theDelta;
	double theWeight = m_weight + updateWeight;
	return theWeight;
	
}


