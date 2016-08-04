//Node (neuron) in C++ neural network
//Written by Andrew Johnson on June 10th, 2016
//email: ajohnson@cern.ch


#include NeuralNode.h


//Neuron Constructor for a given layer 
NeuralNode(int layerIndex){
	setBias(false);
	setLayerIndex(layerIndex);
	m_downstreamConnections.clear();
	m_upstreamConnections.clear();
	m_hasResponse = false;
	m_response = 0.0;
	m_responseDerivative = 0.0;
	m_hasDelta = false;
	m_delta = 0.0;
	m_target = 0.0;
	return;
}


int NeuralNode::getLayerIndex(){
	return m_layerIndex;
}

double NeuralNode::getResponse(){

	if(isBiasNode){
		m_response = 1.0;
		m_responseDerivative = 0.0;
	}
	else{
		double inputSum = 0.0;
		for(vector<Connection*>::iterator inputIter = m_upstreamConnections.begin(); inputIter < m_upstreamConnections.end(); inputIter++){
			double inputWeight = (*inputIter)->getWeight();
			double inputResponse = (*inputIter)->getInputNode()->getResponse();
			inputSum += inputWeight*inputResponse;
		}
		m_response = sigmoid(inputSum);
		m_responseDerivative = sigmoidDerivative(inputSum);
		m_hasResponse = true;
	}

	return m_response;
}

double NeuralNode::getResponseDerivative(){
	return m_responseDerivative;
}

double NeuralNode::getDelta(){
	m_delta = 0.0;
	double deriv = getResponseDerivative();
	if(m_hasDelta) return m_delta;
	else if(isOutputNode()){
		m_delta = (m_response - m_target)*deriv;
	}
	else{
		double deltaSum = 0.0;
		for(int i = 0; i < m_downstreamConnections.size(); ++i){
			double theWeight = m_downstreamConnections[i]->getWeight();
			double theDelta = m_downstreamConnections[i]->getOutputNode()->getDelta;
			deltaSum+= theWeight*theDelta;
		}
		m_delta = deltaSum*deriv;
	}
	m_hasDelta = true;

	//Update the downstream weights
	for(vector<Connection*>::iterator connectIter = m_downstreamConnections.begin(); connectIter != m_downstreamConnections.end(); ++connectIter()){
		(*connectIter)->trainWeight();
	}

	return m_delta;
}


void NeuralNode::backPropagation(){
	if(isBiasNode() || isInputNode()) getDelta();
	else{
		for(vector<Connection*>::iterator connectIter = m_upstreamConnections.begin(); connectIter != m_upstreamConnections.end(); ++connectIter){
			(*connectIter)->getInputNode()->backPropagation();
		}
	}
}


void NeuralNode::clearDelta(){
	if(m_hasDelta){
		m_hasDelta = false;
		m_delta = 0.0;
		for(vector<Connection*>::iterator connectIter = m_downstreamConnections.begin(); connectIter != m_downstreamConnections.end(); ++connectIter){
			(*connectIter)->getOutputNode()->clearDelta();
		}
	}
}


void NeuralNode::clearResponse(){
	m_response = 0.0;
	m_responseDerivative = 0.0;
	m_hasResponse = false;
}


double NeuralNode::getNDownstreamConnections(){
	return m_downstreamConnections.size();
}

double NeuralNode::getNUpstreamConnections(){
        return m_upstreamConnections.size();
}

bool isBiasNode(){
	return m_isBiasNode;
}

bool isInputNode(){
	return (isBiasNode() == false && getNUpstreamConnections() == 0);
}

bool isOutputNode(){
	return (isBiasNode() == false && getNDownstreamConnections() == 0);
}

vector<Connection*> getDownstreamConnections(){
	return m_downstreamConnections;
}

vector<Connection*> getUpstreamConnections(){
	return m_upstreamConnections;
}

void addDownstreamConnection(Connection* connection){
	m_downstreamConnections.push_back(connection);
}

void addUpstreamConnection(Connection* connection){
	m_upstreamConnections.push_back(connection);
}

void addDownstreamConnections(vector<Connection*> connections){
	vector<Connection*>::iterator connectionIter;
	for(connectionIter = connections.begin(); connectionIter != connections.end(); connectionIter++){
		addDownstreamConnection(*connectionIter);
	}
}

void addUpstreamConnections(vector<Connection*> connections){
	vector<Connection*>::iterator connectionIter;
	for(connectionIter = connections.begin(); connectionIter != connections.end(); connectionIter++){
		addUpstreamConnection(*connectionIter);
	}
}


void NeuralNode::setBias(bool isBias){
	m_isBiasNode = isBias;
}

void NeuralNode::setLayerIndex(int theLayerIndex){
	m_layerIndex = theLayerIndex;
}

void NeuralNode::setResponse(double theResponse){
	m_response = theResponse;
	m_hasResponse = true;
}

void NeuralNode::setResponseWithSum(double theSum){
	m_response = sigmoid(theSum);
	m_responseDerivative = sigmoidDerivative(theSum);
	m_hasResponse = true;
}

void NeuralNode::setTarget(double theTarget){
	m_target = theTarget;
}

//Sigmoid activation function - each neuron takes a weighted sum of inputs from all neurons in the previous layer and passes them through the activation function
double NeuralNode::sigmoid(double inputSum){
	return 1.0 / (1.0 + exp(-1.0*inputSum));
}

//Derivative of sigmoid function - needed for backpropagation (lol chain rule)
double NeuralNode::sigmoidDerivative(double inputSum){
	return sigmoid(sum) * (1.0 - sigmoid(sum));
}
