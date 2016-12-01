#include "NeuralNetwork.h"

using namespace std;

NeuralNetwork::NeuralNetwork(int nInputs, int nOutputs, int nHiddenLayers, int nNodesPerLayer){
	m_nInputs = nInputs;
	m_nOutputs = nOutputs;
	m_nHiddenLayers = nHiddenLayers;
	m_nNodesPerLayer = nNodesPerLayer;
	m_connections.clear();
	m_neuralNodes.clear();

	addLayer(0, m_nInputs, "linear");

	for(int i = 1; i <= nHiddenLayers; ++i){
		addLayer(i, m_nNodesPerLayer, "sigmoid");
	}

	addLayer(m_nHiddenLayers+1, m_nOutputs, "linear");
}

void NeuralNetwork::addLayer(int layerIndex, int numLayerNodes, string function){
	for(int i = 0; i < numLayerNodes; ++i){
		NeuralNode *theNode = new NeuralNode(layerIndex, function);
		if(layerIndex > 0){
			vector<NeuralNode*> prevLayer = getLayer(layerIndex-1);
			for(vector<NeuralNode*>::iterator iter = prevLayer.begin(); iter != prevLayer.end(); ++iter){
				Connection *theConnection = new Connection(1.0, *iter, theNode);
				m_connections.push_back(theConnection);
			}
		}
		m_neuralNodes.push_back(theNode);
	}

	if(layerIndex > 0 && layerIndex < m_nHiddenLayers+1){
		NeuralNode *theBiasNode = new NeuralNode(layerIndex, function);
		m_neuralNodes.push_back(theBiasNode);
	}

}


void NeuralNetwork::clearNetworkResponse(){
	for(vector<NeuralNode*>::iterator iter = m_neuralNodes.begin(); iter != m_neuralNodes.end(); ++iter){
		(*iter)->clearResponse();
	}
}


vector<NeuralNode*> NeuralNetwork::getBiasNodes(){
	vector<NeuralNode*> biasNodes; 
	biasNodes.clear();
	for(vector<NeuralNode*>::iterator iter = m_neuralNodes.begin(); iter != m_neuralNodes.end(); ++iter){
		if((*iter)->isBiasNode()) biasNodes.push_back(*iter);
	}
	return biasNodes;
}


vector<NeuralNode*> NeuralNetwork::getInputLayer(){
	return getLayer(0);
}


vector<NeuralNode*> NeuralNetwork::getOutputLayer(){
	return getLayer(m_nHiddenLayers+1);
}


vector<NeuralNode*> NeuralNetwork::getLayer(int layerIndex){
	vector<NeuralNode*> theLayer;
	theLayer.clear();

	for(vector<NeuralNode*>::iterator iter = m_neuralNodes.begin(); iter != m_neuralNodes.end(); ++iter){
		if((*iter)->getLayerIndex() == layerIndex) theLayer.push_back(*iter);
	}
	
	return theLayer;
}


void NeuralNetwork::randomizeWeights(){
	for(vector<Connection*>::iterator iter = m_connections.begin(); iter != m_connections.end(); ++iter){
		int randomInt = rand()%10;
		double randomDouble = (double)(randomInt - 5.0)/10.0;
		(*iter)->setWeight(randomDouble);
	}
}


void NeuralNetwork::setNetworkLearningRate(double theRate){
	for(vector<Connection*>::iterator iter = m_connections.begin(); iter != m_connections.end(); ++iter){
		(*iter)->setLearningRate(theRate);
	}
}


void NeuralNetwork::setTargets(vector<double> targets){
	if(targets.size() != m_nOutputs){
		cout << "ERROR: wrong number of targets\n";
	}
	vector<NeuralNode*> outputs = getOutputLayer();
	for(int i = 0; i < m_nOutputs; ++i){
		outputs[i]->setTarget(targets[i]);
	}

}

//Still need getResponse, BP
