#include "Connection.h"
#include "NeuralNode.h"
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>

using namespace std;

class NeuralNetwork{

	public:
		NeuralNetwork(int nInputs, int nOutputs, int nHiddenLayers, int nodesPerLayer);
		~NeuralNetwork();

		vector<NeuralNode*> getBiasNodes();
		vector<NeuralNode*> getInputLayer();
		vector<NeuralNode*> getOutputLayer();
		vector<NeuralNode*> getLayer(int layerIndex);

	private:
		int m_nInputs;
		int m_nOutputs;
		int m_nHiddenLayers;
		int m_nNodesPerLayer;

		vector<Connection*> m_connections;
		vector<NeuralNode*> m_neuralNodes;
};
