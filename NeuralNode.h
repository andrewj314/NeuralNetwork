//Header file for NeuralNode class
//Andrew Johnson, June 10th, 2016
//ajohnson@cern.ch
//
//
include <vector>
include <string>
include <iostream>
include <fstream>
include <math.h>
include <stdio.h>
include <stdlib.h>
include <Connection.h>

using namespace std;

class NeuralNode{

	public:
		//Create a new neuron in a given layer
		NeuralNode(int layerIndex);
		//Neuron destructor
		~NeuralNode();

		int getLayerIndex();
		double getResponse();
		double getResponseDerivative();
		double getDelta();
		double getNDownstreamConnections();
		double getNUpstreamConnections();
		bool isInputNode();
		bool isOutputNode();
		bool isBiasNode();
		vector<Connection*> getDownstreamConnections();
		vector<Connection*> getUpstreamConnections();
	        

		void addDownstreamConnection(Connection* connection);
		void addUpstreamConnection(Connection* connection);
		void addDownstreamConnections(vector<Connection*> connections);
		void addUpstreamConnections(vector<Connection*> connections);

		void backProp();
		void clearDelta();
		void clearResponse();

		void setBias(bool isBias);
		void setLayerIndex(int theLayerIndex);
		//CONSIDER USING TEMPLATES FOR THESE
		void setResponse(double theResponse);
		void setResponseWithSum(double theSum); 	
		void setTarget(double theTarget);


	private:
		double sigmoidDerivative(double inputSum);
		double sigmoid(double inputSum);

		vector<Connection*> m_downstreamConnections;
		vector<Connection*> m_upstreamConnections;

		int m_layerIndex;
		bool m_isBiasNode;
		bool m_hasResponse;
		double m_response;
		double m_responseDerivative;

		bool m_hasDelta;
		double m_delta;
		double m_target;

		


};
