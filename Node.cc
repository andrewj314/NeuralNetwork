//Node (neuron) in C++ neural network
//Written by Andrew Johnson on June 10th, 2016
//email: ajohnson@cern.ch


#include NeuralNode.h






//Sigmoid activation function - each neuron takes a weighted sum of inputs from all neurons in the previous layer and passes them through the activation function
double NeuralNode::sigmoid(double inputSum){
	return 1.0 / (1.0 + exp(-1.0*inputSum));
}

//Derivative of sigmoid function - needed for backpropagation (lol chain rule)
double NeuralNode::sigmoidDerivative(double inputSum){
	return sigmoid(sum) * (1.0 - sigmoid(sum));
}
