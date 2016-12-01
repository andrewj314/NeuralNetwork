OBJS = Connection.o NeuralNode.o NeuralNetwork.o
CC = g++
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG) 
LFLAGS = -Wall $(DEBUG)

NN : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o NN

Connection.o : Connection.h Connection.cpp NeuralNode.h
	$(CC) $(CFLAGS) Connection.cpp

NeuralNode.o : NeuralNode.h NeuralNode.cpp Connection.h
	$(CC) $(CFLAGS) NeuralNode.cpp

NeuralNetwork.o : NeuralNetwork.h NeuralNetwork.cpp NeuralNode.h Connection.h 
	$(CC) $(CFLAGS) NeuralNetwork.cpp


clean:
	\rm *.o *~ NN

