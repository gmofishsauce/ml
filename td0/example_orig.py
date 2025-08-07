import numpy as np

class NeuralNetwork():
    """
    A simple 3-layer neural network.

    This network has 3 input neurons, a hidden layer with 4 neurons,
    and an output layer with 1 neuron. It uses the sigmoid activation
    function.
    """
    def __init__(self):
        # Seed the random number generator for consistency.
        # This helps in getting the same results every time the script is run.
        np.random.seed(1)

        # Initialize weights with random values in the range -1 to 1.
        # synapse_0 connects the input layer (3 neurons) to the hidden layer (4 neurons).
        self.synapse_0 = 2 * np.random.random((3, 4)) - 1
        
        # synapse_1 connects the hidden layer (4 neurons) to the output layer (1 neuron).
        self.synapse_1 = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        """
        The sigmoid activation function. It squashes values to a range between 0 and 1.
        Calculates: 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function.
        This indicates the slope of the sigmoid curve, which tells us how
        confident the network is about an existing weight.
        """
        return np.multiply(x, (1 - x))

    def predict(self, inputs):
        """
        This is the forward propagation or "thinking" part of the network.
        It passes inputs through the network to generate an output.
        """
        # --- Input Handling ---
        # Ensure the input is a 2D array (matrix) for matmul.
        # If the input is a 1D vector, we reshape it into a 1xN matrix.
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        # --- Forward Propagation ---
        # Pass the inputs through the first layer of synapses.
        layer_1_input = np.matmul(inputs, self.synapse_0)
        # Apply the sigmoid activation function.
        layer_1_output = self.sigmoid(layer_1_input)
        
        # Pass the output of the hidden layer through the second layer of synapses.
        layer_2_input = np.matmul(layer_1_output, self.synapse_1)
        # Apply the sigmoid activation function to get the final output.
        layer_2_output = self.sigmoid(layer_2_input)
        
        return layer_1_output, layer_2_output

    def train(self, training_set_inputs, training_set_outputs, num_iterations, learning_rate=1.0):
        """
        Trains the neural network by adjusting the synaptic weights.
        """
        print(f"Starting training for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            # --- Forward Propagation ---
            layer_1, layer_2 = self.predict(training_set_inputs)

            # --- Backpropagation ---
            # This is where the network learns by correcting its errors.

            # 1. Calculate the error for the output layer (layer_2).
            # This is the difference between the desired output and the predicted output.
            layer_2_error = training_set_outputs - layer_2
            
            if (iteration % 10000) == 0:
                print(f"  Error after {iteration} iterations: {str(np.mean(np.abs(layer_2_error)))}")

            # 2. Calculate the "delta" for the output layer.
            # We multiply the error by the derivative of the sigmoid at layer_2's output.
            # This "weights" the error, reducing the update for confident predictions.
            # We use np.multiply() for explicit element-wise multiplication.
            layer_2_delta = np.multiply(layer_2_error, self.sigmoid_derivative(layer_2))

            # 3. Calculate the error for the hidden layer (layer_1).
            # We determine how much layer_1 contributed to layer_2's error by
            # propagating the error backward through the synapse_1 weights.
            layer_1_error = np.matmul(layer_2_delta, self.synapse_1.T)

            # 4. Calculate the "delta" for the hidden layer.
            layer_1_delta = np.multiply(layer_1_error, self.sigmoid_derivative(layer_1))

            # --- Weight Update ---
            # 5. Calculate the necessary adjustments for the weights.
            # We are using the outputs from the previous layer to determine the
            # weight adjustment.
            synapse_1_adjustment = np.matmul(layer_1.T, layer_2_delta)
            synapse_0_adjustment = np.matmul(training_set_inputs.T, layer_1_delta)

            # 6. Update the synaptic weights with the adjustments, scaled by the learning rate.
            self.synapse_1 += learning_rate * synapse_1_adjustment
            self.synapse_0 += learning_rate * synapse_0_adjustment
            
        print("Training complete.")


if __name__ == "__main__":

    # Create an instance of our neural network
    neural_network = NeuralNetwork()

    print("Initial random synaptic weights (synapse_0):")
    print(neural_network.synapse_0)
    print("\nInitial random synaptic weights (synapse_1):")
    print(neural_network.synapse_1)
    print("-" * 30)

    # --- Training Data ---
    # Input dataset (4 examples, 3 features each)
    training_inputs = np.array([[0, 0, 1],
                                [0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

    # Output dataset (4 examples, 1 output each)
    # The .T transposes this to be a 4x1 matrix.
    training_outputs = np.array([[0, 1, 1, 0]]).T

    # Train the neural network using the training data.
    # We iterate 60,000 times and use a learning rate of 0.5.
    neural_network.train(training_inputs, training_outputs, num_iterations=60000, learning_rate=0.5)
    print("-" * 30)

    print("Synaptic weights after training (synapse_0):")
    print(neural_network.synapse_0)
    print("\nSynaptic weights after training (synapse_1):")
    print(neural_network.synapse_1)
    print("-" * 30)

    # --- Testing the network with a 1D vector ---
    # This demonstrates that the network can handle a single new situation.
    print("Testing with a new situation [1, 0, 0] -> ?")
    
    # Create the 1D input vector
    new_input = np.array([1, 0, 0])
    
    # Get the prediction
    hidden_state, output = neural_network.predict(new_input)
    
    print("Prediction:")
    print(output)


