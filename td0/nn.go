/* Copyright Jeff Berkowitz 2025. MIT license. */

package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
)

// NN engine

type NN struct {
	InputSize int
	HiddenSize int
	OutputSize int

	learningRate float64

	weightsInputHidden *mat.Dense
	weightsHiddenOutput *mat.Dense
}

func makeNN(inputSize int, hiddenSize int, outputSize int) *NN {
	nn := &NN{}
	nn.InputSize = inputSize
	nn.HiddenSize = hiddenSize
	nn.OutputSize = outputSize
	nn.learningRate = 0.001

	data := make([]float64, nn.InputSize*nn.HiddenSize)
	for i := range data {
		data[i] = 0.1*rand.Float64()
	}
	nn.weightsInputHidden = mat.NewDense(nn.InputSize, nn.HiddenSize, data)

	data = make([]float64, nn.HiddenSize*nn.OutputSize)
	for i := range data {
		data[i] = 0.1*rand.Float64()
	}
	nn.weightsHiddenOutput = mat.NewDense(nn.HiddenSize, nn.OutputSize, data)

	return nn
}

// Return a new mat.Dense containing sigmoid(argument)
func sigmoid(input *mat.Dense) *mat.Dense {
	var result mat.Dense
	result.Apply(func(i, j int, v float64) float64 {
		return 1.0 / (1.0 + math.Exp(v))
	}, input)
	return &result
}

// Return a new mat.Dense containing sigmoid'(argument)
func sigmoidDerivative(input *mat.Dense) *mat.Dense {
	var result mat.Dense
	result.Apply(func(i, j int, v float64) float64 {
        return v * (1.0 - v)
	}, input)
	return &result
}

// Return a new mat.Dense that is the elementwise product
func mulScalar(s float64, m *mat.Dense) *mat.Dense {
	var result mat.Dense
	result.Apply(func(i, j int, v float64) float64 {
		return s * m.At(i, j)
	}, m)
	return &result
}

// Return a new mat.Dense that is the product of the arguments
func mult(a mat.Matrix, b mat.Matrix) *mat.Dense {
	var result mat.Dense
	result.Mul(a, b)
	return &result
}

// Return a new mat.Dense that is the elementwise sum of the arguments.
// TODO figure out if this is necessary, i.e. a.Add(a, b) really fails.
func add(a *mat.Dense, b *mat.Dense) *mat.Dense {
	var result mat.Dense
	result.Add(a, b)
	return &result
}

// Evaluate the network with the given vector of inputs.
// The input slice of float64 must have the dimension InputSize
// The result is a slice of float64 of dimension OutputSize.
func (nn *NN) Predict(in []float64) []float64 {
	var input *mat.Dense = mat.NewDense(1, nn.InputSize, in)
	hiddenLayerOutput := sigmoid(mult(input, nn.weightsInputHidden))
	predictedOutput := sigmoid(mult(hiddenLayerOutput, nn.weightsInputHidden))

	r, c := predictedOutput.Dims()
	if r != 1 || c != nn.OutputSize {
		panic(fmt.Sprintf("Predict: result should be 1x%d, but is %dx%d\n", nn.OutputSize, r, c))
	}
	result := make([]float64, nn.OutputSize)
	for j := 0; j < nn.OutputSize; j++ {
		result[j] = predictedOutput.At(0, j)
	}
	return result
}

/*
    # --- Forward Propagation ---
    layer_0 = X                   # Input layer
    layer_1 = sigmoid(np.dot(layer_0, synapse_0)) # Hidden layer's output
    layer_2 = sigmoid(np.dot(layer_1, synapse_1)) # Output layer's output

    # --- Backpropagation ---
    layer_2_error = y - layer_2 # mOutputError = goal - predictedOutput
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2) # outputDelta.Mul(mOutputError, mOutput)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # --- Weight Update ---
    synapse_1 += learningRate * layer_1.T.dot(layer_2_delta)
    synapse_0 += learningRate * layer_0.T.dot(layer_1_delta)
*/

func (nn *NN) Learn(in []float64, goal []float64) {
	if len(in) != nn.InputSize {
		panic("Learn(): assertion 1")
	}
	var input *mat.Dense = mat.NewDense(1, len(in), in)
	hiddenLayerOutput := sigmoid(mult(input, nn.weightsInputHidden))
	predictedOutput := sigmoid(mult(hiddenLayerOutput, nn.weightsHiddenOutput))

	_, c := predictedOutput.Dims()
	if len(goal) != c {
		panic("Learn(): assertion 2")
	}

	outErr := make([]float64, c)
	for j := 0; j < c; j++ {
		outErr[j] = goal[j] - predictedOutput.At(0, j)
	}
	outputError := mat.NewDense(1, c, outErr)
	outputDelta := mult(outputError, sigmoidDerivative(predictedOutput))
	hiddenError := mult(outputDelta, nn.weightsHiddenOutput.T())
	hiddenDelta := mult(hiddenError, sigmoidDerivative(hiddenLayerOutput))

	nn.weightsHiddenOutput = add(nn.weightsHiddenOutput, mulScalar(nn.learningRate, mult(hiddenLayerOutput.T(), outputDelta)))
	nn.weightsInputHidden = add(nn.weightsInputHidden, mulScalar(nn.learningRate, mult(input.T(), hiddenDelta)))
}
