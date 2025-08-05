/* Copyright Jeff Berkowitz 2025. MIT license. */

package main

import (
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

func sigmoid(output *mat.Dense, input *mat.Dense) {
	output.Apply(func(i, j int, v float64) float64 {
		return 1.0 / (1.0 + math.Exp(v))
	}, input)
}

func sigmoidDerivative(output *mat.Dense, input *mat.Dense) {
	output.Apply(func(i, j int, v float64) float64 {
        return v * (1.0 - v)
	}, input)
}

// Evaluate the network with the given vector of inputs.
// Return the max or min value depending on the boolean argument.
// The input vector must have the dimension InputSize
func (nn *NN) Predict(in []float64, getMax bool) float64 {
	var hiddenLayerInput mat.Dense
	var hiddenLayerOutput mat.Dense

	var input *mat.Dense = mat.NewDense(1, nn.InputSize, in)
	hiddenLayerInput.Mul(input, nn.weightsInputHidden)
	sigmoid(&hiddenLayerOutput, &hiddenLayerInput)

	var outputLayerInput mat.Dense
	var predictedOutput mat.Dense

	outputLayerInput.Mul(&hiddenLayerOutput, nn.weightsHiddenOutput)
	sigmoid(&predictedOutput, &outputLayerInput)

	// Now return the max or min of the output vector
	var result float64
	if getMax {
		result = -1e37
		for i := 0; i < nn.OutputSize; i++ {
			if predictedOutput.At(0, i) > result {
				result = predictedOutput.At(0, i)
			}
		}
	} else {
		result = 1e37
		for i := 0; i < nn.OutputSize; i++ {
			if predictedOutput.At(0, i) < result {
				result = predictedOutput.At(0, i)		
			}
		}
	}
	return result
}

/*
func (nn *NN) Learn(in []float64, getMax bool, label float64) {
	predicted_output := nn.Predict(in, getMax)
	output_error := label - predicted_output
	output_delta := output_error * self.sigmoid_derivative(predicted_output)
}
*/

// XXX implement training
func (nn *NN) Learn() {
}

