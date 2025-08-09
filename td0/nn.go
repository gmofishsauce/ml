/* Copyright Jeff Berkowitz 2025. MIT license. */

package main

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
)

// NN engine

type NN struct {
	InputSize  int
	HiddenSize int
	OutputSize int

	learningRate float64

	weightsInputHidden  *mat.Dense
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
		data[i] = 0.1 * rand.Float64()
	}
	nn.weightsInputHidden = mat.NewDense(nn.InputSize, nn.HiddenSize, data)

	data = make([]float64, nn.HiddenSize*nn.OutputSize)
	for i := range data {
		data[i] = 0.1 * rand.Float64()
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
func mulScalar(s float64, b *mat.Dense) *mat.Dense {
	r, c := b.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			b.Set(i, j, s*b.At(i, j))
		}
	}
	return b
}

// Return a new mat.Dense that is the elementwise product
func mulElements(a *mat.Dense, b *mat.Dense) *mat.Dense {
	r, c := a.Dims()
	var result *mat.Dense = mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, a.At(i, j)*b.At(i, j))
		}
	}
	return result
}

// Return a new mat.Dense that is the product of the arguments
func matmul(a mat.Matrix, b mat.Matrix) *mat.Dense {
	var result mat.Dense
	result.Mul(a, b)
	return &result
}

// Return a new mat.Dense that is the elementwise sum of the arguments.
func add(a *mat.Dense, b *mat.Dense) *mat.Dense {
	var result mat.Dense
	result.Add(a, b)
	return &result
}

func sub(a *mat.Dense, b *mat.Dense) *mat.Dense {
	var result mat.Dense
	result.Sub(a, b)
	return &result
}

func matFromSlice(in []float64) *mat.Dense {
	return mat.NewDense(1, len(in), in)
}

func sliceFromMat(in *mat.Dense) []float64 {
	return mat.Row(nil, 0, in)
	/*
		r, c := in.Dims()
		result := make([]float64, c)
		for j := 0; j < c; j++ {
			result[j] = predictedOutput.At(0, j)
		}
		return result
	*/
}

// Forward pass (predict) - internal interface taking mat.Dense pointers
func (nn *NN) matPredict(input *mat.Dense) (hidden *mat.Dense, predicted *mat.Dense) {
	hiddenLayerOutput := sigmoid(matmul(input, nn.weightsInputHidden))
	predictedOutput := sigmoid(matmul(hiddenLayerOutput, nn.weightsHiddenOutput))
	return hiddenLayerOutput, predictedOutput
}

// External interface - evaluate the network with the given vector of inputs.
// The input slice of float64 must have the dimension InputSize
// The result is a slice of float64 of dimension OutputSize.
func (nn *NN) Predict(in []float64) (predicted []float64) {
	input := matFromSlice(in)
	r, c := input.Dims()
	if r != 1 || c != nn.InputSize {
		panic("Predict(): assertion 1")
	}

	// The internal interface returns the result of the hidden layer in
	// addition to the predicted output for use in training. This external
	// interface just returns the predicted output.
	_, predictedMat := nn.matPredict(input)
	return sliceFromMat(predictedMat)
}

// Back propagate the difference between the input and the goal
func (nn *NN) Learn(in []float64, goal []float64) {
	if len(in) != nn.InputSize {
		panic("Learn(): assertion 1")
	}
	input := matFromSlice(in)

	hiddenOutput, output := nn.matPredict(input)
	_, c := output.Dims()
	if len(goal) != c {
		panic("Learn(): assertion 2")
	}

	outputError := sub(matFromSlice(goal), output)
	outputDelta := mulElements(outputError, sigmoidDerivative(output))

	hiddenError := matmul(outputDelta, nn.weightsHiddenOutput.T())
	hiddenDelta := mulElements(hiddenError, sigmoidDerivative(hiddenOutput))

	hiddenOutputAdjust := matmul(hiddenOutput.T(), outputDelta)
	inputHiddenAdjust := matmul(input.T(), hiddenDelta)

	nn.weightsHiddenOutput = add(nn.weightsHiddenOutput, mulScalar(nn.learningRate, hiddenOutputAdjust))
	nn.weightsInputHidden = add(nn.weightsInputHidden, mulScalar(nn.learningRate, inputHiddenAdjust))
}
