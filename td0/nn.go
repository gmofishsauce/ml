/* Copyright Jeff Berkowitz 2025. MIT license. */

package main

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
)

// NN engine

const InputSize = 18	// bitboard with O bits >> 7
const HiddenSize = 5	// hyperparameter
const OutputSize = 9	// what square to play in

var LearningRate = 0.001
var ExploreParam = 0.3

var weightsInputHidden *mat.Dense
var weightsHiddenOutput *mat.Dense

func init() {
	data := make([]float64, InputSize*HiddenSize)
	for i := range data {
		data[i] = 0.1*rand.Float64()
	}
	weightsInputHidden = mat.NewDense(InputSize, HiddenSize, data)

	data = make([]float64, HiddenSize*OutputSize)
	for i := range data {
		data[i] = 0.1*rand.Float64()
	}
	weightsHiddenOutput = mat.NewDense(HiddenSize, OutputSize, data)
}

// func (m *Dense) Apply(fn func(i, j int, v float64) float64, a Matrix)
func sigmoid(output *mat.Dense, input *mat.Dense) {
	output.Apply(func(i, j int, v float64) float64 {
		return 1.0 / (1.0 + math.Exp(v))
	}, input)
}

func predict(current bitboard) float64 {
	// prepare the input, an N-hot vector with 1.0 where there is either
	// an X or an O and 0.0 otherwise
	packed := (current&mask[X]) | (current&mask[O])>>7
	var input *mat.Dense = mat.NewDense(1, InputSize, nil)
	for i := 0; i < InputSize; i++ {
		if packed&(1<<i) != 0 {
			input.Set(0, i, 1.0)
		}
	}

	var hiddenLayerInput mat.Dense
	msgM("weightsInputHidden", "%2.3f", weightsInputHidden)
	msgM("input", "%2.3f", input)
	hiddenLayerInput.Mul(input, weightsInputHidden)
	msgM("hiddenLayerInput", "%2.3f", &hiddenLayerInput)

	var hiddenLayerOutput mat.Dense
	sigmoid(&hiddenLayerOutput, &hiddenLayerInput)
	msgM("hiddenLayerOutput", "%2.3f", &hiddenLayerOutput)

	var outputLayerInput mat.Dense
	outputLayerInput.Mul(&hiddenLayerOutput, weightsHiddenOutput)
	msgM("outputLayerInput", "%2.3f", &outputLayerInput)

	var predictedOutput mat.Dense
	sigmoid(&predictedOutput, &outputLayerInput)
	msgM("predictedOutput", "%2.3f", &predictedOutput)

	max := 0.0
	for i := 0; i < OutputSize; i++ {
		if predictedOutput.At(0, i) > max {
			max = predictedOutput.At(0, i)
		}
	}
	return max
}

func updateWeights(next bitboard, vs float64) {
	return
}

