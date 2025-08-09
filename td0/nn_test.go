/* These tests were written by GPT4.1 running in VSCode in August 2025 */

package main

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func floatEquals(a, b float64, tol float64) bool {
	return math.Abs(a-b) < tol
}

func TestSigmoid(t *testing.T) {
	input := mat.NewDense(1, 2, []float64{0, 1})
	result := sigmoid(input)
	expected := []float64{
		1.0 / (1.0 + math.Exp(0)),  // sigmoid(0) = 0.5
		1.0 / (1.0 + math.Exp(-1)), // sigmoid(1) ≈ 0.7310586
	}
	for i, v := range mat.Row(nil, 0, result) {
		if !floatEquals(v, expected[i], 1e-6) {
			t.Errorf("sigmoid: got %v, want %v", v, expected[i])
		}
	}
}

func TestSigmoidDerivative(t *testing.T) {
	input := mat.NewDense(1, 2, []float64{0.5, 0.8})
	result := sigmoidDerivative(input)
	expected := []float64{0.5 * (1 - 0.5), 0.8 * (1 - 0.8)}
	for i, v := range mat.Row(nil, 0, result) {
		if !floatEquals(v, expected[i], 1e-6) {
			t.Errorf("sigmoidDerivative: got %v, want %v", v, expected[i])
		}
	}
}

func TestMulScalar(t *testing.T) {
	input := mat.NewDense(1, 2, []float64{2, 3})
	result := mulScalar(2, input)
	expected := []float64{4, 6}
	for i, v := range mat.Row(nil, 0, result) {
		if v != expected[i] {
			t.Errorf("mulScalar: got %v, want %v", v, expected[i])
		}
	}
}

func TestMulElements(t *testing.T) {
	a := mat.NewDense(1, 2, []float64{2, 3})
	b := mat.NewDense(1, 2, []float64{4, 5})
	result := mulElements(a, b)
	expected := []float64{8, 15}
	for i, v := range mat.Row(nil, 0, result) {
		if v != expected[i] {
			t.Errorf("mulElements: got %v, want %v", v, expected[i])
		}
	}
}

func TestMatmul(t *testing.T) {
	a := mat.NewDense(1, 2, []float64{1, 2})
	b := mat.NewDense(2, 1, []float64{3, 4})
	result := matmul(a, b)
	if result.At(0, 0) != 11 {
		t.Errorf("matmul: got %v, want 11", result.At(0, 0))
	}
}

func TestAdd(t *testing.T) {
	a := mat.NewDense(1, 2, []float64{1, 2})
	b := mat.NewDense(1, 2, []float64{3, 4})
	result := add(a, b)
	expected := []float64{4, 6}
	for i, v := range mat.Row(nil, 0, result) {
		if v != expected[i] {
			t.Errorf("add: got %v, want %v", v, expected[i])
		}
	}
}

func TestSub(t *testing.T) {
	a := mat.NewDense(1, 2, []float64{5, 7})
	b := mat.NewDense(1, 2, []float64{2, 3})
	result := sub(a, b)
	expected := []float64{3, 4}
	for i, v := range mat.Row(nil, 0, result) {
		if v != expected[i] {
			t.Errorf("sub: got %v, want %v", v, expected[i])
		}
	}
}

func TestMatFromSliceAndSliceFromMat(t *testing.T) {
	in := []float64{1, 2, 3}
	matVal := matFromSlice(in)
	out := sliceFromMat(matVal)
	for i := range in {
		if in[i] != out[i] {
			t.Errorf("matFromSlice/sliceFromMat: got %v, want %v", out[i], in[i])
		}
	}
}

func TestNNPredictAndLearn(t *testing.T) {
	inputSize := 3
	hiddenSize := 4
	outputSize := 1
	nn := makeNN(inputSize, hiddenSize, outputSize)

	input := []float64{0.5, -0.2, 0.1}
	goal := []float64{0.8}

	initialPred := nn.Predict(input)[0]
	initialErr := math.Abs(initialPred - goal[0])

	var finalPred float64
	var finalErr float64
	steps := 10000
	logInterval := 200
	for i := 0; i < steps; i++ {
		nn.Learn(input, goal)
		if (i+1)%logInterval == 0 || i == 0 {
			pred := nn.Predict(input)[0]
			err := math.Abs(pred - goal[0])
			t.Logf("Step %d: prediction %.4f, error %.4f", i+1, pred, err)
		}
		if i == steps-1 {
			finalPred = nn.Predict(input)[0]
			finalErr = math.Abs(finalPred - goal[0])
		}
	}

	if math.IsNaN(finalPred) || math.IsInf(finalPred, 0) {
		t.Errorf("Predict after Learn: got invalid value %v", finalPred)
	}

	if finalErr >= initialErr {
		t.Errorf("Network did not converge: initial error %.4f, final error %.4f", initialErr, finalErr)
	}

	t.Logf("Initial prediction: %.4f, error: %.4f", initialPred, initialErr)
	t.Logf("Final prediction: %.4f, error: %.4f", finalPred, finalErr)
}
