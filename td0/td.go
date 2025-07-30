/* Copyright Jeff Berkowitz 2025. MIT license. */

package main

import (
	"fmt"
	"os"
)

// Tic-tac-toe game with TD(0) reinforcement learning
//
// Board state. Board is number 1..9 from upper left
// to lower right. Positions are stored in a uint32
// sort "one hot": bits 1..9 are X moves, 10..19 are
// O moves, and neither is an empty square.
type bitboard uint32
var current bitboard

const X = 0
const O = 1
const NumGames = 2

var shift[2] int = [2]int{1, 10}
var mask[2] bitboard  = [2]bitboard{0b111111111<<shift[X], 0b111111111<<shift[O]}

func main() {
	msg("Firing up...")

	for i := 0; i < NumGames; i++ {
		current = 0

		for player := X; !isFinal(current); player = 1-player {
			move, estimate := chooseMove(player)
			updateWeights(move, estimate)
			current = current|move
		}
	}

	fmt.Fprintln(os.Stderr, "done")
}

var unshiftedWinningPositions = []bitboard {
	0b000000111, 0b000111000, 0b111000000,
	0b001001001, 0b010010010, 0b100100100,
	0b100010001, 0b001010100,
}

// Return true if no more plays are available or if there is a winner
func isFinal(position bitboard) bool {
	filledSquares := (current&mask[O]>>shift[O]) | (current&mask[X]>>shift[X])
	if filledSquares == 0b111111111 {
		return true // all squares filled - draw
	}
	for _, winner := range(unshiftedWinningPositions) {
		if position&mask[X] == winner<<shift[X] || position&mask[O] == winner<<shift[O] {
			return true
		}
	}
	return false
}

// Return true if the move is not on top of a previous move
func isLegal(move bitboard) bool {
	filledSquares := (current&mask[O]>>shift[O]) | (current&mask[X]>>shift[X])
	return move&filledSquares == 0
}

// Choose the next move. This function only chooses legal moves.
func chooseMove(player int) (bitboard, float64) {
	var result bitboard
	var candidate bitboard
	max := -100.0

	for i := 0; i < 9; i++ {
		candidate = 1<<i
		if isLegal(candidate) {
			value := eval(current|candidate)
			if value > max {
				result = candidate
				max = value
			}
		}
	}
	if result == 0 {
		panic("no move selection")
	}
	return result, max
}

func updateWeights(next bitboard, vs float64) {
	return
}

func eval(position bitboard) float64 {
	return 0.5;
}

func msg(format string, a ...any) {
	if len(format) == 0 {
		msg("internal error: msg() called with empy message")
		return
	}
	if format[len(format)-1] != '\n' {
		format += "\n"
	}
    fmt.Fprintf(os.Stderr, format, a...)
}

