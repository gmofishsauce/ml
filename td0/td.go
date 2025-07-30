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
const ALL = 0b111111111

// Old FORTRAN trick
func other(player int) int {
	return 1 - player
}

var shift[2] int = [2]int{0, 16}
var mask[2] bitboard  = [2]bitboard{ALL<<shift[X], ALL<<shift[O]}
var name[2] string = [2]string{"X", "O"}

func main() {
	msg("Firing up...")

	for i := 0; i < NumGames; i++ {
		current = 0

		for player := X; !isFinal(current); player = other(player) {
			move, estimate := chooseMove(player)
			updateWeights(move, estimate)
			current = current|move
			display()
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
	filledSquares := (current&mask[O])>>shift[O] | (current&mask[X])>>shift[X]
	if filledSquares == ALL {
		msg("draw")
		return true // all squares filled - draw
	}
	for _, winner := range(unshiftedWinningPositions) {
		if position&mask[X] == winner<<shift[X] {
			msg("X win")
			return true
		}
		if position&mask[O] == winner<<shift[O] {
			msg("O win")
			return true
		}
	}
	return false
}

// Choose the next move. This function only chooses legal moves.
func chooseMove(player int) (bitboard, float64) {
	var result bitboard
	max := -100.0

	for i := 0; i < 9; i++ {
		candidate := bitboard((1<<i)<<shift[player])
		blocker := bitboard((1<<i)<<shift[other(player)])
		value := 0.0
		if candidate&current == 0 && blocker&current == 0 { // legal
			value = eval(current|candidate)
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
		fmt.Fprintln(os.Stderr, "")
		return
	}
	if format[len(format)-1] != '\n' {
		format += "\n"
	}
    fmt.Fprintf(os.Stderr, format, a...)
}

func mark(pos int) string {
	if current&(1<<(shift[X]+pos)) != 0 {
		return "X"
	}
	if current&(1<<(shift[O]+pos)) != 0 {
		return "O"
	}
	return " "
}

func display() {
	msg(" %s | %s | %s ", mark(0), mark(1), mark(2))
	msg("------------")
	msg(" %s | %s | %s ", mark(3), mark(4), mark(5))
	msg("------------")
	msg(" %s | %s | %s ", mark(6), mark(7), mark(8))
	msg("")
}

