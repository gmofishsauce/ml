/* Copyright Jeff Berkowitz 2025. MIT license. */

package main

import (
	"fmt"
	"math/rand"
	"os"
)

// Number of iterations to run
const NumGames = 10*1000*1000

// Tic-tac-toe game with TD(0) reinforcement learning
//
// Board state. Board is number 1..9 from upper left
// to lower right. Positions are stored in a uint32
// sort "one hot": bits 0..8 are X moves, 16..24 are
// O moves, and neither bit set means an empty square.
type bitboard uint32

const X = 0
const O = 1
const ALL = 0b111111111

// Old FORTRAN trick
func other(player int) int {
	return 1 - player
}

var shift[2] int = [2]int{0, 16}
var mask[2] bitboard  = [2]bitboard{ALL<<shift[X], ALL<<shift[O]}
var name[2] string = [2]string{"X", "O"}

var verbose bool = false
var quiet bool = false
var progname string

func main() {
	progname = os.Args[0]
	for i := 1; i < len(os.Args); i++ {
		if os.Args[i] == "-v" {
			verbose = true
		} else if os.Args[i] == "-q" {
			quiet = true
		}
	}

	msg("Firing up...")

	var results []int = []int{0, 0, 0, 0}

	for i := 0; i < NumGames; i++ {
		var current bitboard = 0

		for player := X; !isFinal(current); player = other(player) {
			move, estimate := chooseMove(current, player)
			updateWeights(move, estimate)
			current = current|move
		}
		results[status(current)]++
	}

	msg("X won %d times, O won %d times, and there were %d draws\n",
		results[1], results[2], results[3])
	msg("done")
}

var unshiftedWinningPositions = []bitboard {
	0b000000111, 0b000111000, 0b111000000,
	0b001001001, 0b010010010, 0b100100100,
	0b100010001, 0b001010100,
}

// Return true if no more plays are available or if there is a winner
func isFinal(position bitboard) bool {
	display(position)
	return status(position) > 0
}

// Return the state: 0 for in progress, 1 for X win, 2 for O win, 3 for draw.
func status(position bitboard) int {
	for _, winner := range(unshiftedWinningPositions) {
		winmask := winner<<shift[X]
		if position&winmask == winmask {
			//msg("X win")
			return 1
		}
		winmask = winner<<shift[O]
		if position&winmask == winmask {
			//msg("O win")
			return 2
		}
	}

	filledSquares := (position&mask[O])>>shift[O] | (position&mask[X])>>shift[X]
	if filledSquares == ALL {
		//msg("draw")
		return 3
	}

	return 0
}

// Choose the next move. This function only chooses legal moves.
func chooseMove(current bitboard, player int) (bitboard, float64) {
	var result bitboard
	max := -100.0

	for i := 0; i < 9; i++ {
		candidate := bitboard((1<<i)<<shift[player])
		blocker := bitboard((1<<i)<<shift[other(player)])
		value := 0.0
		if candidate&current == 0 && blocker&current == 0 { // legal
			value = eval(current|candidate)
			if value >= max {
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
	return rand.Float64()
}

// IO functions to end

func msg(format string, a ...any) {
	if quiet {
		return
	}
	if len(format) == 0 {
		fmt.Fprintln(os.Stderr, "")
		return
	}
	format = progname + ": " + format
	if format[len(format)-1] != '\n' {
		format += "\n"
	}
    fmt.Fprintf(os.Stderr, format, a...)
}

func mark(c bitboard, pos int) string {
	if c&(1<<(shift[X]+pos)) != 0 {
		return "X"
	}
	if c&(1<<(shift[O]+pos)) != 0 {
		return "O"
	}
	return " "
}

func display(c bitboard) {
	if !verbose || c == 0 { // don't print empty boards
		return
	}
	msg(" %s | %s | %s ", mark(c, 0), mark(c, 1), mark(c, 2))
	msg("------------")
	msg(" %s | %s | %s ", mark(c, 3), mark(c, 4), mark(c, 5))
	msg("------------")
	msg(" %s | %s | %s ", mark(c, 6), mark(c, 7), mark(c, 8))
	msg("")
}

