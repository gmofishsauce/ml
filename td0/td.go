/* Copyright Jeff Berkowitz 2025. MIT license. */

package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"os"
)

// Number of iterations to run
const NumGames = 10

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

const (
	StatusRunning int = iota
	StatusXWin
	StatusOWin
	StatusDraw
)

var shift[2] int = [2]int{0, 16}
var mask[2] bitboard  = [2]bitboard{ALL<<shift[X], ALL<<shift[O]}
var name[2] string = [2]string{"X", "O"}

var nn *NN = makeNN(18, 6, 9)

var verbose bool = false
var quiet bool = false
var progname string

var ExploreParam = 0.3 // TODO epsilon and its shrink rate

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
	Q_learn(results)
	msg("X won %d times, O won %d times, and there were %d draws\n",
		results[1], results[2], results[3])

	msg("done")
}

func Q_learn(results []int) {
	for i := 0; i < NumGames; i++ {
		var current bitboard = 0                                        // "initialize S"

		for player := X; !isFinal(current); player = other(player) {
			position := current
			move, QofSgivenA := choose(current, player)                 // "choose A from S"
			current = current|move                                      // "take action A, observe R, S'"
			r := reward(current, player)                                // current is now Q(S',A)
			// LR is learning rate, gamma is discount rate, Q(S, A) is saved above
			// max_a(Q(S', a)) is the best result for the ***OTHER*** player
			// Q(S, A) := Q(S, A) + LR * [ r + gamma * (max_a(Q(S', a))) - Q(S, A) ]

			update(position, r, QofSgivenA)
		}
		results[status(current)]++
	}
}

func update(position bitboard, r float64, QofSgivenA float64) {
	msg("%x %f %f", position, r, QofSgivenA)
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

func status(position bitboard) int {
	for _, winner := range(unshiftedWinningPositions) {
		winmask := winner<<shift[X]
		if position&winmask == winmask {
			//msg("X win")
			return StatusXWin
		}
		winmask = winner<<shift[O]
		if position&winmask == winmask {
			//msg("O win")
			return StatusOWin
		}
	}

	filledSquares := (position&mask[O])>>shift[O] | (position&mask[X])>>shift[X]
	if filledSquares == ALL {
		//msg("draw")
		return StatusDraw
	}

	return StatusRunning
}

// TODO I don't understand what to use for rewards. It seems like (1, 0, -1) make
// sense for {X win, draw, O win}, but then what to do for "in progress"? If it's 0,
// then "in progress" gets rewarded the same as achieving a draw, which doesn't seem
// right because a draw is better than "in progress which carries a risk of loss.

func reward(position bitboard, player int) float64 {
	s := status(position)
	if s == StatusRunning || s == StatusDraw {
		return 0
	}
	if s == StatusXWin {
		return 1
	}
	return -1 // O win
}

// Choose the next move. This function only chooses legal moves.
// TODO epsilon
func choose(current bitboard, player int) (bitboard, float64) {
	var result bitboard
	max := -100.0

	for i := 0; i < 9; i++ {
		candidate := bitboard((1<<i)<<shift[player])
		blocker := bitboard((1<<i)<<shift[other(player)])
		value := 0.0
		if candidate&current == 0 && blocker&current == 0 { // legal
			value = eval(current|candidate, player)
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

func eval(position bitboard, player int) float64 {
	// prepare the input, an N-hot vector with 1.0 where there is either
	// an X or an O and 0.0 otherwise
	packed := (position&mask[X]) | (position&mask[O])>>7
	var input []float64 = make([]float64, nn.InputSize)
	for i := 0; i < nn.InputSize; i++ {
		if packed&(1<<i) != 0 {
			input[i] = 1.0
		}
	}

	return nn.Predict(input, (player == X))
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

// print a matrix m with the given label and per-element format
func msgM(label string, elementFormat string, m mat.Matrix) {
	if len(elementFormat) == 0 {
		elementFormat = "%f"
	}
	if len(label) == 0 || label[len(label)-1] != '\n' {
		label += "\n"
	}
	fa := mat.Formatted(m)
	msg(label+elementFormat, fa)
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

