/* Copyright Jeff Berkowitz 2025. MIT license. */

package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
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
const NumSquares = 9

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

// Hyperparameters
const InputSize = 18
const HiddenSize = 18
var epsilon float64 = 0.5 // not very large
var epsilon_decay float64 = 0.75 // very rapid
var gamma float64 = 0.99

var nn *NN = makeNN(18, HiddenSize, 1)

var verbose bool = false
var quiet bool = false
var progname string

func main() {
	progname = os.Args[0]
	msg("Firing up...")

	for i := 1; i < len(os.Args); i++ {
		if os.Args[i] == "-v" {
			verbose = true
		} else if os.Args[i] == "-q" {
			quiet = true
		} else {
			fatal(fmt.Sprintf("usage: %s [-v|-q]", progname))
			// does not return
		}
	}

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
			// We just switched (or initialized) the players, so the quality of the current
			// position is based on the other player's move choice.
			position := current
			qOfS := QofS(position, other(player))
			action, _ := choose(current, player, epsilon)               // "choose A from S"
			current = current|action                                    // "take action A, observe R, S'"
			r := reward(current, player)

			// LR is learning rate, gamma is discount rate, Q(S, A) is saved above
			// max_a(Q(S', a)) is found by choose() with epsilon of 0.0
			// _, maxQ := choose(current, player, 0.0)
			// Q(S, A) := Q(S, A) + LR * [ r + gamma * (max_a(Q(S', a))) - Q(S, A) ]
			// LR is in the NN and the Q(S, A) part is built into the NN (I think...)

			_, maxQ := choose(current, other(player), 0.0)
			targetQ := make([]float64, 1)
			targetQ[0] = r + gamma * (1 - maxQ) - qOfS
			nn.Learn(bitboardToFloatVec(position), targetQ)
		}

		results[status(current)]++
		if epsilon > 0.001 {
			epsilon *= epsilon_decay
		}
	}
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
		return 0.5
	}
	if s == StatusXWin {
		return 1
	}
	return 0 // O win
}

func getLegalMoves(current bitboard) []int {
	result := []int{}

	packed := (current&mask[X])>>shift[X] | (current&mask[O])>>shift[O]
	for i := 0; i < 9; i++ {
		if packed&(1<<i) == 0 {
			result = append(result, i)
		}
	}
	return result
}

func getRandomMove(current bitboard, player int) bitboard {
	choices := getLegalMoves(current)
	return (1<<choices[rand.Intn(len(choices))]) << shift[player]
}

// TODO XXX needs to return 0.0 and an empty bitboard when isFinal() is true.

// Choose the next move. This function only chooses legal moves. For the X player,
// we choose the maximum from the N QofS values. For the O player, we choose the
// minimum. Inside the function estimator (neural net), the same thing happens for
// each evaluation.
func choose(current bitboard, player int, epsilon float64) (bitboard, float64) {
	var minmax float64

	if isFinal(current) {
		return 0, 0.5
	}
	if rand.Float64() < epsilon {
		move := getRandomMove(current, player)
		return move, QofS(move, player)
	}

	if player == X {
		minmax = -1e37
	} else {
		minmax = 1e37
	}

	var move bitboard
	for i := 0; i < 9; i++ {
		candidate := bitboard((1<<i)<<shift[player])
		blocker := bitboard((1<<i)<<shift[other(player)])
		value := 0.0
		if candidate&current == 0 && blocker&current == 0 { // legal
			value = QofS(current|candidate, player)
			if player == X && value > minmax {
				move = candidate
				minmax = value
			} else if player == O && value < minmax {
				move = candidate
				minmax = value
			}
		}
	}
	if move == 0 {
		panic("no move selection")
	}
	return move, minmax
}

func bitboardToFloatVec(position bitboard) []float64 {
	// prepare the input, an N-hot vector with 1.0 where there is either
	// an X or an O and 0.0 otherwise.
	packed := (position&mask[X]) | (position&mask[O])>>7
	var result []float64 = make([]float64, 2*NumSquares)
	for i := 0; i < 2*NumSquares; i++ {
		if packed&(1<<i) != 0 {
			result[i] = 1.0
		}
	}
	return result
}

// Return the estimated value of the position bitboard for the player X or O.
func QofS(position bitboard, player int) float64 {
	input := bitboardToFloatVec(position)
	result := nn.Predict(input)
	if len(result) != 1 {
		panic(fmt.Sprintf("QofS(): expected result vector of 1 from NN, got len %d\n", len(result)))
	}
	return result[0]
}

// IO functions to end

func fatal(format string, a ...any) {
	quiet = false
	if len(format) == 0 {
		panic("fatal() called with no message")
	}
	//msg(format, a) does not work. Why not?
	format = progname + ": " + format
	if format[len(format)-1] != '\n' {
		format += "\n"
	}
    fmt.Fprintf(os.Stderr, format, a...)
	os.Exit(1)
}

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
		elementFormat = "%.3f"
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

