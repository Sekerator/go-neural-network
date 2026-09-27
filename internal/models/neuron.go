package models

import "math"

const (
	INPUT_NEURON  = 1
	HIDDEN_NEURON = 2
	OUTPUT_NEURON = 3
)

type Neuron struct {
	ID     int
	Result float64
	Bias   float64
	Type   int

	LeftSynapseIDs  []int
	RightSynapseIDs []int
}

func (n *Neuron) SetResult(result float64) {
	n.Result = math.Tanh(result + n.Bias)
}
