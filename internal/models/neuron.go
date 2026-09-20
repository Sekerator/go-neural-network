package models

import "math"

type Neuron struct {
	ID     int
	Result float64
	Bias   float64

	LeftSynapseIDs  []int
	RightSynapseIDs []int
}

func (n *Neuron) SetResult(result float64) {
	n.Result = math.Tanh(result + n.Bias)
}
