package models

type Neuron struct {
	ID     int
	Result float64
	Bias   float64

	LeftSynapseIDs  []int
	RightSynapseIDs []int
}
