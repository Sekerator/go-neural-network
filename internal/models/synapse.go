package models

type Synapse struct {
	ID     int
	Weight float64

	LeftNeuronID  int
	RightNeuronID int
}
