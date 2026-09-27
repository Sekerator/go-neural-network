package services

import (
	"errors"
	"fmt"
)

type NnInitData struct {
	InputNeuronCount  int
	HiddenLayerCount  int
	HiddenNeuronCount []int
	OutputNeuronCount int

	MutationBiasChance   int
	MutationWeightChance int

	MutationBiasRate   float64
	MutationWeightRate float64
}

func (d NnInitData) Validate() error {
	if d.InputNeuronCount <= 0 || d.OutputNeuronCount <= 0 {
		return errors.New("input and output neuron counts must be positive")
	}

	if d.HiddenLayerCount != len(d.HiddenNeuronCount) {
		return errors.New("hidden layer count not equal to hidden neuron count")
	}

	for i, count := range d.HiddenNeuronCount {
		if count <= 0 {
			return fmt.Errorf("hidden layer %d neuron count must be positive, got %d", i, count)
		}
	}

	return nil
}
