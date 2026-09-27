package services

import (
	"errors"
	"github.com/Sekerator/go-neural-network/internal/models"
	"math/rand"
)

type EvolutionNn struct {
	*Nn
}

func NewEvolutionNn(data NnInitData) *EvolutionNn {
	return &EvolutionNn{
		Nn: NewNn(data),
	}
}

func (n *EvolutionNn) Mutate() error {
	for _, neuron := range n.Neurons {
		if neuron.Type != models.INPUT_NEURON {
			if rand.Intn(100) < n.data.MutationBiasChance {
				neuron.Bias += (rand.Float64() - 0.5) * n.data.MutationBiasRate
			}
		}
	}

	for _, synapse := range n.Synapses {
		if rand.Intn(100) < n.data.MutationWeightChance {
			synapse.Weight += (rand.Float64() - 0.5) * n.data.MutationWeightRate
		}
	}

	return nil
}

func Cross(nn1, nn2 EvolutionNn, chance int) (EvolutionNn, error) {
	result := nn1.Clone()
	if result == nil {
		return EvolutionNn{}, errors.New("failed to clone nn")
	}

	for id, neuron := range result.Neurons {
		if neuron.Type != models.INPUT_NEURON {
			if rand.Intn(100) < chance {
				neuron.Bias = nn2.Neurons[id].Bias
			}
		}
	}

	for id, synapse := range result.Synapses {
		if rand.Intn(100) < chance {
			synapse.Weight = nn2.Synapses[id].Weight
		}
	}

	return EvolutionNn{Nn: result}, nil
}
