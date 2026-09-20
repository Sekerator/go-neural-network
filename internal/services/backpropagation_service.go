package services

import "errors"

type BackpropagationNn struct {
	*Nn
}

func NewBackpropagationNn(data NnInitData) *BackpropagationNn {
	return &BackpropagationNn{
		Nn: NewNn(data),
	}
}

func (n *BackpropagationNn) Train(expectedData []float64) error {
	if len(expectedData) != n.data.OutputNeuronCount {
		return errors.New("expected data length must be equal to output length")
	}

	n.CalculateResults()

	neuronErrors := make(map[int]float64)
	i := 0
	for _, neuronId := range n.OutputNeurons {
		neuronErrors[neuronId] = (n.Neurons[neuronId].Result - expectedData[i]) * (1 - (n.Neurons[neuronId].Result * n.Neurons[neuronId].Result))
		i++

		for _, synapseId := range n.Neurons[neuronId].LeftSynapseIDs {
			n.Synapses[synapseId].Weight -= n.data.MutationWeightRate * neuronErrors[neuronId] * n.Neurons[n.Synapses[synapseId].LeftNeuronID].Result
		}
		n.Neurons[neuronId].Bias -= n.data.MutationBiasRate * neuronErrors[neuronId]
	}

	for i = n.data.HiddenLayerCount - 1; i >= 0; i-- {
		for _, neuronId := range n.HiddenLayers[i] {
			outputErrors := 0.0
			for _, synapseId := range n.Neurons[neuronId].RightSynapseIDs {
				outputErrors += n.Synapses[synapseId].Weight * neuronErrors[n.Synapses[synapseId].RightNeuronID]
			}

			neuronErrors[neuronId] = (1 - (n.Neurons[neuronId].Result * n.Neurons[neuronId].Result)) * outputErrors

			for _, synapseId := range n.Neurons[neuronId].LeftSynapseIDs {
				n.Synapses[synapseId].Weight -= n.data.MutationWeightRate * neuronErrors[neuronId] * n.Neurons[n.Synapses[synapseId].LeftNeuronID].Result
			}
			n.Neurons[neuronId].Bias -= n.data.MutationBiasRate * neuronErrors[neuronId]
		}
	}

	return nil
}
