package services

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"neural_network/internal/models"
)

type Nn struct {
	data          NnInitData
	InputNeurons  []int
	OutputNeurons []int
	HiddenLayers  map[int][]int

	Neurons  map[int]*models.Neuron
	Synapses map[int]*models.Synapse
}

func NewNn(data NnInitData) *Nn {
	return &Nn{
		data: data,
	}
}

func (n *Nn) Init() error {
	neuronId := 0
	synapseId := 0

	if n.data.InputNeuronCount == 0 || n.data.OutputNeuronCount == 0 {
		return errors.New("input neuron count cannot be zero")
	}

	n.HiddenLayers = make(map[int][]int)
	n.Neurons = make(map[int]*models.Neuron)
	n.Synapses = make(map[int]*models.Synapse)

	for range n.data.InputNeuronCount {
		n.Neurons[neuronId] = &models.Neuron{
			ID:     neuronId,
			Result: 0,
			Bias:   rand.Float64() - 0.5,

			RightSynapseIDs: make([]int, 0),
			LeftSynapseIDs:  make([]int, 0),
		}

		n.InputNeurons = append(n.InputNeurons, neuronId)
		neuronId++
	}

	for i := range n.data.HiddenLayerCount {
		for range n.data.HiddenNeuronCount[i] {
			n.Neurons[neuronId] = &models.Neuron{
				ID:     neuronId,
				Result: 0,
				Bias:   rand.Float64() - 0.5,

				RightSynapseIDs: make([]int, 0),
				LeftSynapseIDs:  make([]int, 0),
			}

			n.HiddenLayers[i] = append(n.HiddenLayers[i], neuronId)
			neuronId++
		}
	}

	for range n.data.OutputNeuronCount {
		n.Neurons[neuronId] = &models.Neuron{
			ID:     neuronId,
			Result: 0,
			Bias:   rand.Float64() - 0.5,

			RightSynapseIDs: make([]int, 0),
			LeftSynapseIDs:  make([]int, 0),
		}

		n.OutputNeurons = append(n.OutputNeurons, neuronId)
		neuronId++
	}

	if n.data.HiddenLayerCount == 0 {
		for _, inputId := range n.InputNeurons {
			for _, outputId := range n.OutputNeurons {
				n.Synapses[synapseId] = &models.Synapse{
					ID:            synapseId,
					Weight:        rand.Float64() - 0.5,
					LeftNeuronID:  inputId,
					RightNeuronID: outputId,
				}

				n.Neurons[inputId].RightSynapseIDs = append(n.Neurons[inputId].RightSynapseIDs, synapseId)
				n.Neurons[outputId].LeftSynapseIDs = append(n.Neurons[outputId].LeftSynapseIDs, synapseId)
				synapseId++
			}
		}
	} else {
		for _, inputId := range n.InputNeurons {
			for _, hiddenId := range n.HiddenLayers[0] {
				n.Synapses[synapseId] = &models.Synapse{
					ID:            synapseId,
					Weight:        rand.Float64() - 0.5,
					LeftNeuronID:  inputId,
					RightNeuronID: hiddenId,
				}

				n.Neurons[inputId].RightSynapseIDs = append(n.Neurons[inputId].RightSynapseIDs, synapseId)
				n.Neurons[hiddenId].LeftSynapseIDs = append(n.Neurons[hiddenId].LeftSynapseIDs, synapseId)
				synapseId++
			}
		}

		for i := 0; i < n.data.HiddenLayerCount-1; i++ {
			for _, hiddenId1 := range n.HiddenLayers[i] {
				for _, hiddenId2 := range n.HiddenLayers[i+1] {
					n.Synapses[synapseId] = &models.Synapse{
						ID:            synapseId,
						Weight:        rand.Float64() - 0.5,
						LeftNeuronID:  hiddenId1,
						RightNeuronID: hiddenId2,
					}

					n.Neurons[hiddenId1].RightSynapseIDs = append(n.Neurons[hiddenId1].RightSynapseIDs, synapseId)
					n.Neurons[hiddenId2].LeftSynapseIDs = append(n.Neurons[hiddenId2].LeftSynapseIDs, synapseId)
					synapseId++
				}
			}
		}

		for _, hiddenId := range n.HiddenLayers[n.data.HiddenLayerCount-1] {
			for _, outputId := range n.OutputNeurons {
				n.Synapses[synapseId] = &models.Synapse{
					ID:            synapseId,
					Weight:        rand.Float64() - 0.5,
					LeftNeuronID:  hiddenId,
					RightNeuronID: outputId,
				}

				n.Neurons[hiddenId].RightSynapseIDs = append(n.Neurons[hiddenId].RightSynapseIDs, synapseId)
				n.Neurons[outputId].LeftSynapseIDs = append(n.Neurons[outputId].LeftSynapseIDs, synapseId)
				synapseId++
			}
		}
	}

	return nil
}

func (n *Nn) GetResults() []float64 {
	n.CalculateResults()

	var results []float64

	for _, neuronId := range n.OutputNeurons {
		results = append(results, n.Neurons[neuronId].Result)
	}

	return results
}

func (n *Nn) CalculateResults() {
	for i := 0; i < n.data.HiddenLayerCount; i++ {
		for _, hiddenNeuronId := range n.HiddenLayers[i] {
			result := 0.0
			for _, synapseId := range n.Neurons[hiddenNeuronId].LeftSynapseIDs {
				result += n.Neurons[n.Synapses[synapseId].LeftNeuronID].Result * n.Synapses[synapseId].Weight
			}
			n.Neurons[hiddenNeuronId].SetResult(result)
		}
	}

	for _, outputNeuronId := range n.OutputNeurons {
		result := 0.0
		for _, synapseId := range n.Neurons[outputNeuronId].LeftSynapseIDs {
			result += n.Neurons[n.Synapses[synapseId].LeftNeuronID].Result * n.Synapses[synapseId].Weight
		}
		n.Neurons[outputNeuronId].SetResult(result)
	}
}

func (n *Nn) Clone() *Nn {
	clone := NewNn(n.data)
	err := clone.Init()
	if err != nil {
		return nil
	}

	copy(clone.InputNeurons, n.InputNeurons)
	copy(clone.OutputNeurons, n.OutputNeurons)

	clone.HiddenLayers = make(map[int][]int, n.data.HiddenLayerCount)
	for id, value := range n.HiddenLayers {
		clone.HiddenLayers[id] = make([]int, len(value))
		copy(clone.HiddenLayers[id], value)
	}

	clone.Neurons = make(map[int]*models.Neuron, len(n.Neurons))
	for id, neuron := range n.Neurons {
		clone.Neurons[id] = &models.Neuron{
			ID:              neuron.ID,
			Result:          neuron.Result,
			Bias:            neuron.Bias,
			LeftSynapseIDs:  make([]int, len(neuron.LeftSynapseIDs)),
			RightSynapseIDs: make([]int, len(neuron.RightSynapseIDs)),
		}
		copy(clone.Neurons[id].LeftSynapseIDs, neuron.LeftSynapseIDs)
		copy(clone.Neurons[id].RightSynapseIDs, neuron.RightSynapseIDs)
	}

	clone.Synapses = make(map[int]*models.Synapse, len(n.Synapses))
	for id, synapse := range n.Synapses {
		clone.Synapses[id] = &models.Synapse{
			ID:            synapse.ID,
			Weight:        synapse.Weight,
			LeftNeuronID:  synapse.LeftNeuronID,
			RightNeuronID: synapse.RightNeuronID,
		}
	}

	return clone
}

func (n *Nn) SetInput(inputData []float64) error {
	if len(inputData) != len(n.InputNeurons) {
		return fmt.Errorf(
			"input data length %d does not match number of input neurons %d",
			len(inputData),
			len(n.InputNeurons),
		)
	}

	maxInput := 0.0
	for _, value := range inputData {
		maxInput = math.Max(maxInput, math.Abs(value))
	}

	if maxInput != 0 {
		for i := range inputData {
			inputData[i] /= maxInput
		}
	}

	i := 0
	for _, neuronId := range n.InputNeurons {
		n.Neurons[neuronId].Result = inputData[i]
		i++
	}

	return nil
}
