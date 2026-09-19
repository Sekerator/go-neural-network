package services

import "neural_network/internal/models"

type BackpropagationNn struct {
	data          NnInitData
	inputNeurons  map[int]models.Neuron
	outputNeurons map[int]models.Neuron
	hiddenLayers  map[int]map[int]models.Neuron
}

func NewBackpropagationNn(data NnInitData) NnService {
	return &BackpropagationNn{
		data: data,
	}
}

func (n *BackpropagationNn)

func (n *BackpropagationNn) Init(data NnInitData) {

}
