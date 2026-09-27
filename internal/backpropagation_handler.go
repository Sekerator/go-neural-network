package internal

import (
	"errors"
	"neural_network/internal/services"
)

func (h *Handler) CreateBackpropagationNn() error {
	h.backpropagationNn = services.NewBackpropagationNn(h.data)
	err := h.backpropagationNn.Init()
	if err != nil {
		return err
	}

	return nil
}

func (h *Handler) TrainBackpropagationNn(iterationCount int, data [][][]float64) error {
	if len(data) == 0 {
		return errors.New("no data")
	}
	if h.backpropagationNn == nil {
		return errors.New("no backpropagation nn")
	}

	for range iterationCount {
		for _, v := range data {
			err := h.backpropagationNn.SetInput(v[0])
			if err != nil {
				return err
			}

			h.backpropagationNn.CalculateResults()
			err = h.backpropagationNn.Train(v[1])
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (h *Handler) GetResultBackpropagationNn(input []float64) ([]float64, error) {
	if h.backpropagationNn == nil {
		return nil, errors.New("no backpropagation nn")
	}

	err := h.backpropagationNn.SetInput(input)
	if err != nil {
		return nil, err
	}

	return h.backpropagationNn.GetResults(), nil
}
