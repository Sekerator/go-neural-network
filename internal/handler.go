package internal

import (
	"neural_network/internal/services"
)

type Handler struct {
	data              services.NnInitData
	backpropagationNn *services.BackpropagationNn
}

func NewHandler(data services.NnInitData) *Handler {
	return &Handler{data: data}
}

func (h *Handler) CreateBackpropagationNn() error {
	h.backpropagationNn = services.NewBackpropagationNn(h.data)
	err := h.backpropagationNn.Init()
	if err != nil {
		return err
	}

	return nil
}

func (h *Handler) Train(iterationCount int, data [][][]float64) error {
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

func (h *Handler) GetResult(input []float64) []float64 {
	err := h.backpropagationNn.SetInput(input)
	if err != nil {
		return nil
	}

	return h.backpropagationNn.GetResults()
}
