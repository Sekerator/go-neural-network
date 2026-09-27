package internal

import (
	"neural_network/internal/services"
)

type Handler struct {
	data              services.NnInitData
	backpropagationNn *services.BackpropagationNn

	brains map[int]*Brain
}

func NewHandler(data services.NnInitData) *Handler {
	return &Handler{
		data: data,
	}
}
