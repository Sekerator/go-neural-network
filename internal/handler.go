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
