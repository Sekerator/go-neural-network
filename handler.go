package neuralnetwork

import (
	"github.com/Sekerator/go-neural-network/internal/services"
)

type NnInitData = services.NnInitData

type Handler struct {
	data              services.NnInitData
	backpropagationNn *services.BackpropagationNn

	brains map[int]*Brain
}

func NewHandler(data NnInitData) *Handler {
	return &Handler{
		data: data,
	}
}
