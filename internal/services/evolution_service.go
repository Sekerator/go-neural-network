package services

type EvolutionNn struct {
	*Nn
}

func NewEvolutionNn(data NnInitData) *EvolutionNn {
	return &EvolutionNn{
		Nn: NewNn(data),
	}
}

func (n *EvolutionNn) Train() error {

	return nil
}
