package services

type NnInitData struct {
	InputNeuronCount  int
	HiddenLayerCount  int
	HiddenNeuronCount []int
	OutputNeuronCount int

	MutationBiasChance   int // for evolution
	MutationWeightChance int // for evolution

	MutationBiasRate   float64
	MutationWeightRate float64
}
