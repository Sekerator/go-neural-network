package services

type NnInitData struct {
	inputNeuronCount  int
	hiddenLayerCount  int
	hiddenNeuronCount []int
	outputNeuronCount int

	mutationBiasChance   int
	mutationWeightChance int
	mutationBiasRate     float64
	mutationWeightRate   float64
}

type NnService interface {
	GetResults() []float64
	Clone() NnService
	Mutate() error
	SetInput([]float64) error
	Init(data NnInitData) error
	Train() error
}
