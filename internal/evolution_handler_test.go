package internal

import (
	"neural_network/internal/services"
	"testing"
)

func testInitData() services.NnInitData {
	return services.NnInitData{
		InputNeuronCount:  2,
		HiddenLayerCount:  2,
		HiddenNeuronCount: []int{4, 2},
		OutputNeuronCount: 1,

		MutationBiasChance:   10,
		MutationWeightChance: 10,

		MutationBiasRate:   0.05,
		MutationWeightRate: 0.05,
	}
}

func scoresByID(brains map[int]*Brain) map[int]float64 {
	scoreBoard := make(map[int]float64, len(brains))
	for id := range brains {
		scoreBoard[id] = float64(id)
	}
	return scoreBoard
}

func TestTrainEvolutionNnPopulationSizes(t *testing.T) {
	for _, count := range []int{1, 2, 5, 10, 200} {
		h := NewHandler(testInitData())
		brains, err := h.CreateEvolutionNn(count)
		if err != nil {
			t.Fatalf("count %d: create: %v", count, err)
		}
		if len(brains) != count {
			t.Fatalf("count %d: got %d brains", count, len(brains))
		}

		for range 3 {
			if err := h.TrainEvolutionNn(scoresByID(brains)); err != nil {
				t.Fatalf("count %d: train: %v", count, err)
			}
		}
	}
}

func TestTrainEvolutionNnKeepsTopBrains(t *testing.T) {
	h := NewHandler(testInitData())
	brains, err := h.CreateEvolutionNn(20)
	if err != nil {
		t.Fatal(err)
	}

	// Ids 19 and 18 have the highest scores and must be kept as is.
	top := []int{19, 18}
	before := make(map[int]*services.EvolutionNn, len(top))
	for _, id := range top {
		before[id] = brains[id].nn
	}

	if err := h.TrainEvolutionNn(scoresByID(brains)); err != nil {
		t.Fatal(err)
	}

	for _, id := range top {
		if brains[id].nn != before[id] {
			t.Errorf("top brain %d was replaced", id)
		}
	}
}

func TestTrainEvolutionNnInvalidScoreBoard(t *testing.T) {
	h := NewHandler(testInitData())
	brains, err := h.CreateEvolutionNn(5)
	if err != nil {
		t.Fatal(err)
	}

	missing := scoresByID(brains)
	delete(missing, 0)

	unknown := scoresByID(brains)
	delete(unknown, 0)
	unknown[100] = 1

	for name, scoreBoard := range map[string]map[int]float64{
		"empty":   {},
		"missing": missing,
		"unknown": unknown,
	} {
		if err := h.TrainEvolutionNn(scoreBoard); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestTrainEvolutionNnWithoutBrains(t *testing.T) {
	h := NewHandler(testInitData())
	if err := h.TrainEvolutionNn(map[int]float64{0: 1}); err == nil {
		t.Error("expected error")
	}
}

func TestCreateEvolutionNnInvalid(t *testing.T) {
	bad := testInitData()
	bad.HiddenNeuronCount = []int{4, 0}

	h := NewHandler(bad)
	if _, err := h.CreateEvolutionNn(10); err == nil {
		t.Error("invalid config: expected error")
	}
	if h.brains != nil {
		t.Error("invalid config: brains must stay unset")
	}

	h = NewHandler(testInitData())
	for _, count := range []int{0, -1} {
		if _, err := h.CreateEvolutionNn(count); err == nil {
			t.Errorf("count %d: expected error", count)
		}
	}
}

func TestCrossEvolutionNnUnknownBrain(t *testing.T) {
	h := NewHandler(testInitData())
	if _, err := h.CreateEvolutionNn(3); err != nil {
		t.Fatal(err)
	}

	if err := h.CrossEvolutionNn(0, 42, 50); err == nil {
		t.Error("expected error for unknown donor")
	}
	if err := h.CrossEvolutionNn(42, 0, 50); err == nil {
		t.Error("expected error for unknown target")
	}
	if err := h.CrossEvolutionNn(0, 1, 50); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestGetResultDoesNotModifyInput(t *testing.T) {
	h := NewHandler(testInitData())
	brains, err := h.CreateEvolutionNn(1)
	if err != nil {
		t.Fatal(err)
	}

	input := []float64{3, 5}
	if _, err := brains[0].GetResult(input); err != nil {
		t.Fatal(err)
	}
	if input[0] != 3 || input[1] != 5 {
		t.Errorf("input was modified: %v", input)
	}
}
