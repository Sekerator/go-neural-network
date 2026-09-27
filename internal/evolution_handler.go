package internal

import (
	"errors"
	"math/rand"
	"neural_network/internal/services"
	"sort"
	"sync"
)

type Brain struct {
	ID int
	nn *services.EvolutionNn
}

func (h *Handler) CreateEvolutionNn(count int) (map[int]*Brain, error) {
	var errorsList []error
	var wg sync.WaitGroup
	var sc sync.Mutex

	h.brains = make(map[int]*Brain, count)

	for i := range count {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sc.Lock()
			h.brains[i] = &Brain{
				ID: i,
				nn: services.NewEvolutionNn(h.data),
			}
			err := h.brains[i].nn.Init()
			if err != nil {
				errorsList = append(errorsList, err)
			}
			sc.Unlock()
		}()
	}

	wg.Wait()

	if len(errorsList) > 0 {
		return nil, errorsList[0]
	}

	return h.brains, nil
}

func (h *Handler) SetInputAllEvolutionNn(input []float64) error {
	if len(h.brains) == 0 {
		return errors.New("no brains")
	}

	var errorsList []error
	var wg sync.WaitGroup

	for _, brain := range h.brains {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := brain.nn.SetInput(input)
			if err != nil {
				errorsList = append(errorsList, err)
			}
		}()
	}

	wg.Wait()

	if len(errorsList) > 0 {
		return errorsList[0]
	}

	return nil
}

func (b *Brain) GetResult(input []float64) ([]float64, error) {
	err := b.nn.SetInput(input)
	if err != nil {
		return nil, err
	}

	return b.nn.GetResults(), nil
}

func (b *Brain) Mutate() error {
	return b.nn.Mutate()
}

func (h *Handler) CrossEvolutionNn(intoId, fromId, fromIdDominationChance int) error {
	if len(h.brains) == 0 {
		return errors.New("no brains")
	}

	nn, err := services.Cross(*h.brains[intoId].nn, *h.brains[fromId].nn, fromIdDominationChance)
	if err != nil {
		return err
	}

	h.brains[intoId].nn = &nn

	return nil
}

func (h *Handler) TrainEvolutionNn(scoreBoard map[int]float64) error {
	var errorsList []error
	var wg sync.WaitGroup
	var sc sync.Mutex

	type Item struct {
		Key   int
		Value float64
	}

	items := make([]Item, 0, len(scoreBoard))

	for k, v := range scoreBoard {
		items = append(items, Item{k, v})
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].Value < items[j].Value
	})

	counter := -1
	for _, item := range items {
		counter++
		wg.Add(1)
		go func(counterGo int) {
			defer wg.Done()
			if counterGo < len(items)/10 {
			} else if counterGo < len(items)/2 {
				err := h.brains[item.Key].nn.Mutate()
				if err != nil {
					sc.Lock()
					errorsList = append(errorsList, err)
					sc.Unlock()
				}
			} else if counterGo < len(items) {
				err := h.CrossEvolutionNn(item.Key, rand.Intn(len(h.brains)), 50)
				if err != nil {
					sc.Lock()
					errorsList = append(errorsList, err)
					sc.Unlock()
				}
			}
		}(counter)
	}

	wg.Wait()

	if len(errorsList) > 0 {
		return errorsList[0]
	}

	return nil
}
