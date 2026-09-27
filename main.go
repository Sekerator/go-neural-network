package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"neural_network/internal"
	"neural_network/internal/services"
	"os"
	"sort"
	"strconv"
	"sync"
)

func main() {
	evolutionTest()
}

func evolutionTest() {
	data := services.NnInitData{
		InputNeuronCount:  2,
		HiddenLayerCount:  3,
		HiddenNeuronCount: []int{6, 4, 2},
		OutputNeuronCount: 1,

		MutationBiasChance:   10,
		MutationWeightChance: 10,

		MutationBiasRate:   0.05,
		MutationWeightRate: 0.05,
	}

	handler := internal.NewHandler(data)
	brains, err := handler.CreateEvolutionNn(10000)
	if err != nil {
		panic(err)
	}

	var trainData [][][]float64

	file, err := os.Open("train-data.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ';'

	records, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	for _, row := range records {
		if len(row) < 3 {
			panic(fmt.Sprintf("invalid row: %v", row))
		}

		var values [3]float64
		for i := range values {
			values[i], err = strconv.ParseFloat(row[i], 64)
			if err != nil {
				panic(err)
			}
		}
		a, b, result := values[0], values[1], values[2]

		trainData = append(trainData, [][]float64{{a, b}, {result}})
	}

	iterCount := 10000
	fmt.Print("Введите количество итераций обучения: ")
	fmt.Fscan(os.Stdin, &iterCount)

	for i := range iterCount {
		fmt.Print("Итерация: ")
		fmt.Println(i)
		scoreBoard := scoreBrains(brains, trainData)

		err = handler.TrainEvolutionNn(scoreBoard)
		if err != nil {
			panic(err)
		}
	}

	scoreBoard := scoreBrains(brains, trainData)

	type Item struct {
		Key   int
		Value float64
	}

	items := make([]Item, 0, len(scoreBoard))

	for k, v := range scoreBoard {
		items = append(items, Item{k, v})
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].Value > items[j].Value
	})

	fmt.Println()
	for {
		var num1 float64
		var num2 float64

		fmt.Print("Введите 1 цифру для сравнения: ")
		fmt.Fscan(os.Stdin, &num1)

		fmt.Print("Введите 2 цифру для сравнения: ")
		fmt.Fscan(os.Stdin, &num2)

		if num1 == 1515 && num2 == 1515 {
			break
		}

		results, err := brains[items[0].Key].GetResult([]float64{num1, num2})
		if err != nil {
			panic(err)
		}
		fmt.Print("Результат: ")
		if math.Round(results[0]) == 1 {
			fmt.Println("Цифра 2 больше")
			fmt.Println(results[0])
			fmt.Println(math.Round(results[0]))
		} else if math.Round(results[0]) == 0 {
			fmt.Println("Равны")
			fmt.Println(results[0])
			fmt.Println(math.Round(results[0]))
		} else if math.Round(results[0]) == -1 {
			fmt.Println("Цифра 1 больше")
			fmt.Println(results[0])
			fmt.Println(math.Round(results[0]))
		} else {
			fmt.Println(results[0])
			fmt.Println(math.Round(results[0]))
		}
	}
}

// scoreBrains returns the number of correct answers for every brain.
func scoreBrains(brains map[int]*internal.Brain, trainData [][][]float64) map[int]float64 {
	var wg sync.WaitGroup
	var sc sync.Mutex

	scoreBoard := make(map[int]float64, len(brains))
	for id := range brains {
		scoreBoard[id] = 0
	}

	for id, brain := range brains {
		wg.Add(1)
		go func() {
			defer wg.Done()
			correct := 0.0
			for _, tdata := range trainData {
				results, err := brain.GetResult(tdata[0])
				if err != nil {
					panic(err)
				}
				if math.Round(results[0]) == tdata[1][0] {
					correct++
				}
			}

			sc.Lock()
			scoreBoard[id] = correct
			sc.Unlock()
		}()
	}
	wg.Wait()

	return scoreBoard
}

func backpropagationTest() {
	data := services.NnInitData{
		InputNeuronCount:  2,
		HiddenLayerCount:  3,
		HiddenNeuronCount: []int{6, 4, 2},
		OutputNeuronCount: 1,

		MutationBiasChance:   10,
		MutationWeightChance: 10,

		MutationBiasRate:   0.05,
		MutationWeightRate: 0.05,
	}

	handler := internal.NewHandler(data)
	err := handler.CreateBackpropagationNn()
	if err != nil {
		panic(err)
	}

	var trainData [][][]float64

	file, err := os.Open("train-data.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ';'

	records, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	for _, row := range records {
		if len(row) < 3 {
			panic(fmt.Sprintf("invalid row: %v", row))
		}

		var values [3]float64
		for i := range values {
			values[i], err = strconv.ParseFloat(row[i], 64)
			if err != nil {
				panic(err)
			}
		}
		a, b, result := values[0], values[1], values[2]

		trainData = append(trainData, [][]float64{{a, b}, {result}})
	}

	iterCount := 10000
	fmt.Print("Введите количество итераций обучения: ")
	fmt.Fscan(os.Stdin, &iterCount)
	err = handler.TrainBackpropagationNn(iterCount, trainData)
	if err != nil {
		panic(err)
	}

	fmt.Println()
	for {
		var num1 float64
		var num2 float64

		fmt.Print("Введите 1 цифру для сравнения: ")
		fmt.Fscan(os.Stdin, &num1)

		fmt.Print("Введите 2 цифру для сравнения: ")
		fmt.Fscan(os.Stdin, &num2)

		if num1 == 1515 && num2 == 1515 {
			break
		}

		results, err := handler.GetResultBackpropagationNn([]float64{num1, num2})
		if err != nil {
			panic(err)
		}
		fmt.Print("Результат: ")
		if math.Round(results[0]) == 1 {
			fmt.Println("Цифра 2 больше")
			fmt.Println(results[0])
			fmt.Println(math.Round(results[0]))
		} else if math.Round(results[0]) == 0 {
			fmt.Println("Равны")
			fmt.Println(results[0])
			fmt.Println(math.Round(results[0]))
		} else if math.Round(results[0]) == -1 {
			fmt.Println("Цифра 1 больше")
			fmt.Println(results[0])
			fmt.Println(math.Round(results[0]))
		} else {
			fmt.Println(results[0])
			fmt.Println(math.Round(results[0]))
		}
	}
}
