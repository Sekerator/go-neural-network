package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"neural_network/internal"
	"neural_network/internal/services"
	"os"
	"strconv"
)

func main() {
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
		var a, b, result float64
		a, err = strconv.ParseFloat(row[0], 64)
		b, err = strconv.ParseFloat(row[1], 64)
		result, err = strconv.ParseFloat(row[2], 64)
		if err != nil {
			panic(err)
		}

		trainData = append(trainData, [][]float64{{a, b}, {result}})
	}

	iterCount := 10000
	fmt.Print("Введите количество итераций обучения: ")
	fmt.Fscan(os.Stdin, &iterCount)
	err = handler.Train(iterCount, trainData)
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

		results := handler.GetResult([]float64{num1, num2})
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
