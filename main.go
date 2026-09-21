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

	trainData := [][][]float64{
		{{50, 5}, {-1}},
		{{12, 3}, {-1}},
		{{99, 1}, {-1}},
		{{25, 7}, {-1}},
		{{8, 2}, {-1}},
		{{60, 10}, {-1}},
		{{17, 4}, {-1}},
		{{45, 11}, {-1}},
		{{73, 22}, {-1}},
		{{31, 9}, {-1}},
		{{14, 6}, {-1}},
		{{88, 44}, {-1}},
		{{27, 13}, {-1}},
		{{100, 20}, {-1}},
		{{55, 5}, {-1}},
		{{42, 21}, {-1}},
		{{19, 8}, {-1}},
		{{64, 16}, {-1}},
		{{90, 30}, {-1}},
		{{33, 12}, {-1}},
		{{7, 1}, {-1}},
		{{81, 9}, {-1}},
		{{29, 14}, {-1}},
		{{66, 33}, {-1}},
		{{48, 24}, {-1}},
		{{15, 5}, {-1}},
		{{39, 13}, {-1}},
		{{72, 18}, {-1}},
		{{54, 27}, {-1}},
		{{22, 11}, {-1}},
		{{95, 19}, {-1}},
		{{36, 6}, {-1}},
		{{84, 28}, {-1}},
		{{63, 7}, {-1}},
		{{41, 20}, {-1}},
		{{58, 29}, {-1}},
		{{76, 38}, {-1}},
		{{24, 3}, {-1}},
		{{69, 23}, {-1}},
		{{52, 26}, {-1}},

		{{3, 30}, {1}},
		{{2, 9}, {1}},
		{{1, 99}, {1}},
		{{7, 25}, {1}},
		{{2, 8}, {1}},
		{{10, 60}, {1}},
		{{4, 17}, {1}},
		{{11, 45}, {1}},
		{{22, 73}, {1}},
		{{9, 31}, {1}},
		{{6, 14}, {1}},
		{{44, 88}, {1}},
		{{13, 27}, {1}},
		{{20, 100}, {1}},
		{{5, 55}, {1}},
		{{21, 42}, {1}},
		{{8, 19}, {1}},
		{{16, 64}, {1}},
		{{30, 90}, {1}},
		{{12, 33}, {1}},
		{{1, 7}, {1}},
		{{9, 81}, {1}},
		{{14, 29}, {1}},
		{{33, 66}, {1}},
		{{24, 48}, {1}},
		{{5, 15}, {1}},
		{{13, 39}, {1}},
		{{18, 72}, {1}},
		{{27, 54}, {1}},
		{{11, 22}, {1}},
		{{19, 95}, {1}},
		{{6, 36}, {1}},
		{{28, 84}, {1}},
		{{7, 63}, {1}},
		{{20, 41}, {1}},
		{{29, 58}, {1}},
		{{38, 76}, {1}},
		{{3, 24}, {1}},
		{{23, 69}, {1}},
		{{26, 52}, {1}},
		{{5001, 5000}, {-1}},
		{{5000, 5001}, {1}},
		{{9999, 9998}, {-1}},
		{{9998, 9999}, {1}},
		{{10001, 10000}, {-1}},
		{{10000, 10001}, {1}},

		{{10, 10}, {0}},
		{{1, 1}, {0}},
		{{5, 5}, {0}},
		{{20, 20}, {0}},
		{{100, 100}, {0}},
		{{7, 7}, {0}},
		{{42, 42}, {0}},
		{{13, 13}, {0}},
		{{99, 99}, {0}},
		{{0, 0}, {0}},
		{{25, 25}, {0}},
		{{64, 64}, {0}},
		{{3, 3}, {0}},
		{{88, 88}, {0}},
		{{11, 11}, {0}},
		{{50, 50}, {0}},
		{{17, 17}, {0}},
		{{72, 72}, {0}},
		{{6, 6}, {0}},
		{{30, 30}, {0}},
	}

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
