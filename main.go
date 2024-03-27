package main

import "C"
import (
	"fmt"
	"github.com/sbromberger/gompi"
	"math/rand"
	"time"
)

func extractLU(A [][]float64) ([][]float64, [][]float64) {
	n := len(A)
	L := generateEmptyMatrix(int32(n))
	U := generateEmptyMatrix(int32(n))

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i < j {
				U[i][j] = A[i][j]
			} else if i == j {
				L[i][j] = 1
				U[i][j] = A[i][j]
			} else {
				L[i][j] = A[i][j]
			}
		}
	}

	return L, U
}

func invertMatrix(matrix [][]float64, c *mpi.Communicator) [][]float64 {
	identity := generateIdentityMatrix(int32(len(matrix)))
	I := generateEmptyMatrix(int32(len(matrix)))
	L, U := extractLU(matrix)

	rank := c.Rank()
	size := c.Size()
	n := len(matrix)

	for i := 0; i < n; i++ {
		if i%size == rank {
			Y := forwardSubstitution(L, identity[i])

			// Вот здесь есть лютый подвох. По нормальному при подсчете матрицы через LU мы заполняем столбцы поочередно
			// Но так как у нас гошка и я не хочу плодить лишний буфер с циклом, то я сделал так
			// Я заполняю построчно, так проще и быстрее, но матрица итоговая будет транспонирована
			// Впринципе если надо можно привести ее в норму, но это уже после
			I[i] = backwardSubstitution(U, Y)

			if rank != 0 {
				c.SendFloat64s(I[i], 0, i)
			}
		}
	}

	if rank == 0 {
		for i := 0; i < n; i++ {
			if i%size != 0 {
				I[i], _ = c.RecvFloat64s(i%size, i)
			}
		}
	}

	return I
}

func multiplyMatrices(A [][]float64, B [][]float64) [][]float64 {
	result := make([][]float64, len(A))
	for i := range result {
		result[i] = make([]float64, len(B[0]))
	}

	for i := 0; i < len(A); i++ {
		for j := 0; j < len(B[0]); j++ {
			sum := 0.0
			for k := 0; k < len(A[0]); k++ {
				sum += A[i][k] * B[k][j]
			}
			result[i][j] = sum
		}
	}

	return result
}

func generateRandomMatrix(n int32) [][]float64 {
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float64() * 10
		}
	}
	return matrix
}

func generateEmptyMatrix(n int32) [][]float64 {
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
	}
	return matrix
}

func generateIdentityMatrix(n int32) [][]float64 {
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
		matrix[i][i] = 1
	}
	return matrix
}

func printMatrix(matrix [][]float64) {
	for i := 0; i < len(matrix); i++ {
		fmt.Printf("---------")
	}
	fmt.Println()

	for _, row := range matrix {
		fmt.Printf("|")

		for i, val := range row {
			if i+1 < len(row) {
				fmt.Printf("%7.3f, ", val)
			} else {
				fmt.Printf("%7.3f", val)
			}
		}

		fmt.Printf("|\n")
	}

	for i := 0; i < len(matrix); i++ {
		fmt.Printf("---------")
	}
	fmt.Println()
}

func forwardSubstitution(L [][]float64, B []float64) []float64 {
	Y := make([]float64, len(B))

	for i, b := range B {
		sum := 0.0

		for j := 0; j < i; j++ {
			sum += L[i][j] * Y[j]
		}

		Y[i] = (b - sum) / L[i][i]
	}
	return Y
}

func backwardSubstitution(U [][]float64, Y []float64) []float64 {
	X := make([]float64, len(Y))

	for i := len(Y) - 1; i >= 0; i-- {
		sum := 0.0

		for j := i + 1; j < len(Y); j++ {
			sum += U[i][j] * X[j]
		}

		X[i] = (Y[i] - sum) / U[i][i]
	}

	return X
}

func luDecomposeParallel(A [][]float64, c *mpi.Communicator) {
	rank := c.Rank()
	size := c.Size()
	n := len(A)

	for i := 0; i < n; i++ {
		c.BcastFloat64s(A[i], i%size)

		j := rank

		for ; j < i+1; j += size {
		}

		for ; j < n; j += size {
			A[j][i] = A[j][i] / A[i][i]

			for k := i + 1; k < n; k++ {
				A[j][k] = A[j][k] - A[i][k]*A[j][i]
			}
		}
	}

	if rank != 0 {
		for i := rank; i < n; i += size {
			c.SendFloat64s(A[i], 0, i)
		}
	} else {
		for i := 0; i < n; i++ {
			if i%size != 0 {
				A[i], _ = c.RecvFloat64s(i%size, i)
			}
		}
	}

}

func sliceFromElements[T any](elems ...T) (slice []T) {
	for _, elem := range elems {
		slice = append(slice, elem)
	}

	return slice
}

func main() {
	mpi.Start(true)
	defer mpi.Stop()

	c := mpi.NewCommunicator(nil)

	n := int32(0)

	if c.Rank() == 0 {
		fmt.Printf("Enter matrix size: ")
		fmt.Scanf("%d", &n)
	}
	c.Barrier()

	buf := sliceFromElements(n)
	c.BcastInt32s(buf, 0)

	n = buf[0]

	A := generateEmptyMatrix(n)
	if c.Rank() == 0 {
		A = generateRandomMatrix(n)
	}

	for _, slice := range A {
		c.BcastFloat64s(slice, 0)
	}

	//if c.Rank() == 0 {
	//	printMatrix(A)
	//}

	start := time.Now()

	luDecomposeParallel(A, c)
	invertMatrix(A, c)
	//I := invertMatrix(A, c)

	end := time.Now()

	if c.Rank() == 0 {
		//printMatrix(I)
		fmt.Println("Time:", end.Sub(start).Milliseconds())
	}

	//B := invertMatrix(A,C)
	//printMatrix(B)
	//
	//printMatrix(multiplyMatrices(A, B))
}
