build:
	go build -o mpi main.go

NP ?= 4

.PHONY: mpi
mpi:
	make build
	mpirun -np $(NP) ./mpi
