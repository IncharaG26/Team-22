# Distributed Sorting Using MPI

**Team 22 | Distributed Systems Project**

## 1. Project Overview

Distributed Sorting Using MPI is a parallel computing project that implements and evaluates distributed sorting algorithms using the **Message Passing Interface (MPI)**.

The project aims to sort large datasets by distributing data across multiple processes, performing local sorting, exchanging data between processes, and combining the results into a globally sorted sequence.

Two distributed sorting algorithms are implemented and compared:

- **Sample Sort**
- **Bitonic Sort**

The implementation is developed using **C++17 and OpenMPI** and evaluated using synthetic datasets and a real-world dataset from the UCI Machine Learning Repository.

## 2. Objectives

- Implement distributed sorting using MPI.
- Understand parallel processing and inter-process communication.
- Compare Sample Sort and Bitonic Sort.
- Evaluate different splitter selection strategies.
- Analyse the effect of input distributions and process counts.
- Measure execution time, communication overhead, throughput, and load balancing.
- Validate sorting correctness.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| C++17 | Algorithm implementation |
| MPI | Distributed communication |
| OpenMPI | MPI execution environment |
| Google Colab | Development and experimentation |
| Python | Dataset preprocessing and result analysis |
| Pandas | Dataset handling |
| Matplotlib | Performance visualisation |

## 4. Algorithms Implemented

### 4.1 Sample Sort

Sample Sort divides the input dataset among multiple MPI processes. Each process locally sorts its assigned elements, selects samples, and participates in determining splitters.

The splitters divide the data into buckets. The processes exchange elements using MPI communication, followed by merging the received sorted elements to obtain the final globally sorted sequence.

Splitter strategies implemented:

- **Regular Sampling:** Selects samples at regular intervals.
- **Random Sampling:** Selects samples randomly.
- **Oversampling:** Selects additional samples to improve partition balance.

### 4.2 Bitonic Sort

Bitonic Sort is a comparison-based parallel sorting algorithm.

The input is distributed across MPI processes. Processes perform local sorting and participate in a sequence of compare-and-exchange operations with partner processes to achieve a globally sorted result.

Bitonic Sort requires a power-of-two number of MPI processes in this implementation.

## 5. Dataset

The project uses the **Online Retail Dataset** from the UCI Machine Learning Repository.

**Dataset details:**

- Dataset Name: Online Retail
- Source: UCI Machine Learning Repository
- Number of records: 541,909
- Format: Excel (.xlsx)
- Sorting attribute: UnitPrice

The UnitPrice column is extracted from the original dataset and converted into integer pence values for sorting.

**Dataset link:**  
https://archive.ics.uci.edu/dataset/352/online-retail

The dataset documentation and any uploaded dataset files are available in the `dataset/` directory.

## 6. Project Workflow

1. Load the input dataset.
2. Extract and preprocess the numerical sorting keys.
3. Divide the input data among MPI processes.
4. Perform local sorting on individual processes.
5. Execute distributed sorting using the selected algorithm.
6. Exchange data between processes.
7. Merge or finalize the sorted output.
8. Verify global sorting correctness.
9. Collect performance metrics.
10. Analyse and visualise experimental results.

## 7. Performance Metrics

The following metrics are used to evaluate the algorithms:

| Metric | Description |
|---|---|
| Total Execution Time | Overall time taken for sorting |
| Local Sorting Time | Time spent sorting locally |
| Communication Time | Time spent exchanging data between processes |
| Merge Time | Time spent combining sorted data |
| Throughput | Number of elements processed per second |
| Load Imbalance | Measures the distribution of workload among processes |
| Correctness | Verifies whether the final output is sorted correctly |

## 8. Experimental Evaluation

Experiments are conducted to analyse:

- Sample Sort versus Bitonic Sort.
- Regular, random, and oversampling splitter strategies.
- Different input distributions such as uniform, Gaussian, sorted, reverse, and duplicate values.
- Different MPI process counts.
- Performance on different dataset sizes using the UCI Online Retail dataset.

The results are collected in CSV format and visualised using Python.

## 9. Repository Structure

```text
Team-22/
│
├── dataset/
│   ├── README.md
│   └── Online Retail.xlsx
│
├── team22_distributed_sort.cpp
│
├── Team22_Distributed_Sorting_UCI_Online_Retail.ipynb
│
├── team22_uci_retail_results.csv
│
└── README.md
```

## 10. Installation and Execution

### Requirements

- C++ compiler supporting C++17
- OpenMPI
- Python 3
- Google Colab or Linux environment

### Install OpenMPI

On Ubuntu or Google Colab:

```bash
sudo apt-get update
sudo apt-get install openmpi-bin libopenmpi-dev g++
```

### Compile the C++ Program

```bash
mpic++ -O3 -std=c++17 team22_distributed_sort.cpp -o team22_sort
```

### Run with MPI

The program supports file-based execution using the following command format:

```bash
mpirun --allow-run-as-root --oversubscribe -np 4 ./team22_sort --file INPUT.txt sample oversample 4
```

For Bitonic Sort:

```bash
mpirun --allow-run-as-root --oversubscribe -np 4 ./team22_sort --file INPUT.txt bitonic regular 4
```

**Note:** The input file must contain one integer sorting key per line. The Google Colab notebook includes dataset preprocessing and experiment execution instructions.

## 11. Results and Analysis

Experimental results are stored in CSV format and analysed through graphs in the Google Colab notebook.

The analysis compares execution time, throughput, communication overhead, correctness, and load balancing across algorithms and configurations.

Actual performance depends on dataset size, process count, hardware resources, and communication overhead.

## 12. Conclusion

This project demonstrates how distributed sorting algorithms can be implemented using MPI to perform parallel data processing.

By implementing Sample Sort and Bitonic Sort, the project evaluates the impact of algorithm selection, splitter strategies, input characteristics, and process count on sorting performance.

The experiments provide insights into the trade-off between computation, communication overhead, and workload distribution in distributed systems.

## 13. References

1. UCI Machine Learning Repository – Online Retail Dataset: https://doi.org/10.24432/C5BW33
2. MPI Forum – Message Passing Interface Standard: https://www.mpi-forum.org/
3. OpenMPI Documentation: https://docs.open-mpi.org/

---

**Project:** Distributed Sorting Using MPI  
**Team:** 22  
**Language:** C++17  
**Framework:** MPI / OpenMPI
