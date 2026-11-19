#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>
#include <fstream>

struct Record {
    int value;
    long long id;
};

struct RecordLess {
    bool operator()(const Record& a, const Record& b) const {
        if (a.value != b.value) return a.value < b.value;
        return a.id < b.id;
    }
};

static MPI_Datatype make_record_type() {
    MPI_Datatype type;
    Record sample{};
    int lengths[2] = {1, 1};
    MPI_Aint offsets[2];
    MPI_Datatype types[2] = {MPI_INT, MPI_LONG_LONG_INT};
    MPI_Get_address(&sample.value, &offsets[0]);
    MPI_Get_address(&sample.id, &offsets[1]);
    offsets[1] -= offsets[0];
    offsets[0] = 0;
    MPI_Type_create_struct(2, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

static std::vector<Record> generate_input(int n, const std::string& distribution) {
    std::vector<Record> data(n);
    std::mt19937_64 rng(1234567);
    if (distribution == "gaussian") {
        std::normal_distribution<double> dist(0.0, 100000.0);
        for (int i = 0; i < n; ++i) {
            double x = std::max(-1000000.0, std::min(1000000.0, dist(rng)));
            data[i] = {static_cast<int>(std::llround(x)), i};
        }
    } else if (distribution == "duplicates") {
        std::uniform_int_distribution<int> dist(0, 9);
        for (int i = 0; i < n; ++i) data[i] = {dist(rng), i};
    } else {
        std::uniform_int_distribution<int> dist(-1000000, 1000000);
        for (int i = 0; i < n; ++i) data[i] = {dist(rng), i};
        if (distribution == "sorted")
            std::sort(data.begin(), data.end(), RecordLess{});
        else if (distribution == "reverse") {
            std::sort(data.begin(), data.end(), RecordLess{});
            std::reverse(data.begin(), data.end());
        }
    }
    return data;
}

static void distribute_input(const std::vector<Record>& input, std::vector<Record>& local,
                             int n, int rank, int p, MPI_Datatype record_type) {
    std::vector<int> counts(p), displs(p);
    int base = n / p, rem = n % p, offset = 0;
    for (int r = 0; r < p; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
    local.resize(counts[rank]);
    MPI_Scatterv(rank == 0 ? input.data() : nullptr, counts.data(), displs.data(), record_type,
                 local.data(), counts[rank], record_type, 0, MPI_COMM_WORLD);
}

static std::vector<Record> choose_samples(const std::vector<Record>& local,
                                          const std::string& strategy, int p, int oversample,
                                          int rank) {
    std::vector<Record> samples;
    if (p <= 1 || local.empty()) return samples;
    int wanted = (strategy == "oversample") ? oversample * (p - 1) : (p - 1);
    int count = std::min<int>(wanted, local.size());
    samples.reserve(count);
    if (strategy == "random") {
        std::mt19937 rng(98765 + rank);
        std::uniform_int_distribution<int> pick(0, static_cast<int>(local.size()) - 1);
        for (int i = 0; i < count; ++i) samples.push_back(local[pick(rng)]);
    } else {
        for (int i = 0; i < count; ++i) {
            size_t index = std::min(local.size() - 1,
                static_cast<size_t>((static_cast<double>(i) + 0.5) * local.size() / count));
            samples.push_back(local[index]);
        }
    }
    return samples;
}

static std::vector<Record> sample_sort(std::vector<Record>& local, int p, int rank,
                                       MPI_Datatype record_type, const std::string& strategy,
                                       int oversample, double& local_sort_time,
                                       double& communication_time, double& merge_time) {
    double t = MPI_Wtime();
    std::sort(local.begin(), local.end(), RecordLess{});
    local_sort_time += MPI_Wtime() - t;

    t = MPI_Wtime();
    std::vector<Record> samples = choose_samples(local, strategy, p, oversample, rank);
    int sample_count = static_cast<int>(samples.size());
    std::vector<int> sample_counts(p), sample_displs(p);
    MPI_Allgather(&sample_count, 1, MPI_INT, sample_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int total_samples = 0;
    for (int r = 0; r < p; ++r) {
        sample_displs[r] = total_samples;
        total_samples += sample_counts[r];
    }
    std::vector<Record> all_samples(total_samples);
    MPI_Allgatherv(samples.data(), sample_count, record_type, all_samples.data(),
                   sample_counts.data(), sample_displs.data(), record_type, MPI_COMM_WORLD);
    std::sort(all_samples.begin(), all_samples.end(), RecordLess{});

    std::vector<Record> splitters;
    if (p > 1 && !all_samples.empty()) {
        for (int i = 1; i < p; ++i) {
            size_t index = std::min(all_samples.size() - 1,
                static_cast<size_t>((static_cast<double>(i) * all_samples.size()) / p));
            splitters.push_back(all_samples[index]);
        }
    } else if (p > 1) {
        splitters.assign(p - 1, Record{INT_MAX, LLONG_MAX});
    }

    std::vector<std::vector<Record>> buckets(p);
    auto begin = local.begin();
    for (int dest = 0; dest < p; ++dest) {
        if (dest == p - 1) {
            buckets[dest].assign(begin, local.end());
        } else {
            auto end = std::upper_bound(begin, local.end(), splitters[dest], RecordLess{});
            buckets[dest].assign(begin, end);
            begin = end;
        }
    }
    std::vector<int> send_counts(p), send_displs(p), recv_counts(p), recv_displs(p);
    int send_total = 0;
    for (int r = 0; r < p; ++r) {
        send_counts[r] = static_cast<int>(buckets[r].size());
        send_displs[r] = send_total;
        send_total += send_counts[r];
    }
    std::vector<Record> send_buffer;
    send_buffer.reserve(send_total);
    for (auto& b : buckets) send_buffer.insert(send_buffer.end(), b.begin(), b.end());
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int recv_total = 0;
    for (int r = 0; r < p; ++r) {
        recv_displs[r] = recv_total;
        recv_total += recv_counts[r];
    }
    std::vector<Record> received(recv_total);
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), record_type,
                  received.data(), recv_counts.data(), recv_displs.data(), record_type,
                  MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - t;

    t = MPI_Wtime();
    std::sort(received.begin(), received.end(), RecordLess{});
    merge_time += MPI_Wtime() - t;
    return received;
}

static std::vector<Record> bitonic_sort(std::vector<Record>& local, int n, int p, int rank,
                                        MPI_Datatype record_type, double& local_sort_time,
                                        double& communication_time, double& merge_time) {
    int chunk = (n + p - 1) / p;
    local.resize(chunk, Record{INT_MAX, LLONG_MAX});
    double t = MPI_Wtime();
    std::sort(local.begin(), local.end(), RecordLess{});
    local_sort_time += MPI_Wtime() - t;
    std::vector<Record> partner_data(chunk);

    for (int k = 2; k <= p; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = rank ^ j;
            t = MPI_Wtime();
            MPI_Sendrecv(local.data(), chunk, record_type, partner, 100,
                         partner_data.data(), chunk, record_type, partner, 100,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            communication_time += MPI_Wtime() - t;

            t = MPI_Wtime();
            std::vector<Record> combined;
            combined.reserve(2 * chunk);
            std::merge(local.begin(), local.end(), partner_data.begin(), partner_data.end(),
                       std::back_inserter(combined), RecordLess{});
            bool keep_low = ((rank & k) == 0) == ((rank & j) == 0);
            if (keep_low)
                std::copy(combined.begin(), combined.begin() + chunk, local.begin());
            else
                std::copy(combined.end() - chunk, combined.end(), local.begin());
            merge_time += MPI_Wtime() - t;
        }
    }
    local.erase(std::remove_if(local.begin(), local.end(),
                               [](const Record& x) { return x.id == LLONG_MAX; }), local.end());
    return local;
}

static bool gather_and_verify(const std::vector<Record>& local, const std::vector<Record>& original,
                              int n, int rank, int p, MPI_Datatype record_type,
                              double& communication_time) {
    double t = MPI_Wtime();
    int local_n = static_cast<int>(local.size());
    std::vector<int> counts(p), displs(p);
    MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int total = 0;
    if (rank == 0) {
        for (int r = 0; r < p; ++r) { displs[r] = total; total += counts[r]; }
    }
    std::vector<Record> all(rank == 0 ? total : 0);
    MPI_Gatherv(local.data(), local_n, record_type, all.data(), counts.data(), displs.data(),
                record_type, 0, MPI_COMM_WORLD);
    communication_time += MPI_Wtime() - t;

    int ok = 1;
    if (rank == 0) {
        if (static_cast<int>(all.size()) != n) ok = 0;
        else {
            for (size_t i = 1; i < all.size(); ++i)
                if (RecordLess{}(all[i], all[i - 1])) { ok = 0; break; }
            std::vector<unsigned char> seen(n, 0);
            if (ok) for (const auto& x : all) {
                if (x.id < 0 || x.id >= n || seen[x.id] || original[x.id].value != x.value) {
                    ok = 0; break;
                }
                seen[x.id] = 1;
            }
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok == 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Datatype record_type = make_record_type();

    int n = 1000000, oversample = 4;
    std::string algorithm = "sample", strategy = "oversample", distribution = "uniform";
    bool file_mode = (argc > 1 && std::string(argv[1]) == "--file");
    std::string input_path;

    if (file_mode) {
        if (argc < 5) {
            if (rank == 0) std::cerr << "Usage: mpirun -np P ./team22_sort --file INPUT.txt sample|bitonic regular|random|oversample [oversample_factor]\\n";
            MPI_Type_free(&record_type);
            MPI_Finalize();
            return 1;
        }
        input_path = argv[2];
        algorithm = argv[3];
        strategy = argv[4];
        if (argc > 5) oversample = std::max(1, std::stoi(argv[5]));
    } else {
        if (argc > 1) n = std::stoi(argv[1]);
        if (argc > 2) algorithm = argv[2];
        if (argc > 3) strategy = argv[3];
        if (argc > 4) distribution = argv[4];
        if (argc > 5) oversample = std::max(1, std::stoi(argv[5]));
    }

    std::vector<Record> original;
    int input_ok = 1;
    if (file_mode) {
        if (rank == 0) {
            std::ifstream in(input_path);
            if (!in) input_ok = 0;
            else {
                long long value;
                long long id = 0;
                while (in >> value) {
                    if (value < INT_MIN || value > INT_MAX) { input_ok = 0; break; }
                    original.push_back({static_cast<int>(value), id++});
                }
                n = static_cast<int>(original.size());
                if (in.bad() || n == 0) input_ok = 0;
            }
        }
        MPI_Bcast(&input_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        distribution = "uci_online_retail_unitprice";
    }

    int valid = (input_ok && n >= 0 && (algorithm == "sample" || algorithm == "bitonic") &&
                 (strategy == "regular" || strategy == "random" || strategy == "oversample") &&
                 (file_mode || distribution == "uniform" || distribution == "gaussian" ||
                  distribution == "sorted" || distribution == "reverse" ||
                  distribution == "duplicates"));
    if (algorithm == "bitonic" && (p & (p - 1)) != 0) valid = 0;
    if (n > INT_MAX / std::max(1, p)) valid = 0;
    if (!valid) {
        if (rank == 0) std::cerr << "Usage: mpirun -np P ./team22_sort N sample|bitonic regular|random|oversample uniform|gaussian|sorted|reverse|duplicates [oversample_factor]\n"
                                  << "Bitonic Sort requires P to be a power of two. N must fit MPI int counts.\n";
        MPI_Type_free(&record_type);
        MPI_Finalize();
        return 1;
    }

    if (!file_mode && rank == 0) original = generate_input(n, distribution);
    std::vector<Record> local;
    distribute_input(original, local, n, rank, p, record_type);

    double local_sort_time = 0.0, communication_time = 0.0, merge_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();
    std::vector<Record> sorted;
    if (algorithm == "sample")
        sorted = sample_sort(local, p, rank, record_type, strategy, oversample,
                             local_sort_time, communication_time, merge_time);
    else
        sorted = bitonic_sort(local, n, p, rank, record_type,
                              local_sort_time, communication_time, merge_time);
    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - total_start;

    int local_output_count = static_cast<int>(sorted.size());
    std::vector<int> output_counts(rank == 0 ? p : 0);
    MPI_Gather(&local_output_count, 1, MPI_INT, output_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    double load_imbalance = 0.0;
    if (rank == 0 && n > 0) {
        int largest = *std::max_element(output_counts.begin(), output_counts.end());
        load_imbalance = static_cast<double>(largest) / (static_cast<double>(n) / p);
    }

    bool correct = gather_and_verify(sorted, original, n, rank, p, record_type, communication_time);
    double max_total, max_local, max_comm, max_merge;
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sort_time, &max_local, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&communication_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&merge_time, &max_merge, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Algorithm: " << algorithm << "\nProcesses: " << p
                  << "\nInput keys: " << n << "\nDistribution: " << distribution;
        if (algorithm == "sample") std::cout << "\nSplitter strategy: " << strategy;
        std::cout << "\nCorrectness: " << (correct ? "PASS" : "FAIL")
                  << "\nTotal time (max rank, s): " << max_total
                  << "\nLocal sort time (max rank, s): " << max_local
                  << "\nCommunication time (max rank, s): " << max_comm
                  << "\nMerge/final-sort time (max rank, s): " << max_merge
                  << "\nKeys/second: " << (max_total > 0 ? n / max_total : 0)
                  << "\nLoad imbalance (max rank keys / average keys): " << load_imbalance
                  << "\nPer-rank output counts: ";
        for (int r = 0; r < p; ++r) std::cout << output_counts[r] << (r + 1 == p ? "\n" : ", ");
    }

    MPI_Type_free(&record_type);
    MPI_Finalize();
    return correct ? 0 : 2;
}
