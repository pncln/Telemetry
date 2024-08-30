#include <iostream>
#include <string>
#include <bitset>
#include <vector>
#include <cmath>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

std::vector<int> stringToSignal(const std::string& input) {
    std::vector<int> signal;
    for (char c : input) {
        std::bitset<8> binary(c);
        for (size_t i = 0; i < binary.size(); ++i) {
            signal.push_back(binary[i]);
        }
    }
    return signal;
}

std::string signalToString(const std::vector<int>& signal) {
    std::string output;
    size_t numBits = signal.size();

    for (size_t i = 0; i < numBits; i += 8) {
        std::bitset<8> binary;
        for (size_t j = 0; j < 8; ++j) {
            binary[7 - j] = signal[i + j];
        }

        char character = static_cast<char>(binary.to_ulong());
        output += character;
    }

    return output;
}

std::vector<double> signalToSineWave(const std::vector<int>& signal, double frequency, double amplitude, double sampling_rate) {
    std::vector<double> sineWave;
    double time_step = 1.0 / sampling_rate;
    double omega = 2 * M_PI * frequency;

    for (int bit : signal) {
        for (double t = 0; t < 1.0 / frequency; t += time_step) {
            double value;
            if (bit == 1) {
                value = amplitude * std::sin(omega * t);
            } else {
                value = amplitude * std::sin(omega * t + M_PI);
            }
            sineWave.push_back(value);
        }
    }

    return sineWave;
}

std::vector<double> applyMovingAverageFilter(const std::vector<double>& signal, int window_size) {
    std::vector<double> filteredSignal(signal.size());

    for (size_t i = 0; i < signal.size(); ++i) {
        double sum = 0.0;
        int count = 0;
        for (int j = -window_size; j <= window_size; ++j) {
            if (i + j >= 0 && i + j < signal.size()) {
                sum += signal[i + j];
                ++count;
            }
        }
        filteredSignal[i] = sum / count;
    }

    return filteredSignal;
}

int main() {
    // std::vector<int> signal = {0, 1, 0, 0, 1, 0, 0, 0,
    //                            0, 1, 1, 0, 0, 1, 0, 1,
    //                            0, 1, 1, 0, 1, 1, 0, 0,
    //                            0, 1, 1, 0, 1, 1, 0, 0,
    //                            0, 1, 1, 0, 1, 1, 1, 1};

    std::string input;
    std::cout << "Enter a string to convert to signal: ";
    std::getline(std::cin, input);

    std::vector<int> signal = stringToSignal(input);

    std::vector<int> x(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        x[i] = static_cast<int>(i);
    }

    plt::step(x, signal);
    plt::ylim(-0.5, 1.5);
    plt::xlabel("Bit index");
    plt::ylabel("Signal");
    plt::title("String to Signal Conversion");
    plt::show();

    return 0;
}

int main() {
    std::vector<int> signal = {0, 1, 0, 0, 1, 0, 0, 0,
                               0, 1, 1, 0, 0, 1, 0, 1,
                               0, 1, 1, 0, 1, 1, 0, 0,
                               0, 1, 1, 0, 1, 1, 0, 0,
                               0, 1, 1, 0, 1, 1, 1, 1};

    // wave parameters
    double frequency = 5.0;  // Frequency
    double amplitude = 1.0;  // Amplitude
    double sampling_rate = 100.0;  // Sampling rate

    std::vector<double> sineWave = signalToSineWave(signal, frequency, amplitude, sampling_rate);

    int window_size = 5;
    std::vector<double> filteredSineWave = applyMovingAverageFilter(sineWave, window_size);

    std::vector<double> time(sineWave.size());
    double time_step = 1.0 / sampling_rate;
    for (size_t i = 0; i < sineWave.size(); ++i) {
        time[i] = i * time_step;
    }

    // Original / filtered signal wave
    plt::figure_size(1200, 600);
    plt::subplot(2, 1, 1);
    plt::plot(time, sineWave);
    plt::xlabel("Time (s)");
    plt::ylabel("Amplitude");
    plt::title("Original Sine Wave");

    plt::subplot(2, 1, 2);
    plt::plot(time, filteredSineWave);
    plt::xlabel("Time (s)");
    plt::ylabel("Amplitude");
    plt::title("Filtered Sine Wave with Moving Average");

    plt::show();

    return 0;
}