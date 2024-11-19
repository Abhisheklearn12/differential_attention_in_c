//Trying to learn Differntial attention in C

#include <stdio.h>

// Function to calculate weighted attention scores
void calculateAttention(float inputs[], float weights[], float output[], int size) {
    float total_weight = 0.0;

    // Calculate the total weight
    for (int i = 0; i < size; i++) {
        total_weight += weights[i];
    }

    // Compute the attention-weighted output
    for (int i = 0; i < size; i++) {
        output[i] = (weights[i] / total_weight) * inputs[i];
    }
}

int main() {
    // Example inputs and weights
    float inputs[] = {1.0, 2.0, 3.0, 4.0};
    float weights[] = {0.1, 0.2, 0.3, 0.4};
    int size = sizeof(inputs) / sizeof(inputs[0]);

    // Array to store output
    float output[size];

    // Calculate attention
    calculateAttention(inputs, weights, output, size);

    // Print the results
    printf("Inputs:   ");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", inputs[i]);
    }
    printf("\nWeights:  ");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", weights[i]);
    }
    printf("\nOutputs:  ");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", output[i]);
    }
    printf("\n");

    return 0;
}
 // A very basic version of differential attention, just for learning purpose trying to figure out to make it more advanced.