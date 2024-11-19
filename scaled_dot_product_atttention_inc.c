// Scaled Dot-Product Attention in C

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to display a matrix
void displayMatrix(float* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Matrix multiplication
void matMul(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i * colsB + j] = 0;
            for (int k = 0; k < colsA; k++) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

// Transpose a matrix
void transpose(float* A, float* At, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            At[j * rows + i] = A[i * cols + j];
        }
    }
}

// Apply softmax to a matrix row-wise
void softmax(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float maxVal = -INFINITY;
        for (int j = 0; j < cols; j++) {
            if (matrix[i * cols + j] > maxVal) {
                maxVal = matrix[i * cols + j];
            }
        }

        float sum = 0.0;
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = exp(matrix[i * cols + j] - maxVal);
            sum += matrix[i * cols + j];
        }

        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] /= sum;
        }
    }
}

// Scaled dot-product attention
void scaledDotProductAttention(float* Q, float* K, float* V, float* output, int d_k, int seq_len) {
    float* K_t = (float*)malloc(seq_len * d_k * sizeof(float)); // Transposed K
    float* QK = (float*)malloc(seq_len * seq_len * sizeof(float)); // Q * K^T
    float scale_factor = 1.0 / sqrt(d_k);

    // Step 1: Transpose K
    transpose(K, K_t, seq_len, d_k);

    // Step 2: Compute Q * K^T
    matMul(Q, K_t, QK, seq_len, d_k, seq_len);

    // Step 3: Scale QK by sqrt(d_k)
    for (int i = 0; i < seq_len * seq_len; i++) {
        QK[i] *= scale_factor;
    }

    // Step 4: Apply softmax to QK
    softmax(QK, seq_len, seq_len);

    // Step 5: Compute QK * V
    matMul(QK, V, output, seq_len, seq_len, d_k);

    // Free allocated memory
    free(K_t);
    free(QK);
}

int main() {
    // Example input
    const int seq_len = 3; // Sequence length
    const int d_k = 3;     // Dimensionality of keys, queries, values

    // Query matrix (Q)
    float Q[] = {
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 1.0
    };

    // Key matrix (K)
    float K[] = {
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 1.0
    };

    // Value matrix (V)
    float V[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };

    // Output matrix
    float output[seq_len * d_k];

    // Perform scaled dot-product attention
    scaledDotProductAttention(Q, K, V, output, d_k, seq_len);

    // Display matrices
    displayMatrix(Q, seq_len, d_k, "Query (Q)");
    displayMatrix(K, seq_len, d_k, "Key (K)");
    displayMatrix(V, seq_len, d_k, "Value (V)");
    displayMatrix(output, seq_len, d_k, "Output");

    return 0;
}
// I am just curious about this, like about it's implementation.
