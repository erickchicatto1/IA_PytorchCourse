#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Función de activación sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivada de sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class NeuralNetwork {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    vector<vector<double>> weights_input_hidden;
    vector<vector<double>> weights_hidden_output;

    vector<double> hidden;
    vector<double> output;

    double learningRate;

public:
    NeuralNetwork(int in, int hid, int out, double lr) {
        inputSize = in;
        hiddenSize = hid;
        outputSize = out;
        learningRate = lr;

        srand(time(0));

        // Inicializar pesos aleatorios
        weights_input_hidden.resize(inputSize, vector<double>(hiddenSize));
        weights_hidden_output.resize(hiddenSize, vector<double>(outputSize));

        for(int i = 0; i < inputSize; i++)
            for(int j = 0; j < hiddenSize; j++)
                weights_input_hidden[i][j] = ((double) rand() / RAND_MAX) - 0.5;

        for(int i = 0; i < hiddenSize; i++)
            for(int j = 0; j < outputSize; j++)
                weights_hidden_output[i][j] = ((double) rand() / RAND_MAX) - 0.5;

        hidden.resize(hiddenSize);
        output.resize(outputSize);
    }

    vector<double> forward(vector<double> inputs) {

        // Capa oculta
        for(int j = 0; j < hiddenSize; j++) {
            double sum = 0.0;
            for(int i = 0; i < inputSize; i++) {
                sum += inputs[i] * weights_input_hidden[i][j];
            }
            hidden[j] = sigmoid(sum);
        }

        // Capa de salida
        for(int j = 0; j < outputSize; j++) {
            double sum = 0.0;
            for(int i = 0; i < hiddenSize; i++) {
                sum += hidden[i] * weights_hidden_output[i][j];
            }
            output[j] = sigmoid(sum);
        }

        return output;
    }

    void train(vector<double> inputs, vector<double> targets) {
        forward(inputs);

        vector<double> output_errors(outputSize);
        vector<double> hidden_errors(hiddenSize);

        // Error en salida
        for(int i = 0; i < outputSize; i++)
            output_errors[i] = targets[i] - output[i];

        // Error en capa oculta
        for(int i = 0; i < hiddenSize; i++) {
            hidden_errors[i] = 0.0;
            for(int j = 0; j < outputSize; j++)
                hidden_errors[i] += output_errors[j] * weights_hidden_output[i][j];
        }

        // Actualizar pesos hidden -> output
        for(int i = 0; i < hiddenSize; i++) {
            for(int j = 0; j < outputSize; j++) {
                weights_hidden_output[i][j] += learningRate *
                    output_errors[j] *
                    sigmoid_derivative(output[j]) *
                    hidden[i];
            }
        }

        // Actualizar pesos input -> hidden
        for(int i = 0; i < inputSize; i++) {
            for(int j = 0; j < hiddenSize; j++) {
                weights_input_hidden[i][j] += learningRate *
                    hidden_errors[j] *
                    sigmoid_derivative(hidden[j]) *
                    inputs[i];
            }
        }
    }
};

int main() {

    NeuralNetwork nn(2, 2, 1, 0.5);

    vector<vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    vector<vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };

    // Entrenamiento
    for(int epoch = 0; epoch < 10000; epoch++) {
        for(int i = 0; i < inputs.size(); i++) {
            nn.train(inputs[i], targets[i]);
        }
    }

    // Prueba
    cout << "Resultados:\n";
    for(int i = 0; i < inputs.size(); i++) {
        vector<double> out = nn.forward(inputs[i]);
        cout << inputs[i][0] << " XOR " << inputs[i][1]
             << " = " << out[0] << endl;
    }

    return 0;
}
