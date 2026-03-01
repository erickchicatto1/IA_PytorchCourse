#include <torch/torch.h>
#include <iostream>

// =======================================
// Definir la red neuronal
// =======================================
struct NeuralNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    NeuralNet() {
        fc1 = register_module("fc1", torch::nn::Linear(2, 16));
        fc2 = register_module("fc2", torch::nn::Linear(16, 16));
        fc3 = register_module("fc3", torch::nn::Linear(16, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::sigmoid(fc3->forward(x));
        return x;
    }
};

// =======================================
//  Main
// =======================================
int main() {
    torch::manual_seed(0);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Device: " << device << std::endl;

    // Crear datos aleatorios
    int n_samples = 1000;
    auto X = torch::rand({n_samples, 2}) * 2 - 1;
    auto y = ((X.index({torch::indexing::Slice(), 0}).pow(2) +
               X.index({torch::indexing::Slice(), 1}).pow(2)) > 1)
               .to(torch::kFloat32)
               .unsqueeze(1);

    X = X.to(device);
    y = y.to(device);

    NeuralNet model;
    model.to(device);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.01));

    // =======================================
    //  Entrenamiento
    // =======================================
    for (int epoch = 0; epoch < 50; epoch++) {
        model.train();

        auto output = model.forward(X);
        auto loss = torch::binary_cross_entropy(output, y);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch " << epoch + 1
                      << " Loss: " << loss.item<float>() << std::endl;
        }
    }

    // =======================================
    // Evaluación
    // =======================================
    model.eval();
    auto preds = model.forward(X);
    auto predicted = (preds > 0.5).to(torch::kFloat32);

    auto accuracy = predicted.eq(y).sum().item<float>() / n_samples;
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}
