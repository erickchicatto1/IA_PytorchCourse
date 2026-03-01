#include <torch/torch.h>
#include <iostream>

// ==================================================
// Dataset personalizado
// ==================================================

class SimpleDataset : public torch::data::datasets::Dataset<SimpleDataset> {
private:
    torch::Tensor x_, y_;

public:
    SimpleDataset() {
        x_ = torch::linspace(-10, 10, 200).reshape({-1, 1});
        y_ = 2 * x_ + 1 + torch::randn(x_.sizes()) * 2;
    }

    torch::data::Example<> get(size_t index) override {
        return {x_[index], y_[index]};
    }

    torch::optional<size_t> size() const override {
        return x_.size(0);
    }
};

// ==================================================
//  Modelo
// ==================================================

struct LinearRegressionModel : torch::nn::Module {
    torch::nn::Linear linear{nullptr};

    LinearRegressionModel() {
        linear = register_module("linear", torch::nn::Linear(1, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear->forward(x);
    }
};

// ==================================================
// Main
// ==================================================

int main() {

    torch::manual_seed(0);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Usando dispositivo: " << device << std::endl;

    // Dataset y DataLoader
    auto dataset = SimpleDataset()
        .map(torch::data::transforms::Stack<>());

    auto dataloader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(16).shuffle(true)
    );

    // Modelo
    LinearRegressionModel model;
    model.to(device);

    // Loss y Optimizer
    torch::nn::MSELoss criterion;
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01));

    int epochs = 50;

    // ==================================================
    //  Entrenamiento
    // ==================================================
    for (int epoch = 0; epoch < epochs; epoch++) {
        model.train();
        float total_loss = 0.0;

        for (auto& batch : *dataloader) {

            auto batch_x = batch.data.to(device);
            auto batch_y = batch.target.to(device);

            auto predictions = model.forward(batch_x);
            auto loss = criterion(predictions, batch_y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
        }

        float avg_loss = total_loss / dataset.size().value();

        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch [" << epoch + 1 << "/" << epochs
                      << "] Loss: " << avg_loss << std::endl;
        }
    }

    // ==================================================
    //  Evaluación
    // ==================================================
    model.eval();
    torch::NoGradGuard no_grad;

    auto test_input = torch::tensor({{4.0}}).to(device);
    auto prediction = model.forward(test_input);

    std::cout << "\nPredicción para x=4: "
              << prediction.item<float>() << std::endl;

    // ==================================================
    //  Mostrar parámetros
    // ==================================================
    for (const auto& param : model.named_parameters()) {
        std::cout << "\n" << param.key()
                  << ":\n" << param.value() << std::endl;
    }

    return 0;
}
