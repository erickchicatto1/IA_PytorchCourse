#include <iostream>
#include <cmath>

using namespace std;

// Funcion que queremos minimizar
double f(double x) {
    return x * x;
}

// Derivada de la funcion
double gradiente(double x) {
    return 2 * x;
}

int main() {

    double x = 5.0;          // punto inicial
    double learning_rate = 0.1;
    int iteraciones = 50;

    for (int i = 0; i < iteraciones; i++) {
        double grad = gradiente(x);
        x = x - learning_rate * grad;

        cout << "Iteracion " << i
             << " x = " << x
             << " f(x) = " << f(x)
             << endl;
    }

    cout << "Minimo aproximado en x = " << x << endl;

    return 0;
}
