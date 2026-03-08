#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

// Función para imprimir la matriz extendida
void print_square_matrix(double **A, double sol_vector[], int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cout << setw(10) << A[i][j] << " ";
        }
        cout << " | " << setw(10) << sol_vector[i] << endl;
    }
}

void forward_elimination(double **A, double sol_vector[], int n) {
    for(int k = 0; k < n; k++) {
        // Pivoteo parcial simple (evitar división por cero)
        if (abs(A[k][k]) < 1e-12) {
            for (int i = k + 1; i < n; i++) {
                if (abs(A[i][k]) > abs(A[k][k])) {
                    swap(A[k], A[i]);
                    swap(sol_vector[k], sol_vector[i]);
                    break;
                }
            }
        }

        for(int i = k + 1; i < n; i++) {
            double factor = A[i][k] / A[k][k];
            for(int j = k; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            sol_vector[i] -= factor * sol_vector[k];
        }
    }
}

void back_substitution(double **A, double sol_vector[], int n, double solution[]) {
    for(int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for(int j = i + 1; j < n; j++) {
            sum += A[i][j] * solution[j];
        }
        solution[i] = (sol_vector[i] - sum) / A[i][i];
    }
}

void print_poly(double v[], int n) {
    cout << "f(x) = ";
    for(int i = 0; i < n; i++){
        if(i > 0 && v[i] >= 0) cout << " + ";
        if(v[i] < 0) cout << " - ";
        
        cout << abs(v[i]);
        if(i > 0) cout << "x^" << i;
    }
    cout << endl;
}

void polynomial_regression(double x[], double y[], int degree, int n) {
    int m = degree + 1;
    double **A = new double*[m];
    for(int i = 0; i < m; i++) A[i] = new double[m];
    double *B = new double[m];
    double *coefficients = new double[m];

    // 1. Construir la matriz de ecuaciones normales
    // La matriz es simétrica, podemos optimizar calculando las sumas de potencias una vez
    double *sum_x = new double[2 * degree + 1];
    for(int i = 0; i <= 2 * degree; i++) {
        sum_x[i] = 0;
        for(int j = 0; j < n; j++) sum_x[i] += pow(x[j], i);
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < m; j++) {
            A[i][j] = sum_x[i + j];
        }
        
        double sum_xy = 0;
        for(int j = 0; j < n; j++) sum_xy += pow(x[j], i) * y[j];
        B[i] = sum_xy;
    }

    cout << "\nSistema de Ecuaciones:" << endl;
    print_square_matrix(A, B, m);

    // 2. Resolver el sistema
    forward_elimination(A, B, m);
    back_substitution(A, B, m, coefficients);

    cout << "\nCoeficientes encontrados:" << endl;
    for(int i = 0; i < m; i++) cout << "a" << i << " = " << coefficients[i] << endl;

    print_poly(coefficients, m);

    // 3. Análisis de error
    double sr = 0, st = 0, y_sum = 0;
    for(int i = 0; i < n; i++) y_sum += y[i];
    double y_avg = y_sum / n;

    for(int i = 0; i < n; i++) {
        double y_pred = 0;
        for(int j = 0; j < m; j++) y_pred += coefficients[j] * pow(x[i], j);
        sr += pow(y[i] - y_pred, 2);
        st += pow(y[i] - y_avg, 2);
    }

    double r2 = (st - sr) / st;
    cout << "\nEstadisticas:" << endl;
    cout << "Error estandar (Sy/x): " << sqrt(sr / (n - m)) << endl;
    cout << "Coeficiente de determinacion (r2): " << r2 << endl;

    // Liberar memoria
    for(int i = 0; i < m; i++) delete[] A[i];
    delete[] A;
    delete[] B;
    delete[] coefficients;
    delete[] sum_x;
}

int main() {
    int n, degree;
    cout << "Numero de puntos (n): "; cin >> n;
    cout << "Grado del polinomio: "; cin >> degree;

    // Cambia las rutas según tu estructura de carpetas
    ifstream inx("x_2.txt");
    ifstream iny("y_2.txt");

    if(!inx || !iny){
        cerr << "Error al abrir los archivos de datos." << endl;
        return 1;
    }

    double *X = new double[n];
    double *Y = new double[n];

    for(int i = 0; i < n; i++) {
        inx >> X[i];
        iny >> Y[i];
    }

    polynomial_regression(X, Y, degree, n);

    delete[] X;
    delete[] Y;
    return 0;
}
