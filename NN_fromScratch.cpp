#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>

using namespace std;
using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> Mat;
typedef Matrix<float, 1, Dynamic> RowVec;

class NeuralNetwork {

public:

    vector<int> layers;
    vector<Mat> W;
    vector<RowVec> B;

    NeuralNetwork(const vector<int>& layers) : layers(layers) {

        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> dist(0,1);

        W.reserve(layers.size()-1);
        B.reserve(layers.size()-1);

        for(size_t i=0;i<layers.size()-1;i++){

            Mat w(layers[i], layers[i+1]);

            for(int r=0;r<w.rows();r++)
                for(int c=0;c<w.cols();c++)
                    w(r,c)=dist(gen)*0.1f;

            W.push_back(w);
            B.push_back(RowVec::Zero(layers[i+1]));
        }
    }

    inline Mat relu(const Mat& x){
        return x.cwiseMax(0.0f);
    }

    inline Mat relu_derivative(const Mat& x){
        return (x.array()>0).cast<float>();
    }

    void forward(
        const Mat& X,
        vector<Mat>& A,
        vector<Mat>& Z){

        A.clear();
        Z.clear();

        A.push_back(X);

        for(size_t i=0;i<W.size();i++){

            Mat z(A.back().rows(), W[i].cols());

            z.noalias() = A.back() * W[i];
            z.rowwise() += B[i];

            Z.push_back(z);

            if(i != W.size()-1)
                A.push_back(relu(z));
            else
                A.push_back(z);
        }
    }

    float mse(const Mat& pred,const Mat& y){

        return (pred-y).array().square().mean();
    }

    void train(const Mat& X,const Mat& y,int epochs,float lr){

        vector<Mat> A,Z;
        vector<Mat> dW(W.size());
        vector<RowVec> dB(W.size());

        for(int epoch=0;epoch<epochs;epoch++){

            forward(X,A,Z);

            int m = y.rows();

            Mat delta = (A.back()-y)*(2.0f/m);

            for(int i=W.size()-1;i>=0;i--){

                dW[i].resize(W[i].rows(),W[i].cols());
                dW[i].noalias() = A[i].transpose()*delta;

                dB[i] = delta.colwise().sum();

                if(i>0){

                    Mat new_delta(delta.rows(),W[i].rows());

                    new_delta.noalias() = delta * W[i].transpose();

                    delta = new_delta.array() *
                            relu_derivative(Z[i-1]).array();
                }
            }

            for(size_t i=0;i<W.size();i++){

                W[i].noalias() -= lr * dW[i];
                B[i] -= lr * dB[i];
            }

            if(epoch % 100 == 0)
                cout<<"Epoch "<<epoch<<" Loss "<<mse(A.back(),y)<<endl;
        }
    }

    Mat predict(const Mat& X){

        Mat a = X;

        for(size_t i=0;i<W.size();i++){

            Mat z(a.rows(),W[i].cols());

            z.noalias() = a * W[i];
            z.rowwise() += B[i];

            if(i!=W.size()-1)
                a = relu(z);
            else
                a = z;
        }

        return a;
    }
};

int main(){

    Mat X(5,3);
    X << 60,10,1012,
         65,12,1010,
         70,8,1008,
         75,7,1005,
         80,5,1003;

    Mat y(5,1);
    y << 25,26,27,28,30;

    NeuralNetwork nn({3,8,8,1});

    nn.train(X,y,2000,0.00001f);

    Mat sample(1,3);
    sample << 72,9,1007;

    cout<<"Prediction: "<<nn.predict(sample)<<endl;
}
