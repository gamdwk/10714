#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t begin=0,end;
    for(begin=0;begin<m;begin+=batch)
    {
        //end = min(y.shape[0], begin + batch)
        end = std::min(begin+batch, m);
        size_t batch_size = end - begin;
        float hx[batch_size][k];
        // hx = batch_X @ theta
        for(size_t i=0;i<batch_size;++i){
            for(size_t j=0;j<k;++j){
                hx[i][j]=0;
                for(size_t z=0;z<n;++z){
                    //X:m*n,theta:n*k
                    hx[i][j] += X[(begin+i)*n+z]*theta[z*k+j];
                }
            }
        }
        // Z = np.divide(np.exp(hx), np.reshape(np.exp(hx).sum(axis=1), (hx.shape[0], 1)))
        // Iy = np.zeros(Z.shape)
        // Iy[np.arange(Iy.shape[0]), batch_y] = 1

        float Z_sub_Iy[batch_size][k];
        for(size_t i=0;i<batch_size;++i){
            float exp_sum=0;
            for(size_t j=0;j<k;++j){
                Z_sub_Iy[i][j] = std::exp(hx[i][j]);
                exp_sum += Z_sub_Iy[i][j];
            }
            for(size_t j=0;j<k;++j){
                Z_sub_Iy[i][j] = Z_sub_Iy[i][j]/exp_sum;
                // Z - Iy
                if(j==y[begin+i]){
                    Z_sub_Iy[i][j] -= 1;
                }
            }
        }
        // gradient = (1 / batch_size) * batch_X.T @ (Z - Iy)
        // theta -= lr * gradient
        // float gradient[n][k]
        for(size_t i=0;i<n;++i){
            for(size_t j=0;j<k;++j){
                float gradient=0;
                for(size_t z=0;z<batch_size;++z){
                    //X:m*n,theta:n*k
                    gradient += X[(begin+z)*n+i]*Z_sub_Iy[z][j] / (float)(batch_size);
                }
                theta[i*k+j] -= lr * gradient;
            }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
