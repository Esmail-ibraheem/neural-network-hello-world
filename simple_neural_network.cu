//perceptron.cpp

#include <iostream>

using namespace std;

// Simple Perceptron class
class Perceptron 
{
private:
    double *w;	//weights
    double *x;	//inputs
    int n1;	//n1=n+1, x[0], .... x[n], x[0] is 1
public:
    Perceptron(int nx, double weights[]) 
    {
	n1 = nx;
	w =  new double[nx];
	for(int i = 0; i < n1; i++)
	  w[i] = weights[i];
    }

    //Activation function (Step function)
    int activation(double value) 
    {
        return value >= 0 ? 1 : 0;
    }

    // Predicted output 
    int predictedOutput(double xi[]) 
    {
	x = xi;
        double sum = 0;
        for (int i = 0; i < n1; ++i) 
            sum += x[i] * w[i];
        
	 int y1 = activation(sum);

         return y1;
    }

  // Train the perceptron
  // eta = learning rate
  // epochs = number of times of training 
  // numSamples = number of different input sets (x_data)
  // y is target output
  void train(double *x_data, double *y, double eta, int numSamples, int epochs)
  {
    double *x;

    for (int m = 0; m < epochs; m++){
      for (int k = 0; k < numSamples; k++) {
        x = x_data + k * n1;
        int y1 = predictedOutput( x );
        int error = y[k] - y1;

        // Update weights and bias
        for (int i = 0; i < n1; i++) {
          double dwi = eta * error * x[i];
          w[i] = w[i] + dwi;
        }
      }
    }
  }

    void printWeights()
    {
	cout << "\nPerceptron weights: ";
	for(int i = 0; i < n1; i++)
	  cout << "\n  w" << i <<": " << w[i];
    }

    ~Perceptron()
    {
	delete w;
    }
};


int main() 
{
    // Training data for AND gate, four sets of X
    double inputs[] = 
    {
        1, 0, 0,	//x0=1, x1=0, x2=1
        1, 0, 1,	//x0=1, x1=0, x2=1
        1, 1, 0,	//x0=1, x1=1, x2=0
        1, 1, 1		//x0=1, x1=1, x2=1
    };

    double y[] = {0, 0, 0, 1};	//target outputs (labels)
    double w[] = {1, 1, 1};

    string gates = " AND ";
    
    //Construct a perception with a bias and 2 inputs 
    Perceptron perceptron(3, w);
    
    // Train the perceptron
    perceptron.train(inputs, y, 0.1, 4, 10);

    // Test the perceptron
    double x[3];
    cout << "Testing Perceptron:" << endl;
    x[0] = 1;
    for (int i = 0; i < 4; i++) {
      x[1] = i & 1;
      x[2] = i >> 1;
      int output = perceptron.predictedOutput(x);
      cout << "  " <<  x[2] << gates << x[1] << " = " << output << endl;   
    }

    perceptron.printWeights();
    cout << endl << "Hello, AI World!" << endl;
    
    return 0;
}

	// Use one thread to update one weight in parallel.
	
//Activation function (Step function)
 __device__ __host__ int activate(double value)
{
        return value >= 0 ? 1 : 0;
}

// Predicted output 
__device__ __host__ int predict(double x[], double w[], int n1)
{
        double sum = 0; 
        for (int i = 0; i < n1; ++i)
            sum += x[i] * w[i];

         int y1 = activate(sum);

         return y1;
}

__global__ void trainPerceptron(double *x_data, double *y, double *w, double eta, int n,  
		   int numSamples, int epochs)
{
   double *x;
   int i = threadIdx.x;

   for (int m = 0; m < epochs; m++){ 
     for (int k = 0; k < numSamples; k++) {
        x = x_data + k * n; 
        int y1 = predict(w, x, n);
        int error = y[k] - y1;
        // Update weights and bias 
        double dwi = eta * error * x[i];
        w[i] = w[i] + dwi;
        __syncthreads();
     }
   }
}

// Simple Perceptron class
class Perceptron
{
private:
    double *w;  //weights
    double *x;  //inputs
    int n1;     //n1=n+1, x[0], .... x[n], x[0] is 1
    int wSize;  //size of all weights in bytes
    int numSamples; //number of different input sets (x_data)
    double *d_w;    //device memory to store weights
    double *d_y;    //device memory to sotre target outputs
    double *d_x_data;   //device memory to store samples data

public:
    Perceptron(int nx, int num, double weights[], double *y,  double inputs[])
    {
        n1 = nx;
        numSamples = num;
        wSize = n1 * sizeof(double);
        int samplesSize = numSamples * wSize;
        int outputsSize = numSamples * sizeof(double);
        w =  new double[n1];
        for(int i = 0; i < n1; i++)
          w[i] = weights[i];
        cudaMalloc(&d_w, wSize);
        cudaMalloc(&d_y, outputsSize);
       cudaMalloc(&d_y, outputsSize);
        cudaMalloc(&d_x_data, samplesSize);
        cudaMemcpy(d_w, w, wSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, outputsSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_data, inputs, samplesSize, cudaMemcpyHostToDevice);
    }

    //Activation function (Step function)
    int activation(double value)
    {
        return activate(value);
    }

    // Predicted output 
    int predictedOutput(double xi[])
    {
       int y1 = predict(xi, w, n1);

        return y1;

    }

    // Train the perceptron
    // eta = learning rate
    // epochs = number of times of training 
    // numSamples = number of different input sets (x_data)
    // y is target output
    void train(double eta, int epochs)
    {
       trainPerceptron<<<1, n1>>>(d_x_data, d_y, d_w, eta, n1, numSamples, epochs);
       cudaDeviceSynchronize();
       cudaMemcpy(w, d_w, wSize, cudaMemcpyDeviceToHost);
    }

    void printWeights()
    {
        cout << "\nPerceptron weights: ";
        for(int i = 0; i < n1; i++)
          cout << "\n  w" << i <<": " << w[i];
    }

    ~Perceptron()
    {
        delete w;
        cudaFree(d_w);
        cudaFree(d_y);
        cudaFree(d_x_data);
    }
};

int main()
{
  ....
   //Construct a perception with a bias and 2 inputs 
   // 4 data sets
   Perceptron perceptron(3, 4, w, y, inputs);

   // Train the perceptron with eta = 0.1, and 10 epochs
   perceptron.train(0.1, 10);
   ....
}
