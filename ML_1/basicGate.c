/*
Simple perceptron model 
                        Input (X) ---- W, b ----> [Summation | Activation] -----> Output (Y)
*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

/*
You can change train data to NOT, AND, NAND, NOR, it will work just fine.
It will not work for XOR : it is not modellable by single neuron.
A single neuron can model a single line type of classification. Make diagram on paper
x^y = (x|y) & ~(x&y)
*/

//OR-Gate (just change this input to AND / NOR/ NAND gate to get correct input)
float train_OR[][3] = {
        {0,0,0},
        {0,1,1},
        {1,0,1},
        {1,1,1},
};

float sigmoid(float y){
    return 1.f / (1.f + expf(-y));
}

#define train_count sizeof(train_OR) / sizeof(train_OR[0])

float random_number(void) {
    return (float) rand() / (float) RAND_MAX;
}

float cost(float w1, float w2, float b) {
    float result = 0.0f;
    for(size_t i=0;i<train_count;++i){
        float x1 = train_OR[i][0];
        float x2 = train_OR[i][1];
        float y = x1 * w1 + x2 * w2 + b; // linear combination of weight and input
        //error or distance of actual and expected
        y = sigmoid(y); 
        // printf("y = %f\n", y);
        float d = y - train_OR[i][2];
        result += d*d; //squared error
    }
    result /= train_count;
    return result;
}

int main(void){
    srand(time(0));
    float w1 = random_number()*10.0f;  //weight init 1
    float w2 = random_number()*10.0f;  //weight init 2
    float b = random_number()*5.0f;  // bias init
    float eps = 1e-2; // for derivative (f(x + h) - f(x))/h
    float rate = 1e-2; // use so that my change in the direction of minimum cost will not overshoot

    //we need cost to approach zero
    for(size_t i = 0; i<100000 ;++i) {
        float c = cost(w1 ,w2 ,b);
        float dw1 = (cost(w1+eps, w2, b) - cost(w1, w2, b)) / eps; // dervative of weight 1
        float dw2 = (cost(w1, w2+eps, b) - cost(w1, w2, b)) / eps; // dervative of weight 2
        float db = (cost(w1, w2, b+eps) - cost(w1, w2, b)) / eps; // derivative of bias
        w1 -= rate * dw1; //weight 1 update with rate
        w2 -= rate * dw2; //weight 2 update with rate
        b -= rate * db; //weight with bias
        printf("%zu.  Updated cost/error : %f, updated weight 1: %f, updated weight 2: %f, updated bias %f \n", (i+1), c, w1, w2, b);
    }
    printf("\n------------------------------\n");
    printf("Final weight 1: %f, Final weight 2: %f, final bias %f, final cost %f\n", w1, w2 ,b, cost(w1, w2, b));

    //output
    printf("\n---------OUTPUT----------------\n");
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
            printf("%zu OR %zu  =  %f \n", i, j, sigmoid(w1*i + w2*j + b));
        }
    }

    //Note : If you do not used bias in this problem, you will never be able to get correct output when both input is 0    
    return 0;
}