/*
Simple perceptron model 
                        Input (X) ---- W, b ----> [Summation | Activation] -----> Output (Y)
*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

//linear data distribution
float train[][2] = {
        {1,2},
        {2,4},
        {3,6},
        {4,8},
        {5,10},
};

// float sigmoid(float y){
//     return 1.0f / (1.0f + exp(-y));
// }

#define train_count sizeof(train) / sizeof(train[0])

float random_number(void) {
    return (float) rand() / (float) RAND_MAX;
}

float cost(float w, float b) {
    float result = 0.0f;
    for(size_t i=0;i<train_count;++i){
        float x = train[i][0];
        float y = x * w + b; // linear combination of weight and input
        //error or distance of actual and expected
        // y = sigmoid(y); 
        // printf("y = %f\n", y);
        float d = y - train[i][1];
        result += d*d; //squared error
    }
    result /= train_count;
    return result;
}

int main(){
    srand(time(0));
    float w = random_number()*10.0f;  //weight init
    float b = random_number()*5.0f;  // bias init
    float eps = 1e-3; // for derivative (f(x + h) - f(x))/h
    float rate = 1e-3; // use so that my change in the direction of minimum cost will not overshoot

    //we need cost to approach zero
    for(size_t i = 0; i<5000 ;++i) {
        float dw = (cost(w+eps,b) - cost(w,b)) / eps; // dervative of weight
        float db = (cost(w, b+eps) - cost(w,b)) / eps; // derivative of bias
        w -= rate * dw; //weight with rate
        b -= rate * db; //weight with bias
        printf("%zu  Updated cost/error : %f , updated weight : %f , updated bias %f \n", (i+1), cost(w,b), w, b);
    }
    printf("\n------------------------------\n");
    printf("Final weight : %f , final bias %f\n", w ,b);
    
    return 0;
}