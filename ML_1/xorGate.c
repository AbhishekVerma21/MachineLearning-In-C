/*
One Hidden layer Model
Not for production 
    ---  ---
    ('') ('')
       []

     <---->

*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

/*
    XOR
    x^y = (x|y) & ~(x&y)
*/

typedef float sample[3];

//OR-Gate
sample xor_train[] = {
        {0,0,0},
        {0,1,1},
        {1,0,1},
        {1,1,0},
};

sample *train = xor_train;

#define train_count 4
#define train_frequency 1000000

float random_number(void) {
    return (float) rand() / (float) RAND_MAX;
}

typedef struct {
    //first layer parameters
    //Def: input scale factor to any node is called the parameter of that node 
    float or_w1;
    float or_w2;
    float or_b;
    float nand_w1;
    float nand_w2;
    float nand_b;
    float and_w1;
    float and_w2;
    float and_b;
} Xor;

//to make output bounded
float sigmoid(float y){
    return 1.f / (1.f + expf(-y));
}


//forward propagation
float forward (Xor m, float x1, float x2) {
    //first layer
    // first node output
    float y1 = sigmoid(m.or_w1*x1 + m.or_w2*x2 + m.or_b);
    // second node output
    float y2 = sigmoid(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b);
    //Output layer or second layer
    // final node output
    float y = sigmoid(m.and_w1*y1 + m.and_w2*y2 + m.and_b);
    return y;
}

/*
  Cost function does not know anything about the model.
  It just know the model input and output.
  So, it will just put the data inside the forwrd function and compare the result with output.
  Then it will return the error or cost of the model.
  Using this cost we will update the model parameters (I defined the parameter already)
*/
float cost(Xor m) {
    float result = 0.0f;
    for(size_t i=0; i<train_count; i++){
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2);
        float error = y - train[i][2];
        result += error * error;
    }
    result /= train_count; //just normalize, not mandatory
    return result;
}

/*
I found that if I need to reduce the iteration in cases like these,
I need to increase the also increase the rate
*/
Xor parameter_update(Xor m) {
    float eps = 1e-3; // for derivative (f(x + h) - f(x))/h
    float rate = 1e-1; // use so that my change in the direction of minimum cost will not overshoot
    printf("Start Cost is: %f\n", cost(m));
    for (size_t i = 0; i < train_frequency; ++i) {
        Xor g = m; //updated model
        float saved_val;
        float c = cost(m);
        // printf("Cost after iteration %zu. is : %f\n", i+1, c);
        saved_val = m.or_w1;
        m.or_w1 += eps;
        g.or_w1 = saved_val - rate * ((cost(m) - c) / eps); //updating parameter
        m.or_w1 = saved_val;

        saved_val = m.or_w2;
        m.or_w2 += eps;
        g.or_w2 = saved_val - rate * ((cost(m) - c) / eps); //updating parameter
        m.or_w2 = saved_val;

        saved_val = m.or_b;
        m.or_b += eps;
        g.or_b = saved_val - rate * ((cost(m) - c) / eps); //updating parameter
        m.or_b = saved_val;

        saved_val = m.nand_w1;
        m.nand_w1 += eps;
        g.nand_w1 = saved_val - rate * ((cost(m) - c) / eps); //updating parameter
        m.nand_w1 = saved_val;

        saved_val = m.nand_w2;
        m.nand_w2 += eps;
        g.nand_w2 = saved_val - rate * ((cost(m) - c) / eps); //updating parameter
        m.nand_w2 = saved_val;

        saved_val = m.nand_b;
        m.nand_b += eps;
        g.nand_b = saved_val - rate * ((cost(m) - c) / eps); //updating parameter
        m.nand_b = saved_val;

        saved_val = m.and_w1;
        m.and_w1 += eps;
        g.and_w1 = saved_val - rate * ((cost(m) - c) / eps); //updating parameter
        m.and_w1 = saved_val;

        saved_val = m.and_w2;
        m.and_w2 += eps;
        g.and_w2 = saved_val - rate * ((cost(m) - c) / eps); //updating parameter
        m.and_w2 = saved_val;

        saved_val = m.and_b;
        m.and_b += eps;
        g.and_b = saved_val - rate * ((cost(m) - c) / eps); //updating parameter
        m.and_b = saved_val;

        m = g;
    }
    printf("Final Cost is: %f\n", cost(m));
    printf("Number of training iterations are : %d\n", train_frequency);
    return m;
}

Xor random_xor(void) {
    Xor m;
    srand(time(0));
    m.or_w1 = random_number();
    m.or_w2 = random_number();
    m.or_b = random_number();
    m.nand_w1 = random_number();
    m.nand_w2 = random_number();
    m.nand_b = random_number();
    m.and_w1 = random_number();
    m.and_w2 = random_number();
    m.and_b = random_number();
    return m;
}


int main(void){
    srand(time(0));
    Xor m = random_xor();
    Xor g = parameter_update(m);


    // printf("\n---------OR Neuron Output------\n"); //naming a neuron as or nor nand does not mean it will behave like that, these are just names here.
    // for(size_t i=0;i<2;i++){
    //     for(size_t j=0;j<2;j++){
    //         printf("%zu OR %zu  =  %f \n", i, j, sigmoid(g.or_w1*train[i][0] + g.or_w2*train[i][1] + g.or_b));
    //     }
    // }

    // printf("\n--------NAND Neuron Output-----\n"); //naming a neuron as or nor nand does not mean it will behave like that, these are just names here.
    // for(size_t i=0;i<2;i++){
    //     for(size_t j=0;j<2;j++){
    //         printf("%zu OR %zu  =  %f \n", i, j, sigmoid(g.nand_w1*train[i][0] + g.nand_w2*train[i][1] + g.or_b));
    //     }
    // }


    //output
    printf("\n\n---------FINAL OUTPUT----------------\n");
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
            printf("%zu OR %zu  =  %f \n", i, j, forward(g, i, j));
        }
    }
    /*
        Note : If you do not used bias in this problem, you will never be able to get correct output when both input is 0
        Bias is basically use to shift your model so that when no weight is present the model start from correct pos

        Final cost always depends on the initialization of your weight.
        In this case it is random so sometimes it might comes a bit high.
        
        Your model here is 'g'. where g stores weight and biases. I call it a binary model.
    */   
    return 0;
}