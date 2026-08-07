#include "include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "forward/linear.hpp"
#include "include/linear.hpp"
#include "include/p_head.hpp"
#include "include/cache_in.hpp"
#include "include/cache_out.hpp"
#include "forward/layer_norm.hpp"
#include "forward/embeddings.hpp"
#include "include/netattention.hpp"
#include "include/attention_params.hpp"
#include "include/single_embeddings.hpp"
#include "backpropagation/interface_back.hpp"

// Again my background is beginner here with little concept from C
// Transformer are complex neural network artitecture so I will focus on
// making the things right at first rather than micro level optimization
// The goal is to learn the underlying concept we can always bring the performace up later on.

// I handtyped this again why not
// I created a monolith here never mind.

// nvcc src/attention.cpp src/kernel/math.cu -o src/bin/attention
// ./src/bin/attention

// nvcc src/attention.cpp src/kernel/math.cu src/backpropagation/Kernel/interface_back.cu -o src/bin/attention


// flat memory are efficient but the project for a beginner is so complicated that it will be fine just to get the model running.
// NVIDA cuda core kernel functions
// again optimising this might be even more difficult anyway lets just make it work.

// Note- I was learning and now I realized this is very difficult to split headers.

// backprop https://arxiv.org/pdf/2307.08691
extern "C" void QKmatmul(float *Q, float *Kt, float *out, int M, int N, int batch_size, int num_heads);
extern "C" void ScalerDvisionElem(float *arr, int batch, int n_head, int seq_len, int head_dim);
extern "C" void UpperTriangularMasking(float *arr, float val, int batch_size, int n_head, int seq_len);
extern "C" void softmax(float *arr, float *out, int N, int seq_len, int n_head, int batch_size);
extern "C" void QKVMatmulFinal(float *QK, float *V, float *out, int seq_len, int d_head, int n_head, int batch_size);
extern "C" void ReformShapeWapper(float *arr, float *out, int batch_size, int seq_len, int d_model, int num_head, int head_dim);

extern "C" void multiHeadedAttention(int num_head, int head_dimension, float *ws, float *out, int batch_size, int d_model, int seq_len);
extern "C" void SwapNS(int num_head, int head_dimension, float *ws, float *out, int batch_size, int seq_len, bool reverse);
extern "C" void TransposeKey(int num_heads, int head_dim, float *arr, float *out, int M, int N, int K, bool reverse);

extern "C" void vectorKernel(float *A, float *B, float *C, int N);
extern "C" void softmax2D(float *arr, float *out, int batch_size, int seq_len, int vocab_size);
extern "C" void CrossEntropy(float *x, int *y, float *oneHotOut, float *lossOut, int batch_size, int seq_len, int vocab_size);




class Attention
{
public:
    int d_model;
    int vocab_size;
    int num_heads;
    int seq_len;
    int batch_size;
    int head_dim;

    std::unique_ptr<Utility> utils = std::make_unique<Utility>();
    std::unique_ptr<Embeddings> embeddings; // called in the constructor good.
    std::unique_ptr<Linear> key;
    std::unique_ptr<Linear> query;
    std::unique_ptr<Linear> value;
    std::unique_ptr<Linear> outputProj;
    std::unique_ptr<LayerNorm> layerNorm;

    float *DeviceKt;
    float *DeviceQ;
    float *DeviceQKT;
    // shape is like B, num head, head_dim, seq_len
    float *BTC_MULTI_HEAD_BUFFER_HOST = nullptr; // most of this null pointer is uncessary but ok does not hurt.

    float *B_NUMHEAD_T_T; // QK^T host

    float *B_NUMHEAD_SEQLEN_HEADDIM;

    bool debug = true;

    float *deviceQKTSqrtD;

    float *deviceSoftmaxOut;

    // QKV part allocation, Softmax(QK/sqrt(d_model)) V
    float *QKVOutDeviceOut;
    float *deviceValue;

    // Allocation for re-connecting the heads.
    float *BTCHost = nullptr;
    float *BTCdevice;
    // same buffer for temp
    float *tempDevice;
    float *resedualOutDevice;

    float *tempX = nullptr;

    // ----- Paramater required for standard attention backpropagation not the flash ------
    // at the learning stage of mine this is fine as well

    // UNTIL AND UNLESS YOU REALLY KNOW WHAT YOU ARE DOING
    // YOU WONT EVER KNOW WHAT WENT WRONG
    // BUGS IN KERNEL ARE THAT QUIET ON TOP OF THIS INSANE
    // MATHAMATICS

    float *BATCH_NEAD_TIME_HEADDIM_DEVICE;

    float *S = nullptr;
    float *P = nullptr;
    float *O = nullptr;

    Attention(
        int d_model,
        int vocab_size,
        int num_heads,
        int seq_len,
        int batch_size,
        bool debug)
    {

        this->d_model = d_model;
        this->vocab_size = vocab_size;
        this->num_heads = num_heads;
        this->seq_len = seq_len;
        this->batch_size = batch_size;
        this->debug = debug;

        // Its okay to free this in destructor because we are using this buffer throught the duration of
        // this object or iteration anyway.

        this->embeddings = std::make_unique<Embeddings>(
            this->d_model,
            this->vocab_size,
            this->seq_len,
            this->batch_size,
            debug);

        // Lets seed Q,K,V
        key = std::make_unique<Linear>(d_model, d_model, seq_len, batch_size, num_heads, debug);
        query = std::make_unique<Linear>(d_model, d_model, seq_len, batch_size, num_heads, debug);
        value = std::make_unique<Linear>(d_model, d_model, seq_len, batch_size, num_heads, debug);

        outputProj = std::make_unique<Linear>(d_model, d_model, seq_len, batch_size, num_heads, debug);

        layerNorm = std::make_unique<LayerNorm>(batch_size, seq_len, d_model);

        // Just to test and keep track of things
        // std::cout << "d_model: " << d_model << std::endl;
        // std::cout << "vocab_size: " << vocab_size << std::endl;
        // std::cout << "seq_len: " << seq_len << std::endl;
        // std::cout << "num_heads: " << num_heads << std::endl;
        // std::cout << "batch size: " << batch_size << std::endl;

        this->head_dim = d_model / num_heads;

        // --------- Buffer shape allocations -------------

        // Q K^t matmul device and host allocation
        cudaMalloc((void **)&DeviceKt, batch_size * num_heads * head_dim * seq_len * sizeof(float));
        cudaMalloc((void **)&DeviceQ, batch_size * num_heads * head_dim * seq_len * sizeof(float));
        cudaMalloc((void **)&DeviceQKT, batch_size * num_heads * seq_len * seq_len * sizeof(float));

        // Q K^T Host
        BTC_MULTI_HEAD_BUFFER_HOST = (float *)malloc(batch_size * num_heads * head_dim * seq_len * sizeof(float));

        cudaMalloc((void **)&deviceQKTSqrtD, batch_size * num_heads * seq_len * seq_len * sizeof(float));
        cudaMalloc((void **)&BATCH_NEAD_TIME_HEADDIM_DEVICE, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        B_NUMHEAD_T_T = (float *)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float)); // T,T end shape after QK^T

        // for softmax's
        // I do not deserve an internship so what, people who are doing this wont understand this so I am writing c++ to scare people off.
        cudaMalloc((void **)&deviceSoftmaxOut, batch_size * num_heads * seq_len * seq_len * sizeof(float));

        cudaMalloc((void **)&QKVOutDeviceOut, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        cudaMalloc((void **)&deviceValue, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        B_NUMHEAD_SEQLEN_HEADDIM = (float *)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));

        // Here we will start the memory allocation for (B,T,C) since after computation the shapes are brought back
        this->BTCHost = (float *)malloc(batch_size * seq_len * d_model * sizeof(float)); // we aready have the in buffer allocation so I wont worry about that for right now., this is for output. we bring back the original shape.
        cudaMalloc((void **)&BTCdevice, batch_size * seq_len * d_model * sizeof(float));

        // allocate for that temporary x
        tempX = new float[batch_size * seq_len * d_model * sizeof(float)];

        cudaMalloc((void **)&tempDevice, batch_size * seq_len * d_model * sizeof(float));
        cudaMalloc((void **)&resedualOutDevice, batch_size * seq_len * d_model * sizeof(float));

        // ---- S, P, O ---- are required for backpropagation allocate here according to reqired size.
        S = (float *)malloc(seq_len * batch_size * num_heads * head_dim * sizeof(float));
        P = (float *)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));
        O = (float *)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    };

    ~Attention()
    {

        cudaFree(DeviceKt);
        cudaFree(DeviceQ);
        cudaFree(DeviceQKT);

        cudaFree(BATCH_NEAD_TIME_HEADDIM_DEVICE);

        cudaFree(deviceQKTSqrtD);

        cudaFree(tempDevice);

        (BTC_MULTI_HEAD_BUFFER_HOST != nullptr ? free(BTC_MULTI_HEAD_BUFFER_HOST) : void());

        (BTCHost != nullptr ? free(BTCHost) : void());

        delete[] tempX;

        cudaFree(deviceSoftmaxOut);
        cudaFree(deviceValue);
        cudaFree(QKVOutDeviceOut); // we could free in the individual method but if we never call them then its never free.

        cudaFree(BTCdevice);
        cudaFree(resedualOutDevice);

        free(B_NUMHEAD_T_T);
        free(B_NUMHEAD_SEQLEN_HEADDIM);

        free(S);
        free(P);
        free(O);
    }

public:
    // This method divides by sqrt of d_model so that atlast forward lgoic in this program looks readable to a normal human being.
    void scalerDvisionAcrossMat(float *QKt, int d_model)
    {
        // if (debug == true)
        // {
        //     std::cout << "Before dvision by the scaler" << std::endl;
        //     utils->printFlarArray4D(QKt, batch_size, num_heads, seq_len, seq_len);
        // }

        // copy that value from the pointer to device. dense looking C++ code to scare people. Just Kidiing!
        cudaMemcpy(deviceQKTSqrtD, QKt, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyHostToDevice);

        // // (batch, n_head, seq_len, seq_len)
        ScalerDvisionElem(
            deviceQKTSqrtD,
            batch_size,
            num_heads,
            seq_len,
            head_dim);

        // copy back to QKT
        cudaMemcpy(QKt, deviceQKTSqrtD, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        // if (debug == true)
        // {

        //     std::cout << "Afer division by the scaler " << std::endl;
        //     utils->printFlarArray4D(QKt, batch_size, num_heads, seq_len, seq_len);
        // }
    }

    void masking(float *arr, float val)
    {

        // copy to that deviceQKTSqrtD does not matter now. cudaDeviceSynchronize() waits until the GPU finishes. so use this as a memeory buffer. Later once we get the working result we can fix things like that.
        cudaMemcpy(deviceQKTSqrtD, arr, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyHostToDevice);

        UpperTriangularMasking(deviceQKTSqrtD, val, batch_size, num_heads, seq_len);

        cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess)
        // {
        //     printf("Kernel error: %s\n", cudaGetErrorString(err));
        // }

        // if (debug == true)
        // {
        //     utils->print2DMatrixLastTwo(hostQKT, batch_size, num_heads, seq_len, "QKT Before Mask");
        // }
        cudaMemcpy(arr, deviceQKTSqrtD, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost); // we are copying to the same pointer does not matter if I am not wrong till this point.

        // if (debug == true)
        // {
        //     // std::cout << "QKT unmasked" << std::endl;
        //     // utils->printFlarArray4D(arr, batch_size, num_heads, seq_len, seq_len);

        //     utils->print2DMatrixLastTwo(arr, batch_size, num_heads, seq_len, "");
        // }
    }

    void QKVMatmul(float *QK, float *V, float *out)
    {
        // this is the pattern of learning we could have organized this in the wrapper but its fine here. if we did that then more number of paramaters in the wrapper.
        // deviceQKTSqrtD this as a BUFFER was well, forgive me I am just learning I will tewak and fix this weird naming.
        cudaMemcpy(deviceQKTSqrtD, QK, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyHostToDevice); // QK to GDDR RAM
        cudaMemcpy(deviceValue, V, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);

        QKVMatmulFinal(deviceQKTSqrtD, deviceValue, BATCH_NEAD_TIME_HEADDIM_DEVICE, seq_len, head_dim, num_heads, batch_size);

        cudaMemcpy(out, BATCH_NEAD_TIME_HEADDIM_DEVICE, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // This method swaps the first and second dimension
    // I know hardcoded but fine for now
    void SwapNT(float *arr)
    {
        /*
            Before: [batch_size, n_head, T, d_head]
            After: [batch_size, T, n_head, d_head]
        */
        // if (debug)
        // {
        //     this->utils->print2DTensorOnDemmand(arr, batch_size, num_heads, seq_len, head_dim, 1, 0);
        // }

        // if (debug)
        // {

        //     std::cout << "Before [batch_size, n_head, T, head_dim]" << std::endl;
        //     utils->printFlarArray4D(arr, batch_size, num_heads, seq_len, head_dim);
        // }

        // deviceQKTSqrtD is the buffer with the size:- batch_size * num_heads * seq_len * head_dim;
        // QKVOutDeviceOut will be out with the same shape because we are just swapping the dimension memory allocation is the same.
        cudaMemcpy(BATCH_NEAD_TIME_HEADDIM_DEVICE, arr, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);

        SwapNS(num_heads, head_dim, BATCH_NEAD_TIME_HEADDIM_DEVICE, QKVOutDeviceOut, batch_size, seq_len, true); // I dont expect anyone to be seriously looking at this or reading so I really do not care about the comments right now.
                                                                                                                 // And when I get the working result I am stacking this in one file to make it look scary.
        cudaMemcpy(arr, QKVOutDeviceOut, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

        // if (debug)
        // {

        //     std::cout << "After [batch_size, T, n_head, head_dim] " << std::endl;
        //     utils->printFlarArray4D(arr, batch_size, seq_len, num_heads, head_dim);
        // }

        // if (debug)
        // {
        //     this->utils->print2DTensorOnDemmand(arr, batch_size, seq_len, num_heads, head_dim, 1, 2);
        //     this->utils->print2DTensorOnDemmand(arr, batch_size, num_heads, seq_len, head_dim, 1, 1);
        // }
    }

    // BEFORE (B, T, n_head, head_dim) AFTER: (B, T, C)
    void ReformShape(float *arr)
    {
        /*
            Keep in mind keeping the kerenl logic reuseable does not save you round trip in PCIe BUS.
            Before:- Shape [batch_size, T, n_head, d_head]
            After:- Shape (B, T, C)
        */

        // so basically here we can use the deviceQKTSqrtD buffer because it has the same size.
        cudaMemcpy(BATCH_NEAD_TIME_HEADDIM_DEVICE, arr, batch_size * seq_len * num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice);

        ReformShapeWapper(
            BATCH_NEAD_TIME_HEADDIM_DEVICE,
            BTCdevice,
            batch_size,
            seq_len,
            d_model,
            num_heads,
            head_dim);
        cudaMemcpy(BTCHost, BTCdevice, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

        // if (debug)
        // {
        //     std::cout << "After reform from the method" << std::endl;
        //     utils->printFlatArray3D(BTCHost, batch_size, seq_len, d_model, true);
        // }
    }

    float *forward(Batch currentBatch)
    {

        // if (d_model % num_heads != 0)
        // {
        //     throw std::runtime_error("The number of heads must be perfectly divisible by dimesnion");
        // }

        float *x = embeddings->forward(currentBatch); // this brings us with the (B, T, C) batch because we added encoding and embeddings, encoding for our case fixed

        layerNorm->forward(x); // pass by ref so should modify it

        // -------- There is this number  8 what appeans after layer norm -----
        // ofcourse the layer norm is not learned yet.

        // if (debug)
        // {
        //     std::cout << "Original x" << std::endl;
        //     this->utils->printFlatArray3D(x, batch_size, seq_len, d_model);
        // }

        // we need to store something here in order to add the resudal, a temporary variable
        std::memcpy(tempX, x, batch_size * seq_len * d_model * sizeof(float));

        // if (debug)
        // {
        //     std::cout << "Aftter Norm" << std::endl;
        //     utils->printFlatArray3D(x, batch_size, seq_len, d_model);
        // }

        float *Q = query->forward(x); // We are doing a linear transformation here when we pass in the forward method.
        float *K = key->forward(x);
        float *V = value->forward(x);

        Q = query->reshapeHead();
        K = key->reshapeHead();
        V = value->reshapeHead(); // (B, n_head, T, head_dim) so no tranpose needed or adjustment needed

        // for a valid matrix mul (..., T, head_dim) @ (..., head_dim, T) = (..., T, T) so we need to swap the last two dims

        // if (debug)
        // {
        //     std::cout << " Key matrix before transpose" << std::endl;
        //     utils->print2DMatrixLastTwo(Q, batch_size, num_heads, seq_len, head_dim);
        // }

        float *s = key->teansposeKeyForAttnScore(); // Shape (batch_size, n_head, head_dim, seq_len)

        // if (debug)
        // {
        //     std::cout << "After transpose" << std::endl;
        //     utils->print2DMatrixLastTwo(s, batch_size, num_heads, head_dim, seq_len);
        // }

        cudaMemcpy(DeviceKt, s, seq_len * batch_size * num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(DeviceQ, Q, seq_len * batch_size * num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice);

        QKmatmul(DeviceQ, DeviceKt, DeviceQKT, seq_len, head_dim, batch_size, num_heads); // S(1, 1, T, T)

        cudaMemcpy(B_NUMHEAD_T_T, DeviceQKT, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        // -------- Stage QK^T ----------------
        std::memcpy(S, B_NUMHEAD_T_T, batch_size * num_heads * seq_len * seq_len * sizeof(float));

        // BTC_MULTI_HEAD_BUFFER_HOST AFTER QKT (B, H, T, T)

        // DEBUGGER FOR QK^T
        // if (debug)
        // {
        //     std::cout << "After transpose" << std::endl;
        //     utils->print2DMatrixLastTwo(s, batch_size, num_heads, head_dim, seq_len);

        //     std::cout << " Query " << std::endl;
        //     utils->print2DMatrixLastTwo(Q, batch_size, num_heads, seq_len, head_dim);

        //     std::cout << "Matrix multiplication of QK^T" << std::endl;
        //     utils->print2DMatrixLastTwo(B_NUMHEAD_T_T, batch_size, num_heads, seq_len, seq_len);
        // }

        // if (debug)
        // {
        //     std::cout << "Before scaling " << std::endl;
        //     utils->print2DMatrixLastTwo(B_NUMHEAD_T_T, batch_size, num_heads, seq_len, seq_len);
        // }

        scalerDvisionAcrossMat(B_NUMHEAD_T_T, head_dim); // sqrt(head_dim)

        // if (debug)
        // {

        //     std::cout << "After scalled by sqrt(head_dim) " << sqrtf(head_dim) << std::endl;
        //     utils->print2DMatrixLastTwo(B_NUMHEAD_T_T, batch_size, num_heads, seq_len, seq_len);
        // }

        // -1e9f
        masking(B_NUMHEAD_T_T, -INFINITY);

        // if (debug == true)
        // {
        //     std::cout << "Before softmax " << std::endl;
        //     utils->print2DMatrixLastTwo(B_NUMHEAD_T_T, batch_size, num_heads, seq_len, seq_len);
        // }

        // --- Important Note ------
        // see here
        // apply softmax part!
        cudaMemcpy(deviceQKTSqrtD, B_NUMHEAD_T_T, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyHostToDevice);
        softmax(deviceQKTSqrtD, // Shape(batch_size, n_head, T, T)
                deviceSoftmaxOut,
                batch_size * num_heads * seq_len * seq_len,
                seq_len, num_heads,
                batch_size);

        // copy to host
        cudaMemcpy(B_NUMHEAD_T_T, deviceSoftmaxOut, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        // ------------ Stage Softmax(P) ----------------
        std::memcpy(P, B_NUMHEAD_T_T, batch_size * num_heads * seq_len * seq_len * sizeof(float));

        // deviceQKTSqrtD Shape(batch_size, n_head, T, T)

        // if (debug == true)
        // {

        //     std::cout << "After softmax " << std::endl;
        //     utils->print2DMatrixLastTwo(B_NUMHEAD_T_T, batch_size, num_heads, seq_len, seq_len);
        // }

        // value Shape(batch_size, n_head, seq_len, d_head)
        // deviceQKTSqrtD Shape(batch_size, n_head, T, T)

        // Q K determines what to attend, and V determines where to attend.
        // we will write a basic matmul kernel nothing fancy later we can benchmarket and use tensor cores.

        QKVMatmul(B_NUMHEAD_T_T, V, B_NUMHEAD_SEQLEN_HEADDIM); //  (batch_size, n_head, T, d_head)

        // ------------------ Stage PV ------------------------
        std::memcpy(O, B_NUMHEAD_SEQLEN_HEADDIM, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        // if (debug)
        // {
        //     std::cout << " Value " << std::endl;
        //     utils->print2DMatrixLastTwo(V, batch_size, num_heads, seq_len, head_dim);

        //     std::cout << "After matmul with QKV" << std::endl;
        //     utils->print2DMatrixLastTwo(B_NUMHEAD_SEQLEN_HEADDIM, batch_size, num_heads, seq_len, head_dim);
        // }

        // Now we would want to bring back the shape to after the attention score.
        // Shape(batch, T, n_head, d_head) ..so we wap firt and second
        SwapNT(B_NUMHEAD_SEQLEN_HEADDIM);

        ReformShape(B_NUMHEAD_SEQLEN_HEADDIM); // [B, num_heads, T, head_dim]

        // Now we need to make this go thtrough a Linear Transformation without it the model will just learn to stack information together
        // without learning to mix information from multiple heads together.

        float *projectedBTC = outputProj->forward(B_NUMHEAD_SEQLEN_HEADDIM);

        // I will add the resudual here to give it a context on what's it is attending to
        // if (debug)
        // {
        //     std::cout << " Before adding resedual " << std::endl;
        //     this->utils->printFlatArray3D(BTCHost, batch_size, seq_len, d_model);
        // }

        addResidual(projectedBTC, tempX);

        // if (debug)
        // {
        //     std::cout << " Resedual that got added " << std::endl;
        //     this->utils->printFlatArray3D(tempX, batch_size, seq_len, d_model);

        //     std::cout << " After adding the resedual " << std::endl;
        //     this->utils->printFlatArray3D(BTCHost, batch_size, seq_len, d_model);
        // }

        debug = false;

        return this->BTCHost;
    };

    void addResidual(float *input, float *temp)
    {
        // lets rewrite this BTCHost

        // BTCdevice is the buffer for I wished I had followed proper naming convenstion here, not my fault because you woudln't know the nature of abstraction here.
        // tempDevice is buffer for temp

        // copy both into the buffer.
        cudaMemcpy(BTCdevice, input, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(tempDevice, temp, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);

        vectorKernel(BTCdevice, tempDevice, resedualOutDevice, batch_size * seq_len * d_model);

        // copy this to pointer input, and that same x will be modified
        cudaMemcpy(input, resedualOutDevice, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    }

    float *BorrowBTCDevice()
    {
        return this->BTCdevice;
    }

    AttentionParamaters attentionParamaters; // just the reference so stack allocation is fine

    AttentionParamaters getParamaters()
    {
        LinearParams Q_p{query->getWeight(), query->getBias()};
        LinearParams K_p{query->getWeight(), query->getBias()};
        LinearParams V_p{query->getWeight(), query->getBias()};

        LinearParams OutputProject_p{outputProj->getWeight(), outputProj->getWeight()};

        LinearParams LayerNorm_p{layerNorm->getGamma(), layerNorm->getBetta()};

        SingleEmbeddings emebdding_p{embeddings->getEmbeddingsParamaters()};

        return {
            Q_p, // weight and bias of QKV
            K_p,
            V_p,

            OutputProject_p,
            LayerNorm_p,
            emebdding_p,

            S, // actual value copied from QKV
            P,
            O,

            BTCdevice,
            BATCH_NEAD_TIME_HEADDIM_DEVICE};
    }
};


class AttentionInterface
{

    int d_model;
    int num_heads;
    int vocab_size;
    int batch_size;
    int seq_len;
    bool runDebugger = false;
    bool drop_last = true;

    const std::vector<int> &data;
    std::unique_ptr<Attention> attention;
    std::unique_ptr<DataLoader> dataLoader;
    std::unique_ptr<Utility> utils;

    // turn those result into proballity score
    std::unique_ptr<Linear> lm_head;

    bool debug = true;

    // DEVICE SFOTMAX BEFORE CROSS ENTROPY LOSS IN
    float *DeviceSoftmaxBLin;
    // DEVICE SOFTMAX BEFORE CORSS ENTROPY LOSS OUT
    float *DeviceSoftmaxBLout;

    // ALLOCATE MEMORY FOR Y
    float *yHotEncodeDeviceOut;
    float *outCrossEntropyDevice;

    float *outCrossEntropyHost;

    int *deviceY;

    // ------ For testing one hot encode kernel --------- //
    float *outHotEncodeOut = nullptr;

    std::unique_ptr<AutoGradEngine> autograd;

    // LOW LEVEL BY DESIGN IS LITTLE BIT MESSY COMES WITH A TRADE OFFS ANYWAY

    float *dl_dz_out_device;
    float *dl_dz_out_host;

    float *dl_dw_out_device;
    float *dl_dw_out_host;

    // GPU buffer for the autograd engine in the attention interface.
    NetAttentionParamaters modelParamaters;

    // ---------- Autograd engine declaration --------------------
    float *out_h;

    // even for SWE after years of building things on my own
    // now I realize how stupid I was, sure this is the dumbest possible thing
    // goal is not perfection here, I just want to make it work

    // ---------- Autograd engine weight lm head ---------------------
    float *w_device;
    float *w_out_d;

private:
    // ----------- TEMPORARY DEBUGGER SCRIPT ---------------------

    // B,T,C shape use if you want to see and inspeace device
    void DebugBTCFlatArray3D(
        float *d_arr,
        int B,
        int T,
        int C // vocab size
    )
    {
        float *h_arr = (float *)malloc(B * T * C * sizeof(float));

        cudaMemcpy(h_arr, d_arr, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

        utils->printLastOneOf3D(h_arr, B, T, C);

        free(h_arr);
    }

public:
    AttentionInterface(
        int d_model,
        int num_heads,
        int batch_size,
        int seq_len,
        bool drop_last,
        const std::vector<int> &data, bool debug) : data(data)
    {
        this->d_model = d_model;
        this->vocab_size = data.size();
        this->batch_size = batch_size;
        this->seq_len = seq_len;
        this->drop_last = drop_last;
        this->num_heads = num_heads;
        this->debug = debug;

        utils = std::make_unique<Utility>();

        attention = std::make_unique<Attention>(
            d_model,
            vocab_size,
            num_heads,
            seq_len,
            batch_size,
            debug);

        autograd = std::make_unique<AutoGradEngine>(
            d_model,
            vocab_size,
            num_heads,
            seq_len,
            batch_size,
            debug);

        // BUFFER CPU/GPU allocation for the auto grad engine
        dl_dz_out_host = (float *)malloc(batch_size * seq_len * vocab_size * sizeof(float));
        cudaMalloc((void **)&dl_dz_out_device, batch_size * seq_len * vocab_size * sizeof(float));

        dataLoader = std::make_unique<DataLoader>(batch_size, data, seq_len, drop_last);

        lm_head = std::make_unique<Linear>(d_model, vocab_size, seq_len, batch_size, num_heads, debug);

        cudaMalloc((void **)&DeviceSoftmaxBLin, batch_size * seq_len * vocab_size * sizeof(float)); // wont something like this reserve GDDR RAM for too long till the liftspan of the object? Yes. -Avash
        cudaMalloc((void **)&DeviceSoftmaxBLout, batch_size * seq_len * vocab_size * sizeof(float));

        cudaMalloc((void **)&outCrossEntropyDevice, batch_size * seq_len * sizeof(float)); // (N) across all of the BT we will have the loss.
        cudaMemset(outCrossEntropyDevice, 0, batch_size * seq_len * sizeof(float));        // make it zero for the things that we are not touching.

        // memory for device for y
        cudaMalloc((void **)&deviceY, batch_size * seq_len * sizeof(int));

        // YOU DO NEED THIS PART BECAUSE ANOTHER KERNEL FOR THE CORSS ENTROPY LOSS WILL BE USING THIS.
        cudaMalloc((void **)&yHotEncodeDeviceOut, vocab_size * seq_len * batch_size * sizeof(float)); // (B, T)
        cudaMemset(yHotEncodeDeviceOut, 0, batch_size * seq_len * vocab_size * sizeof(float));        // make default 0

        outCrossEntropyHost = (float *)malloc(batch_size * seq_len * vocab_size * sizeof(float));

        // -- For testing if the kernel launch for one hot works, we have wrapped two kernels for the cross entropy loss REMOVE FOR PERFORMACE
        outHotEncodeOut = (float *)malloc(batch_size * seq_len * vocab_size * sizeof(float));

        cudaMalloc((void **)&out_h, batch_size * seq_len * vocab_size * sizeof(float));

        cudaMalloc((void **)&dl_dw_out_device, batch_size * d_model * vocab_size * sizeof(float));
        dl_dw_out_host = (float *)malloc(batch_size * d_model * vocab_size * sizeof(float));

        // ------------ for w in lm head --------------

        cudaMalloc((void **)&w_device, d_model * vocab_size * sizeof(float));
        cudaMalloc((void **)&w_out_d, d_model * vocab_size * sizeof(float));
    }

    ~AttentionInterface()
    {
        cudaFree(DeviceSoftmaxBLin);
        cudaFree(DeviceSoftmaxBLout);

        cudaFree(yHotEncodeDeviceOut);
        cudaFree(outCrossEntropyDevice);

        cudaFree(deviceY);

        free(outCrossEntropyHost);

        (outHotEncodeOut != nullptr ? free(outHotEncodeOut) : void());

        cudaFree(dl_dz_out_device);
        free(dl_dz_out_host);

        cudaFree(out_h);

        cudaFree(dl_dw_out_device);
        free(dl_dw_out_host);

        cudaFree(w_out_d);
        cudaFree(w_device);
    }

    LinearParams getLmHeadParams()
    {
        return LinearParams{lm_head->getWeight(), lm_head->getBias()};
    }

    void softmaxAcrossProballityCrossEntropyLoss(float *probality, int *y)
    {
        cudaMemcpy(DeviceSoftmaxBLin, probality, batch_size * seq_len * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceY, y, batch_size * seq_len * sizeof(int), cudaMemcpyHostToDevice);

        // this accounts for whatever we are doing softmax on being greater than 32 i.e warp size.
        // THIS IS SOFTMAX ACROSS the logits (B, T, vocab_size)

        /*
            DeviceSoftmaxBLout: Proballity from logits (predicted)
            yHotEncodeDeviceOut: Actual Y (B, T) -> (B, T, vocab_size)
            outCrossEntropyDevice: CE out flat list
        */
        softmax2D(
            DeviceSoftmaxBLin,
            DeviceSoftmaxBLout,
            batch_size,
            seq_len,
            vocab_size);

        cudaMemcpy(probality, DeviceSoftmaxBLout, batch_size * seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

        CrossEntropy(
            DeviceSoftmaxBLout, // out from softmaxed
            deviceY,
            yHotEncodeDeviceOut,
            outCrossEntropyDevice,
            batch_size,
            seq_len,
            vocab_size);

        cudaMemcpy(outCrossEntropyHost, outCrossEntropyDevice, batch_size * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(outHotEncodeOut, yHotEncodeDeviceOut, batch_size * vocab_size * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        if (debug)
        {
            //  Here basically every element of the y must have its position encoded based on all of the vocab_size
            // std::cout << "Vocab size of all BT" << std::endl;
            // utils->printLastOneOf3D(probality, batch_size, seq_len, vocab_size);

            // std::cout << "Before one hot encode" << std::endl;
            // utils->printFlatArray2D(y, batch_size, seq_len);

            // // Shape
            // std::cout << " After one hot encode " << std::endl;
            // utils->printFlatArray3D(outHotEncodeOut, batch_size, seq_len, vocab_size);

            // std::cout << "Apply the cross entropy loss" << std::endl;
            // utils->printFlatArray1D(outCrossEntropyHost, seq_len * batch_size);

            // std::cout << "Predicted" << std::endl;
            // DebugBTCFlatArray3D(DeviceSoftmaxBLout, batch_size, seq_len, vocab_size);

            // std::cout << "Actual" << std::endl;
            // DebugBTCFlatArray3D(yHotEncodeDeviceOut, batch_size, seq_len, vocab_size);
        }

        debug = false;
    }

    void train(int epoch)
    {
        Batch batch;

        for (int i = 0; i < epoch; ++i)
        {
            while (!(batch = dataLoader->iter()).empty())
            {
                // std::cout << batch.width << std::endl;
                // utils->printFlatArray2D(batch.x, seq_len, batch.width);

                // Shape (B, T, C) only pointer dependent upon the channel dimension is here
                // And there is an illegal memory access somewhere here.
                float *x = attention->forward(batch);

                // if (debug)
                // {
                //     std::cout << " Before LM head " << std::endl;
                //     this->utils->printFlatArray3D(x, batch_size, seq_len, d_model);
                // }

                float *prob = lm_head->forward(x); // Shape (B, T, vocab_size) x is not changed here.

                modelParamaters.w_host = lm_head->getWeight();

                // if (debug)
                // {
                //     std::cout << " After LM head " << std::endl;
                //     this->utils->printLastOneOf3D(prob, batch_size, seq_len, vocab_size);
                // }

                softmaxAcrossProballityCrossEntropyLoss(prob, batch.y);

                // if (debug)
                // {
                //     std::cout << " After sfotmax last two dimension " << std::endl;
                //     this->utils->printLastOneOf3D(prob, batch_size, seq_len, vocab_size);
                // }

                // ----------- Lets gather overall paramaters here ---------- //
                modelParamaters.attention_head = attention->getParamaters();
                modelParamaters.lm_head = LinearParams{lm_head->getWeight(), lm_head->getBias()};
                modelParamaters.L = outCrossEntropyDevice;        // CE Out
                modelParamaters.y_actual = yHotEncodeDeviceOut;   // (B, T, vocab_size) from actual
                modelParamaters.y_predicted = DeviceSoftmaxBLout; // (B, T, vocab_size) to predicted proballity

                modelParamaters.dl_dz_out_device = dl_dz_out_device;
                modelParamaters.dl_dz_out_host = dl_dz_out_host;
                modelParamaters.h = x;                                   // this h is the output of lm head Shape(B, T, vocab_size)
                modelParamaters.device_h = attention->BorrowBTCDevice(); // (B, T, d_model) on device
                modelParamaters.device_out_h = out_h;

                // for dl_dw = delta h^T derived in flashback.md

                modelParamaters.dl_dw_device = dl_dw_out_device;
                modelParamaters.dl_dw_host = dl_dw_out_host;

                // these are just borrowed pointers
                modelParamaters.w_device = w_device;
                modelParamaters.wt_out_d = w_out_d;

                // because the backprops needs to be done for each epoch.
                // we need to keep in mind that the things hurting performace like cuda malloc and everything declared
                // inside of the constructor of autograd engine is costly.
                // there is tradeoff between making things modular and fusing everything together.
                // Lets create a buffer for CPU/GPU memory in this class so that we dont overload the system and free it when the object is destroyed.
                autograd->backprop(modelParamaters);
                debug = false;
            }
            dataLoader->resetIterator(); // just the weird logic that I wrote.
        }
    }
};

// God how am I going to romantaseize this?
// its no longer cool for me now.

int main()
{
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    bool debug = true;
    // #ifdef DEBUG
    //     debug = true;
    // #endif

    int d_model = 32;
    int vocab_size; // that depends upon the data that you are passing.
    int num_heads = 2;
    int batch_size = 8;
    int seq_len = 4;
    int epoch = 12;
    bool drop_last = true; // for training set this to false, if someone is serious about this email me. the cost of implementing this feature will affect everything in depth many tradeoffs

    bool hyperParamaterRelease = false;

    std::string path = "./src/data/chunk.txt";

    auto utils = std::make_unique<Utility>();

    auto textEncoderFile = std::make_unique<EncoderText>();

    textEncoderFile->loadTextChunk(path); // load that into char arr

    auto &charPool = textEncoderFile->getFileAsChar();
    auto helper = std::make_unique<Helper>(charPool);

    vocab_size = charPool.size();

    // In this case we would have release the config from our C++ script
    // and python program should be able to read it.

#ifdef PARAMS
    hyperParamaterRelease = true;
#endif

    if (hyperParamaterRelease)
    {
        // Python debugger wants hyperparams to initlize item to totally remove randomness
        // to redice the kernels, if compiled with this flag then the program ends here

        releaseHyperParamaters(
            d_model,
            vocab_size,
            batch_size,
            seq_len,
            num_heads); // write the hyperparamaters down

        return 0;
    }

    // we just need a simple shample.
    epoch = debug == true ? 1 : epoch;

    const std::vector<int> &encodedData = helper->getEncodedList();

    std::unique_ptr<AttentionInterface> model = std::make_unique<AttentionInterface>(
        d_model,
        num_heads,
        batch_size,
        seq_len,
        drop_last,
        encodedData,
        debug);

    model->train(epoch);

    auto end = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize(); // CPU is waiting for the GPU to finish
    std::chrono::duration<double, std::milli> duration = end - start;

    if (!(debug))
    {
        std::cout << "Total time C++ execution (with sync overhead): " << duration.count() << " ms\n";
    }

    return 0;
}
// I wont let this project die