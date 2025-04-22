// onnx_inference_benchmark.cpp
// A program to measure ONNX model inference speed using ONNX Runtime

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <onnxruntime_cxx_api.h>

// Utility function to print model input information
void printNodeInfo(const Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Get input count
    size_t num_input_nodes = session.GetInputCount();
    std::cout << "Number of inputs: " << num_input_nodes << std::endl;
    
    // Get input node names and dimensions
    for (size_t i = 0; i < num_input_nodes; i++) {
        // Get input name
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(i, allocator);
        std::string input_name = input_name_ptr.get();
        std::cout << "Input " << i << " name: " << input_name << std::endl;
        
        // Get input dimensions
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        std::vector<int64_t> input_dims = tensor_info.GetShape();
        std::cout << "Input " << i << " dimensions: ";
        for (auto dim : input_dims) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        // Get input type
        ONNXTensorElementDataType input_type = tensor_info.GetElementType();
        std::cout << "Input " << i << " type: " << input_type << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model> [num_iterations=10]" << std::endl;
        return 1;
    }
    
    // Get command line arguments
    std::string model_path = argv[1];
    int num_iterations = (argc > 2) ? std::stoi(argv[2]) : 10;
    
    try {
        // Initialize environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeBenchmark");
        
        // Session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Create session
        std::cout << "Loading ONNX model: " << model_path << std::endl;
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // Print input node information
        printNodeInfo(session);
        
        // Get input name and shape
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
        std::string input_name = input_name_ptr.get();
        
        Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_dims = tensor_info.GetShape();
        
        // Calculate total input size
        size_t input_tensor_size = 1;
        for (auto dim : input_dims) {
            if (dim > 0) {
                input_tensor_size *= dim;
            }
        }
        
        // Generate random input data
        std::vector<float> input_tensor_values(input_tensor_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        std::cout << "Generating random input data of size " << input_tensor_size << std::endl;
        for (size_t i = 0; i < input_tensor_size; i++) {
            input_tensor_values[i] = dist(gen);
        }
        
        // Get output name
        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
        std::string output_name = output_name_ptr.get();
        
        // Define input tensor
        std::vector<const char*> input_node_names = {input_name.c_str()};
        std::vector<const char*> output_node_names = {output_name.c_str()};
        
        // Create input tensor object from data values
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_dims.data(), input_dims.size()
        );
        
        // Store timing results
        std::vector<double> inference_times;
        inference_times.reserve(num_iterations);
        
        // Warm-up run
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_node_names.data(), 
            &input_tensor, 
            1, 
            output_node_names.data(), 
            1
        );
        
        std::cout << "Running " << num_iterations << " iterations..." << std::endl;
        
        // Begin timing iterations
        for (int i = 0; i < num_iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Run inference
            output_tensors = session.Run(
                Ort::RunOptions{nullptr}, 
                input_node_names.data(), 
                &input_tensor, 
                1, 
                output_node_names.data(), 
                1
            );
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            inference_times.push_back(duration.count());
            
            if ((i + 1) % 10 == 0) {
                std::cout << "Completed " << (i + 1) << " iterations" << std::endl;
            }
        }
        
        // Calculate statistics
        double total_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0);
        double mean_time = total_time / inference_times.size();
        
        std::sort(inference_times.begin(), inference_times.end());
        double median_time = inference_times[inference_times.size() / 2];
        double min_time = inference_times.front();
        double max_time = inference_times.back();
        
        // Calculate standard deviation
        double sq_sum = std::inner_product(
            inference_times.begin(), inference_times.end(), inference_times.begin(), 0.0,
            std::plus<>(), [mean_time](double x, double y) { return (x - mean_time) * (y - mean_time); }
        );
        double std_dev = std::sqrt(sq_sum / inference_times.size());
        
        // Print results
        std::cout << "\n===== Inference Performance Results =====" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Iterations: " << num_iterations << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total time: " << total_time << " ms" << std::endl;
        std::cout << "Average time: " << mean_time << " ms" << std::endl;
        std::cout << "Median time: " << median_time << " ms" << std::endl;
        std::cout << "Min time: " << min_time << " ms" << std::endl;
        std::cout << "Max time: " << max_time << " ms" << std::endl;
        std::cout << "Standard deviation: " << std_dev << " ms" << std::endl;
        std::cout << "Throughput: " << (1000.0 / mean_time) << " inferences/second" << std::endl;
        
        // No need to manually free memory with AllocatedStringPtr
        
    } catch (const Ort::Exception& exception) {
        std::cerr << "ONNX Runtime Error: " << exception.what() << std::endl;
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        return 1;
    }
    
    return 0;
}