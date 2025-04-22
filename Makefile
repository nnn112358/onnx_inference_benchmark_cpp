# Makefile for ONNX inference benchmark

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra

# ONNX Runtime paths
# NOTE: Update these paths to match your ONNX Runtime installation
ONNXRUNTIME_ROOT = /opt/onnxruntime-linux-x64-1.21.0/
ONNXRUNTIME_LIB = /opt/onnxruntime-linux-x64-1.21.0/lib/

# Includes and libraries
INCLUDES = -I$(ONNXRUNTIME_ROOT)/include
LIBS = -L$(ONNXRUNTIME_LIB) -lonnxruntime

# Source and object files
SRC = onnx_inference_benchmark.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = onnx_inference_benchmark

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) -Wl,-rpath,$$ORIGIN

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(OBJ) $(TARGET)

# Run
run: $(TARGET)
	./$(TARGET) model.onnx 100

# Help
help:
	@echo "Makefile for ONNX inference benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the benchmark program (default)"
	@echo "  clean     - Remove object files and executable"
	@echo "  run       - Run the benchmark with default parameters"
	@echo "  help      - Display this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make                                  - Build the program"
	@echo "  make run                              - Run with default model.onnx"
	@echo "  ./$(TARGET) <model_path> [iterations] - Run with custom parameters"
	@echo ""
	@echo "NOTE: Update ONNXRUNTIME_ROOT and ONNXRUNTIME_LIB in the Makefile"
	@echo "      to match your ONNX Runtime installation paths."

.PHONY: all clean run help
