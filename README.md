# **EC2 LLM Instance Calculator**

## **Overview**

This project is a single-page web application designed to help developers and machine learning engineers estimate the memory requirements for running Large Language Models (LLMs) and find the most cost-optimized Amazon EC2 instances for their workloads.

The calculator takes key model and workload characteristics as input—such as model size, precision, and task (inference or training)—to provide tailored, price-aware instance recommendations.

## **How to Use the Calculator**

The calculator is designed to be straightforward and interactive.

1. **Configure Your Workload:**  
   * **Primary Task:** Choose whether you are running Inference or Training. The available options will change based on your selection.  
   * **Model Size:** Enter the size of your model in billions of parameters (e.g., 7 for a 7B model).  
   * **Precision / Quantization:** Select the data type your model uses. Lower precision (like INT8) significantly reduces memory requirements but may affect accuracy.  
   * **Training Type (if Training):** Specify whether you are doing a Full Fine-Tuning or a more memory-efficient PEFT (Parameter-Efficient Fine-Tuning) method like LoRA.  
   * **Inference Parameters (if Inference):**  
     * **Context Length:** The maximum number of tokens in a single input sequence.  
     * **Concurrent Requests:** The number of requests you want the model to handle simultaneously (batch size).  
2. **Calculate & Find Instances:** Click the button to process your inputs.  
3. **Review the Results:**  
   * **Estimated Memory Requirement:** This box shows the total VRAM needed and a breakdown of how it was calculated.  
   * **Instance Recommendations:** The tool provides up to 5 recommendations for both single-accelerator and multi-accelerator setups. The instances are sorted by their absolute lowest hourly cost, but a "Best Value" badge will highlight the instance with the best price-per-GB of accelerator memory.

## **The Cost of AI: Choosing the Right Hardware Accelerator**

The price and performance of running an LLM depend heavily on the underlying hardware. This tool recommends instances from three main categories, each with distinct trade-offs.

#### **NVIDIA GPUs (G-Series, P-Series)**

* **The Generalist:** GPUs are the most flexible and widely supported hardware for AI.  
* **Use when:**  
  * You need to get started quickly or are prototyping.  
  * Your model or framework is not yet optimized for custom silicon.  
  * You need the absolute lowest latency for a single request (high-end P-series instances).  
* **Trade-off:** While highly performant, they can be more expensive for large-scale, sustained workloads compared to specialized chips. The **P-series** (p4, p5, p6) are optimized for high-performance training, while the **G-series** (g5, g6) offer excellent price-performance for inference.

#### **AWS Inferentia**

* **The Inference Specialist:** These chips are purpose-built by AWS to deliver high throughput at the lowest cost-per-inference.  
* **Use when:**  
  * Your primary goal is to minimize the cost of running a model in production with high, consistent traffic.  
* **Trade-off:** Requires a one-time model compilation step using the **AWS Neuron SDK**. This adds an initial development step but pays off in significant cost savings at scale.

#### **AWS Trainium**

* **The Training Specialist:** This is the training-focused counterpart to Inferentia, designed for maximum cost-efficiency in large-scale training jobs.  
* **Use when:**  
  * Your goal is the lowest absolute cost-to-train for large, foundation models.  
  * You are conducting long, multi-day or multi-week training runs where instance costs are a primary concern.  
* **Trade-off:** Like Inferentia, it requires using the **AWS Neuron SDK**. It is highly optimized for distributed training and can be much cheaper than equivalent GPU clusters.

## **Understanding the Calculations **

The memory estimations are based on common industry heuristics.

* **KV Cache (Inference):** During inference, the model stores Key/Value pairs for each token in the context to avoid reprocessing. Its size depends on **Context Length** and **Batch Size (Concurrent Requests)**, and it can be a major memory consumer for long-context models. The tool estimates its size based on these inputs and the model's internal dimensions.  
* **Framework Overhead:** This is a buffer for memory used by the ML framework (PyTorch/TensorFlow), CUDA kernels, and GPU drivers. The calculator estimates this as the larger of **2GB** or **10%** of the model's weight memory to ensure a safe margin.  
* **Training Memory:**  
  * **Full Fine-Tuning:** The calculation assumes a memory-intensive Adam optimizer, which requires storing not only the model weights, but also gradients and multi-parameter optimizer states (often in full FP32 precision).  
  * **PEFT:** For methods like LoRA, the memory requirement is much lower, as only a small fraction of the model's weights are being trained. The calculation is primarily the base model's weights plus a small buffer for the trainable adapter layers.
