LiquidSpikeFormer: Hybrid SNN-LNN-Transformer Architecture for Event-Based Vision

A biologically inspired, hybrid deep learning architecture combining Spiking Neural Networks, Liquid Time-Constant Neurons, and Transformer-based temporal attention for robust and energy-efficient event-driven gesture recognition.

📅 Motivation

Traditional deep learning models operate on synchronous, dense image data. However, biological brains process information through sparse, asynchronous spike signals. Event cameras like Dynamic Vision Sensors (DVS) mimic this biological process by emitting events only when pixel intensity changes occur. These sensors offer high temporal resolution and low energy cost, but pose unique challenges:

Sparse and irregular input data

Need for real-time processing

Lack of standard architectures

LiquidSpikeFormer bridges this gap using:

Spiking dynamics for biological plausibility

Liquid neurons for dynamic time constants

Transformer layers for global sequence modeling

This makes it suitable for neuromorphic hardware and applications such as robotics, AR/VR, prosthetics, and smart surveillance.

🔍 Research Objectives

Model biologically plausible, energy-efficient gesture recognition using event data.

Explore hybrid learning dynamics: SNNs for sparse input, LNNs for time-dependence, and Transformers for global context.

Evaluate generalization and temporal robustness across time-variant DVS input streams.

Move toward publishable results at top-tier AI venues (NeurIPS, ICLR) by refining architecture, interpretability, and training stability.

🔧 Architecture Breakdown

1. Dual Spike Encoders (Fine + Coarse)

Encodes DVS input into discrete time-binned spike tensors

Captures high-frequency and low-frequency event patterns

2. Conv-SNN Spatial-Temporal Blocks

2D convolution per time bin + temporal 1D convolution

Mimics early visual cortex (V1-like feature detection)

3. Liquid Time-Constant Blocks

Neurons with learned dynamic decay (tau)

Simulates membrane potential and adaptive thresholds

One block with neuromodulated gating

4. Spiking Transformer Layers

Sparse sequence of spikes processed via attention

Allows long-range temporal dependency modeling

5. Early-Exit Classifiers

Heads after key stages: early-mid, early-final, and final

Enable low-latency prediction (real-time response)

6. Hybrid Spiking Loss

CrossEntropy + spike sparsity + membrane stability + temporal consistency + threshold variance regularization

🔮 Model Diagram (Conceptual)

                DVS Events
                   │
        ┌---------------------┐
        │  Fine Spike Encoder  │
        │ Coarse Spike Encoder │
        └----------─--------┘
                   │
       Conv-SNN Spatial Temporal Blocks
                   │
          Conv Projection to Embed Dim
                   │
     ┌----------------------------┐
     │ PatchEmbeds + Frame Embeds │
     └-------------─----------┘
                   │
        Merge Projection (4 Streams)
                   │
      Liquid Time Constant Block 1
                   │
      Liquid Time Constant Block 2
                   │
        Transformer Layers (x2)
                   │
     ┌-----─----─-----─-----┐
     │ Early Mid  Early Final  Final │
     └----------------------------┘

🔢 Dataset

DVS Gesture Dataset (from iniLabs)

Format: .aedat 3.1

11 classes (e.g., hand wave, clap, etc.)

Each file contains events: (x, y, timestamp, polarity)

Augmentations:

Spatial jitter

Temporal scaling & cropping

Polarity flipping

Additive Gaussian noise

Normalization:

Time normalization between [0,1]

Poisson spike binning (optional)

🔬 Results (In Progress)

Observations:

High training accuracy (>90%) but test generalization is limited

Overfitting observed despite hybrid loss and data augmentation

Temporal attention helps stabilize late-stage predictions

Spike sparsity improves with adaptive thresholds

Next Experiments:

Ablation studies: Remove ConvSNN, LNN, Transformer independently

Self-supervised pretraining: Use supervised contrastive loss

Generalization benchmarks: Apply to N-MNIST, SHD

Hardware compatibility: Simulate energy use via MACs and spike rate

📈 Metrics Tracked

Accuracy / CrossEntropy Loss

Spike Rate (per-layer)

Threshold Adaptation Curve

Membrane Potential Heatmaps

Temporal Prediction Consistency

Early Exit Usage Rate

🫡 Broader Impact

This work contributes to:

Real-time, low-power AI for edge computing

Biologically grounded neural architectures

Bridging the gap between connectomics and artificial networks

Potential applications in prosthetics, neuromorphic robotics, and surveillance

🤝 Call for Mentorship & Collaboration

I'm currently seeking research mentorship from faculty or labs working at the intersection of:

Spiking Neural Networks (SNNs)

Neuromorphic Engineering

Temporal Attention / Transformer Models

Brain-inspired Learning Rules

This project is under active development, and I hope to grow it into a publication-worthy contribution under the right guidance.

If you're interested in collaborating or offering feedback, please feel free to reach out!
