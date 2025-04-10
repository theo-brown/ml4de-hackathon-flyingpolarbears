# Dynamics-Informed Control Challenge

Welcome to the **xxx**, a two-part benchmark designed to test the *physical plausibility* and *reactivity* of learned dynamical models under controller perturbations.

In this challenge, a machine learning model should be trained to learn the dynamics of the system. Then, an exact controller is used to prod the system to a certain state. The test is whether these learned models behave as expected under control, mimicking real-world physical intuition.

---

## 🔍 Overview

The challenge comprises **two classic control problems**, both centered around **reactive dynamics** under feedback control:

1. **1D Damped Oscillator**
2. **Single Pendulum Swing-Up**

In both tasks, a controller interacts with the learned dynamical system with the aim of driving it toward a desired target behavior. The key is whether the learned model reacts realistically to control signals and whether it generalizes across a wide range of motion trajectories.

---

## ⚙️ Task Descriptions

### 1. Damped Oscillator

- **Objective**: A controller attempts to damp the oscillation of a 1D mass-spring-damper system and bring it to rest at the origin.
- **Test**: Does the learned dynamics model allow for energy to dissipate under control input, as a real damped oscillator would?

### 2. Pendulum Swing-Up

- **Objective**: A controller applies torques to swing up a single pendulum and balance it in the upright position.
- **Test**: Can the model accommodate realistic swing-up behavior, where energy is injected through periodic control and then balanced?

---

## 🎯 Goal of the Challenge

We evaluate whether the **learned dynamics** are:
- **Responsive to control** in a physically plausible way,
- **Robust across perturbations**, effectively sampling different "initial conditions" across a range of motion trajectories,
- Capable of producing **expected behaviors** under each scenario (e.g., damping, swing-up, stabilization).

This challenge encourages the development of models that go *beyond curve-fitting* and exhibit true **dynamical fidelity**.

---

## 📈 Evaluation Criteria

Participants are judged based on:
- **Qualitative behavior** of trajectories under control (e.g., does the pendulum actually swing up?)
- **Robustness across a range of perturbations** or starting conditions,
- **Quantitative metrics** such as energy dissipation (oscillator) or time-to-balance (pendulum).

---

## 🧠 Why This Matters

Machine learning models for physical systems often succeed in trajectory prediction but fail when integrated with control — this challenge probes whether your model can *serve as a proxy for reality*, not just a smoother.