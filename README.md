# Inverse Magnetic Billiards

## This Repository is part of a BsC Mathematics Thesis titled "Numerical Inverse Magnetic Billiards".

It contains implementation of Inverse Magnetic Billiards (invented by Sean Gasiorek).

We present a three-step algorithm for learning periodic orbits in arbitrary closed, bounded domains. In a first step, we create a training-dataset consisting of a pair of coordinates and the corresponding value of the generating function. In the second step, we use a neural network to learn an easily differentiable approximation of the generating function. In the third step, we use gradient descent to find perdiodic orbits via the stationary action principle from Hamiltonian dynamics.
