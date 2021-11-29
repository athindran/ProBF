# ProBF
Implementation of Probabilistic Barrier Certificates submitted to NeuRIPS 2021 safeRL workshop.

This code compares the ProBF-GP framework with the prior art LCBF which uses a neural network. We build upon the code in [1].

![image](images/segwaycomp.JPG)


## LCBF-NN
### Dependencies
1.  cudatoolkit=10.1
2.  python=3.8.5
3.  tensorflow=2.3.0
4.  keras==2.2.4

### Instructions for running the experiment
For running the LCBF, we use the tensorflow backbone provided by the authors of [2]. The environment used for running the code is listed in "tf_env.yml".
1.  segway_nn.py - Train the LCBF-NN with 10 different seeds and test each time on 10 seeds for the segway framework.
2.  quadrotor_nn.py - Train the LCBF-NN with 10 different seeds and test each time on 10 seeds for the quadrotor framework. 

## ProBF-GP ( Our approach )
### Dependencies
1.  cvxpy=1.1.10
2.  gpytorch=1.3.1
3.  cudatoolkit=10.1
4.  python=3.9.2
5.  pytorch=1.7.1

### Instructions for running the experiment
For running the Pro-BF, we use gpytorch for GP training. The environment used for running the code is listed in "pytogpu.yml".
1.  segway_gp.py - Train the ProBF-GP with 10 different seeds and test each time on 10 seeds for the segway framework.
2.  quadrotor_gp.py - Train the ProBF-GP with 10 different seeds and test each time on 10 seeds for the quadrotor framework. 

# Authors
[Athindran Ramesh Kumar](https://sites.google.com/site/athindranrameshkumar)
[Sulin Liu](https://liusulin.github.io/)
[Jaime F. Fisac](https://ece.princeton.edu/people/jaime-fernandez-fisac)
[Peter J. Ramadge](https://ece.princeton.edu/people/peter-j-ramadge)
[Ryan P. Adams](https://www.cs.princeton.edu/~rpa/)
Please reach out for any questions!

# References
[1] Python simulation and hardware library for learning and control. https://github.com/learning-and-control/core.git. Accessed: 2021-11-17.

[2] A. Taylor, A. Singletary, Y. Yue, and A. Ames. Learning for safety-critical control with control
barrier functions. In Learning for Dynamics and Control, pages 708–717. PMLR, 2020.
