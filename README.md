# ProBF
Implementation of ProBF: Learning Probabilistic Safety Certificates with Barrier Functions presented at NeurIPS 2021 SafeRL workshop. (https://arxiv.org/abs/2112.12210)

This code compares the our ProBF-GP framework with the prior art learned Control Barrier Functions which uses a neural network (LCBF-NN) [2]. We build upon the code in [1]. We add a controller that uses the mean and variance of the predictions to solve the ProBF-convex program.

![image](images/segwaycomp.JPG)


# Authors
[Athindran Ramesh Kumar](https://sites.google.com/site/athindranrameshkumar)

[Sulin Liu](https://liusulin.github.io/)

[Jaime F. Fisac](https://ece.princeton.edu/people/jaime-fernandez-fisac)

[Ryan P. Adams](https://www.cs.princeton.edu/~rpa/)

[Peter J. Ramadge](https://ece.princeton.edu/people/peter-j-ramadge)

Please reach out for any questions!

# References
[1] Python simulation and hardware library for learning and control. https://github.com/learning-and-control/core.git. Accessed: 2021-11-17.

[2] A. Taylor, A. Singletary, Y. Yue, and A. Ames. Learning for safety-critical control with control
barrier functions. In Learning for Dynamics and Control, pages 708–717. PMLR, 2020.
