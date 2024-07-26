# KalmanFiltering


## Summary

The goal of this project was to develop a exhaustive set of filtering tool, principally based on Kalman Filtering theory. The algorithms developed in this project were inspired by the book "Optimal State Estimation" written by Dan Simons, of Cleveland State University. However, I did not put into code each and every algorithms developed in his book, only those I found useful for myself. Maybe I'll change that one day.

## Organization of the project

This project is divided into three separate folders.

### filters

The one that interests us the most, filters contains all the scripts where the different filters are defined. They are all separated in several scripts depending on the type of filter. 

To use a particular filter, please refer to its own documentation.

#### kalman_filters.py
This one contains the basic Kalman filter for linear systems. Assume you have a discrete system in the shape:

    x[n+1] = F[n] * x[n] + G[n] * u[n] + w[n] where w[n] ~ N(0, Q[n])
    y[n+1] = H[n+1] * x[n+1] + v[n+1] where v[n+1] ~ N(0, R[n+1])

You can get an estimate x_hat[n] of x[n] using a classical Kalman Filter. This filter is defined in the script as a class object named KalmanFilter, that works for systems with discrete dynamics and measurements. For systems with discrete measurements and continuous dynamics in the shape:

     x_dot(t) = A * x(t) + B * u(t) + w(t) where w(t) ~ N(0, Q)
    y[t_n] = H[n] * x(t_n) + v(t_n) where v(t) ~ N(0, R[n])

The first approach is to discretize this system using a sampling time dt. You can do that automatically using the class object DiscretizedKalmanFilter. 

Please note: there exist in the book a continuous Kalman Filter that will integrate the dynamics of the above system instead of discretizing them, but I haven't devoloped it yet (the class object ContinuousTimeKalmanFilter is still under developement and should not be used).

#### h_infinity_filter.py

In this script, I developed a model of H_infinity filter. You can use it for linear systems, as it will give enhanced precision than a Kalman Filter, although it will consume more computational power.

#### extended_kalman_filters.py

This script contains two new classes of filter, adapted for slightly non-linear systems for which we know not only the dynamics and measurement functions, but also their jacobians. 

First, the HybridExtendedKalmanFilter class allows you to create an Hybrid Extended Kalman Filter (Hybrid EKF). This filter works perfectly for systems with continuous dynamics and discrete measurements. It will integrate the dynamics between two measurements (assuming no noise) in order to compute the approximated state.

Secondly, DiscreteExtendedKalmanFilter allows you to create a classic EKF for systems with both discrete dynamics and measurements.

#### unscented_kalman_filter.py

This script defines only one filter: the UnscentedKalmanFilter, also known as UKF. This filter assumes you have a transition function between the current and next states of your system, that is perturbated by some gaussian noise of known variance and zero mean. It also assumes you have an observation function that is also perturbed by white noise.

Please note: the functions don't have to be linear. In fact, the UKF is very robust for systems with higher degrees of nonlinearity, even more than the EKF. Secondly, this filter does not need to know the derivatives of the functions, as it will sample a bunch of sigma points to get the estimated state through averaging the propagation of said points.

This filters requires more computational resources than the EKF, but is able to handle higher degrees of non linearity with more robustness.

#### particle_filters.py

This script defines two filters. The first one is the classical ParticlewFilter, that creates a set of particles (the number is determined by the user as a trade-off between accuracy and computational cost) that are propagated over periods using the state dynamics and a random distribution that is assumed to represent the uncertainty of the system. The particles' measurements are then computed thanks to the measurement function and another random distribution that is assumed to represent the uncertainty of the measurement. The estimate is then computed to be the weighted average of the particles.

Please note: the random distributions must not necesseraly be gaussian noises. In fact, this filter is the only filter that can handle any type of noise. However, the noises have to use the specific class ProbabilityDensityFunciton defined in stats. 


A second filter is defined in this script: the ExtendedParticleFilter class, that mixes the dynamics of both the EKF and Particle Filter for enhanced precision. 

### stats

This folder is used to defined stats tools to be used only in the particle filters.

The main script is probability_density.py, that defines a class object to represent a PDF as a convenient object. The two other scripts, univariate_pdf.py and multivariate_pdf.py, are used to define several classical PDFs, such as gaussian, uniform, and exponential.

### test

The test folder contains a test script for each filter I created (except the ones of kalman_filters.py, I was lazy). This served originally for me to test my filters, but feel free to use it as an exemple of how to take advantage of the filters I created. 

To run a test script, go into the directory of the whole project (i.e ~/your_path/KalmanFiltering) and run this command (replace test_script_name by the name of the script without the ending ".py"):

    python3 -m test_scripts.test_script_name

## What filter should I use for my problem?

Great question!

To put it simply, start by enquiring what type of uncertainty you are facing (i.e what type of noise perturbates your system's dynamics and measurement). If it is non-gaussian, a particle filter would be better as you can directly use that noise. However, if you can approximate it by a gaussian, it may be better to enquire if you can use a less-consuming algorithm.

If your system is linear with gaussian noise, the usual Kalman Filter is already optimized to handle it. Using a more advanced filter will produce a result of similar accuracy for a much greater cost.

If it is slightly non-linear, try the EKF.

If it is very non linear, try the UKF.

What you can also do is try several of them and see which one gives the best accuracy. 

## How do I use it?

Follow the workflow of the corresponding test script.

But basically:

### Start by initializing it 

For this you need to give all the required information about your system. Make sure your arguments have the corresponding shape.

### Update it

On each epoch, update the filter to approximate the new state of the system. 

For this use the update method, that is common to each filter:

    my_filter.update(*args)

Make sure to give it all the required arguments (i.e at least the real-world measurement y and eventually the control u (just put in a zero array of the corresponding size if there is no control vector in your system)).

### Store the estimates

On each update, make sure to store the estimate (in a list or an array). You can use the get_estimate() method that is common to each filter. 
