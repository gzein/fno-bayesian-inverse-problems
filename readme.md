Darcy's problem in 1D, with special f=exp(a(x)), on domain [0, 1]
In 1D this collapses into an ODE

Step 1: create a way of accessing the forwards map G, never to be accessed in the solution loop, only to generate the data
Step 2: generate data, first without any noise, and then later add random gaussian perturbations
Step 3: construct the posterior distribution using the formula given in Nickl's notes
Step 4: diagnostics and plots - plot the shape of the distribution given by the posterior. How do you even plot a family
        of functions???

to make my life easier, let's start with f(x) = exp(x)
This has a nice closed form solution, so I can find the forwards map in this case ie I know what
G(exp(x))(y) is for any evaluation point y
I supplement this with boundary conditions on {0, 1}
