# What

Write a model that takes the CSV input from the roasting data, and determines the shut-off time so that the roast
is consistent. 

Shut-off time is a continuous value, so this is a regression problem.

Inputs are
1. Roasting temperature (really we want the roasting curve)
2. First crack time (seconds, since roast start)
3. Shut-off time (seconds, since roast start
4. Result / Grade (1-5, so need to one-hot encode this)

I would like to ask the model to provide me a shut-off time, given that I tell it the other inputs. 

E.g. I'm roasting at 235c, first crack hits at 14:35, and I want to have a dark roast - when do I shut off? 

# Problems / Questions

How do we represent the roasting curve in the model?  It isn't a single value.  Does it become a series of features? 



