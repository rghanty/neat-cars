

# neat-cars
AI cars learning tracks using NEAT (NeuroEvolution of Augmenting Topologies). Each car has its own brain (aka Neural Network) and learns to stay within the tarmac as it tries to complete laps around the circuits in the "tracks" folder.

## Explanation

* Using the neat library by Python, a population of cars is dispatched to learn its environment (aka the track) and make progress trying to cover as much distance as possible around the track. If a car goes outside the track, it dies.
* As soon as all cars in a generation die, a new generation with the same population is created which learns from the previous generation and tries to go further. The cars which go further are given a reward whereas cars that die incur a negative reward.
* Eventually, a generation of cars will make it around a lap after learning the track.

## Specifics
* The program uses a config file (config-feedforward.txt) that is used by the neat library to configure neural networks (bias mutation, weight mutation, # of inputs nodes, etc.)
* For this application, the number of input nodes is **7**, corresponding to the number of radars used by each car to evaluate its surroundings, and the number of output nodes is **2**, determining whether it should steer **right**, **left** or keep **straight**.
* Each of this information can be customised in the config file.
* The maximum number of generations to run can be modified as the second argument to run() in **line 209** in main.py
* The track can be changed by changing the track name on **line 12** in main.py
* You can add your own tracks, but make sure to change the pixel location of the spawn point of the cars on **line 26** in main.py
## Preview
Here is Generation 5 which started with 80 cars learning the Interlagos track of which only 2 (eventually 1) survived to complete an entire lap.





https://github.com/rghanty/neat-cars/assets/99227180/5ba59433-a1f9-4a4a-a3e3-3bbc40842d2a





