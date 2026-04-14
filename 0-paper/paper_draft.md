# Visualizing an Algorithmic Artificial Chemistry for de novo functional exploration

## Abstract

- Artifical Chemistry aims to understand the self organization that leds to phenomena such as life
- AlChemy is an interesting AChem approach
- It is hard to understand the output and dynamics of ALChemy (and other AChem)
- We developed a python wrapper and visualization dashboard for high-throughput AlChemy experiments.
- We demonstrate the utility of these tools using a set of newly designed experiments drawn from recent advancements in prebiotic chemistry
- We simulations show that....
- We discuss the implications of this work for prebiotic chemistry, functional program design, and the future of machine learning. 


## Introduction

### Motivation for the Origin of Life

### Review of Recursive Experiments in Chemistry (Surman et al, Doran et al, Asche et al, Finkel-pinter et al)

### Review of AlChemy Simulations/Rules

### Review of AlChemy Papers
"Self-organization in computation and chemistry: Return to AlChemy" Mathis, C., Patel, D., Weimer, W., & Forrest, S. (2024, September 30)
The authors revisit and expand upon a computational model called AlChemy, originally proposed by Walter Fontana and Leo Buss in the 1990s. This model, based on λ calculus, was designed to explore how complex adaptive systems could emerge from simple constituent parts. The authors successfully reproduced the original AlChemy results and extended the analysis using modern computing resources4. This replication effort revealed several unexpected features of the system, demonstrating a mix of dynamical robustness and fragility. The study found that complex, stable organizations emerge more frequently than previously expected. These organizations showed robustness against collapse into trivial fixed points, indicating a level of stability in the emergent structures. Despite the stability of individual organizations, the research revealed that these stable organizations could not be easily combined into higher-order entities. This finding highlights the challenges in scaling up complexity within the AlChemy framework. The authors investigated the impact of random generators used in the model, characterizing the initial distribution of objects and their consequences on the results. This analysis provides insights into the sensitivity of the model to initial conditions. A significant contribution of the paper is a constructive proof showing how an extension of the model, based on typed λ calculus, could simulate transitions between arbitrary states in any possible chemical reaction network. This establishes a concrete connection between AlChemy and chemical reaction networks, bridging computational and chemical domains. The authors discuss potential applications of AlChemy to self-organization in modern programming languages and quantitative approaches to the origin of life. This suggests that the model could have broader implications beyond its original scope, potentially contributing to our understanding of complex systems in both computational and biological contexts. The paper represents a significant revisitation of the AlChemy model, providing new insights into self-organization in computation and chemistry. By reproducing and extending the original results, the authors have demonstrated the model's continued relevance and potential for understanding complex adaptive systems. The paper opens up new avenues for research in the intersection of computation, chemistry, and the study of life's origins.

“What would be conserved if "the tape were played twice"?” Fontana, W., & Buss, L. W. (1994, January 20)
This paper introduces an abstract chemistry, implemented using λ calculus, to explore the concepts of contingency and necessity in evolution. The authors argue that certain features are generic to their abstraction of chemistry and would likely reappear if evolutionary processes were replayed. These features include the emergence of hypercycles of self-reproducing objects, the formation of self-maintaining organizations when self-replication is inhibited, and the ability of these self-maintaining organizations to combine into higher-order structures. The model intentionally lacks explicit thermodynamic notions, spatial constraints, conservation laws, and unequal reaction rates, focusing instead on the minimal features abstracted from chemistry. Inspired by Gould's question of whether the same biological diversity would arise if evolution were replayed, the authors aim to identify features of life that are likely to recur. They address the challenge of studying historical contingency and necessity by creating a model universe where such explorations are possible. The central question is: What fundamental features of biological organization are generic enough to reappear if evolution were replayed under similar conditions, and can these features be identified through an abstract computational model of chemistry? The authors present results from computer experiments using their λ calculus-based model. In Level 0 experiments, where self-copying is allowed, the system consistently evolves towards dominance by either single self-copying functions or hypercycles (mutually catalytic sets of functions). This mirrors Eigen and Schuster's work on hypercycles. In Level 1 experiments, where self-copying is inhibited, the system generates complex self-maintaining organizations. These organizations consist of sets of objects that maintain themselves without any single member engaging in direct copying. These self-maintaining organizations are governed by emergent laws at both the syntactical and functional levels. These laws describe the allowed forms of combination of building blocks and the transformations among objects, respectively. The authors propose (though this is not explicitly demonstrated in a figure) that self-maintaining organizations, once established, can combine into higher-order self-maintaining organizations.

“‘The Arrival of the Fittest’: Toward a Theory of Biological Organization” Fontana, W., & Buss, L. W. (1994, January)
In this article, the authors explore the "existence problem" in evolutionary theory: how biological organizations (alleles, individuals, populations) arise in the first place, before natural selection can act upon them. The authors develop a minimal theory of biological organization using λ calculus to abstract fundamental features of chemistry: the constructive nature of molecular collisions and the diversity of ways to arrive at the same stable product. They employ a stochastic flow reactor to explore the behavior of interacting λ expressions, showing the emergence of fixed systems of transformation characterized by syntactical and functional invariances. These organizations are self-maintaining, robust to perturbation, and possess a definable center. The authors were motivated by the limitations of traditional evolutionary theory, which assumes the prior existence of the entities it seeks to explain. Inspired by the problem of the origin of life and the emergence of multiple levels of organization in the history of life, they seek a framework to understand how these entities are generated in the first place. They argue that focusing solely on the dynamics of alleles, individuals, and populations neglects the underlying question of how these entities arise from simpler components. The main question of the article is as follows: how can we develop a minimal theory of biological organization that explains the emergence of self-maintaining and evolvable entities from simpler components, without relying on prior assumptions about their existence or specific properties, and can λ calculus serve as a useful abstraction for this purpose? The authors introduce the concept of a "constructive dynamical system," where interactions between objects internally construct new objects, leading to a dynamic in phase space and a dynamic of the system's "support" (the set of objects present). They argue that λ calculus provides a natural way to abstract essential features of chemistry, such as the constructive capability of molecules and the existence of multiple pathways to the same product. They show that, within their stochastic flow reactor model, self-maintaining organizations emerge from the collective behavior of λ expressions. These organizations are characterized by syntactical and functional regularities. They also propose a hierarchy of organizational levels: Level 0: Self-copying objects or simple ensembles of copying objects. Level 1: Self-maintaining organizations made of Level 0 objects. Level 2: Self-maintaining meta organizations composed of Level 1 organizations.


## Supercollider Simulation Engine

To support high-throughput experimentation, we developed a high-performance Rust simulation engine, titled the **Functional Supercollider**, which serves as the computational backend for the AlChemy dashboard. The engine implements the core logic of Algorithmic Chemistry and uses a generic trait system to decouple simulation dynamics from particle representation.

The simulation models a reactor, or *Soup*, containing a population of particles. Each particle encapsulates a lambda calculus expression representing an abstract molecule. Evolution proceeds through a stochastic collision loop consisting of three stages:

**Interaction.** Two particles, $A$ and $B$, are randomly selected. Their interaction is defined as function application, producing a new expression:

$$
(A \ B)
$$

**Reaction (Reduction).** The resulting expression is reduced toward normal form. Because lambda calculus is Turing-complete, reduction may not terminate. To ensure bounded computation, the engine enforces limits on reduction steps ($rlimit$) and expression depth ($slimit$). Expressions exceeding these limits are discarded.

**Filtration.** Successfully reduced expressions may be filtered to remove trivial results, such as identity functions or direct reproductions of parent expressions. This promotes diversity and prevents stagnation.

The engine also supports conditional selection through *recursive collision events*. Certain particles act as test functions that evaluate other particles via functional application. If the result reduces to a Church-encoded true value, the tested particle is amplified by reintroducing multiple copies into the population. Otherwise, no amplification occurs. This introduces programmable selection pressure and enables guided exploration of functional expression space.

The Functional Supercollider is exposed to Python through native extensions, enabling integration with the visualization dashboard while maintaining the performance benefits of Rust.

## Experimental Approach

### Visualizing AlChemy Dashboard
- This dashboard serves as a UI that allows users to interact with the Alchemy Core simulator and generate experiments and render interactive plots. This dashboard uses Python + Bokeh to achieve this task.

#### Design goals
- The main goal is to provide an easy to use UI interface for users to run experiments on and visualize the reults of collisions rendered by the Alchemy core simulator
- Users can upload JSON files to analyze/visualize experiment outcomes, perform time series analysis of their experiments as well as export and share their results.
- Users have the option to adjust the total number of collisions to run, polling frequency, random seed, generator type (between Fontana, Btree and uploaded files), Abstraction range, application range, max depth, min depth, max free variables, free variable probability, and initial expression count.
- There are 4 visualization plots that this dashboard generates
    - 1: Entropy Over time (The amount of disorder caused by lambda collisions over time)
    - 2: Unique expressions over time (How many unique expressions are formed by these collisions over time)
    - 3: State histogram at collision (lists the top 20 expressions by frequency)
    - 4: AST visualization (creates an AST tree of selected lambda expression)
  We also included a multi dendrogram tool that allows users to compare the evolution of expressions across various simulations utilizing the levenstein distance formula    
#### Implementation
##### Dashboard Structure
- High level description of the Rust library (functional supercollider)
- This dashboard is implemented through a combination of Python/Bokeh for the UI and Rust extensions that handle collision simulation
- Users have the choice of UV or vanilla pip to build python dependency of Rust extensions
- The overall structure goes like this: the user prodvides input to the UI -> the UI runs the simulation with the Alchemy core simulation -> experiment results are outputted -> the user also recieves interactive visualization plots generated by the alchemy dashboard.
Several visualization tools have been created to enable users to analyze their findings across various simulations.
##### Visualization Tools
1) Entropy graph: this graph plots snapshots of collisions veruss its calculated entropy from metadata provided by the backend collision code. These snapshots are equivalent to time as it represents the state of the soup at a particular collision number. This visualization enables users to understand how disordered their soup of expression gets over each collision and provides a deeper explination as to how populations survive through various entropy points.
2) Unique Expressions graph: Similar to the previous graph, this visualization takes a look at how many unique expressions are present in each snapshot of collisions in the soup. This allows foor users to see how often new expressions form as more and more collisions take place in the soup enabling for analyzing whether or not a specific environment inhibits or promotes diverse species.
3) Jaccard Vs Bray-Curtis Dissimilarity Plots: We decided to put the jaccard vs bray-curtis plots side by side as both are pretty similar, however these graphs gives usuers another visual mode to analyze. The jaccard index Compares if different ecosystem shares similar species (does not care about exact count). it is represented by this equation: |A and B| / |A OR B|.
This is how it was implemented in python:
- 1)Extract names of species from selected snapshots (collision times) and ignore the count
- 2)Find all shared and unique species between the snapshots
- 3)Calculate jaccard distance (similarity = count(shared_species) / count(all_unique_species))
The bray curtis dissimilarity compares the quantity of similar species in different ecosystems . It is represented by this equation: <img width="147" height="58" alt="Screenshot 2026-04-12 163249" src="https://github.com/user-attachments/assets/7929600d-46d7-400d-82f0-e7ae2e6a1aa3" />
This is how it was implemented in python:
- 1)Get the counts of all species in each of the snapshots
- 2)Calculate the difference between each species count 
- 3)Divide difference sum by total sum of species (difference_sum / total_population_sum)

 4) Abstract Syntax tree: The abstract syntax tree allows users to understand how expressions in the soup evolved to its final collision state.
This is how it was implemented in python:
- 1)Break down expression into tokens ( separates characters and symbols)
- 2)Define 3 different node blueprints (Variable, lambda, application) 
- 3)Recursively move through tokens to rebuild expression according to node blueprints 
        - If it sees a \ it is building a lambda
        - If it sees a ( it recursively parses again
        - If it is just the letter it builds a variable node
- 4)Connect all nodes sitting together using application blueprint-> it connects two tokens next to eachother and then connects those formed tokens until the expression is complete
- 5) In order for this tree to be drawn, the AST is traversed and drawn

5) Molecular Sequence Alignment: Inspired by the BLAST gene sequence alignment tool creted by NCBI, we decided to replicate a similar tool that can be used to further analyze the final state of expression s in a simulation on a mathematical level. This tool utilizes the levenstein distance library to caluclate the edit distance and identity% of a selected expression to other expressions in the soup.
Here is how it is implemented:
- 1)Obtain users input of expression selected
- 2)Obtain the top 100 most populated unique molecules in the simulation
- 3)For each molecule in top 100 selected calculate levenstein distance and find the max length between the two compared expressions
- 4)Calculate identity (1 - (Levenstein Distance / MaxLength)) * 100
- 5)Sort all compared molecules by highest identity percentage and return top 10 matches

6) Multi simulation Dendrogram Comparison:Inspired by phylogenetic trees, we created a dendrogram in a subsection of the dashboard to allow users to compare the evolution of expressions across various simulations. This allows for users to visualize how small changes across simulations can affect the popularity and form of expressions.
Here is how it is implemented:
- 1)Create empty dictionary to track expressions
- 2)For each simulation: grab top expressions (number of expressions determined by user)
- 3)Assign a unique color to the simulations
- 4)For each expression: add to dictionary if not yet tracked, record simulations associated with expression 
- 5)Create a list of every unique expression found across all selected simulations
- 6)Calculate Levenstein distance between every possible pair of expressions
- 7)Use the linkage algorithm to build the dendrogram tree (linkage(squareform(dist_matrix), method='average'))
- 8)Draw the graph and color code the expressions based on its experiment ID (if it is attached to more than one simulation then color it black)



##### Experimental Tools

1)Simulate extinction: This tool allows for users to simulate an extinction even by taking an expression and removing it from the soup. The user can pick between the most popular expressions however the most popular expression of all time is highlighted to allow users to replicate a high stakes extinciton event
Implementation Steps:
- 1)Create a new empty dictionary for survivors 
- 2)For each expression present in the final state of the simulation, if it matches the selected target expression, get rid of it else add that expression and its count to the dictionary
- 3)Reconstruct the soup and configure a new simulation, the dictionary has the expression and its count so multiply the expression by that count so that it appears in the soup that many times and is ready for more collisions

2) Simulate Invasive Species: This tool allows for users to add an certain number of copies of an invasive species expression to the soup to replicate how disruptive lambda expressions can affect the envrionment over time and if the environment will be able to still reach equilibrium after a disruption in its environment. the default expression is the identity function /x.x hwoever, users are free to type in any expression they see fit and adjust the number of copies.
Implementation steps:
- 1)Create a new empty soup of expressions and re-add already present molecules
- 2)Add the selected invasive molecule at the specified amount of times to the soup
- 3)Set up a new simulation using this soup and execute the specified amount of collisions



### Numerical Experiments

#### Motivation, connection to real chemistry

#### Characterizing AlChemy Simulations 
- Similarity between simulations (Jaccard vs Bray–Curtis)
- Similarity within simulations (Edit distance for lambda expressions)
- Similarty of expressions between different simulations (Multi-dendrogram edit distance for lambda expressions)
- Similarity of simulations that use different seeds
- Similarity of expressions in recursive simulations
- Similarity of expressions before and after an extinction event (removing the most populated expression from a simulation)
- Similarity of expressions before and after the introduction of an invasive species (adding copies of the identity funciton /x.x 50,100, and 1000 times to comapre the effective in each number of copies)

#### Initial Conditions, recursive steps, non-recursive controls
- All original simulations are ran with:
1) Seed 42
2) 1000 Collisions
3) polling 10
4) Fontana Generator
5) Abs rang 0.1- 0.5
6) app range 0.2 - 0.6
7) min depth 1
8) max depth 5
9) max free vars 2
10) free var prob 0.5
11) intial count 10
- To ensure that a comparison group (outgroup) was used as a mode of comparison in the experiments, an simulations was ran on church numerial expressions. The purpose behind choosing this outgroup is that church numerals represent the nautral numbers and grow linearaly compared to the expressions that collide in the generator. This ensures that the basic iterating nature of a church numeral expression acts as a checkpoint against the more complex evolving lambda expressions.
- I used 10 church numerials to match that there are 10 expressions originally placed in the soup, i used seed 7 and all other settings aligned with the original simulations:
0) \f.\x.x
1) \f.\x.f x
2) \f.\x.f(f x)
3) \f.\x.f(f(f x))
4) \f.\x.f(f(f(f x)))
5) \f.\x.f(f(f(f(f x))))
6) \f.\x.f(f(f(f(f(f x)))))
7) \f.\x.f(f(f(f(f(f(f x))))))
8) \f.\x.f(f(f(f(f(f(f(f x)))))))
9) \f.\x.f(f(f(f(f(f(f(f(f x))))))))

1) Experiment 1: Comparing Different Seeds 
- Step 1: Generate 3 different Fontana experiments with 1000 collisions and seeds 42, 280, and 3017 respectively with default values and initial count 10.
- Step 2: Generate an outgroup using Church numerals and random seed 7
- Step 3: Compare using multi experiment Dendrogram
  
2) Experiment 2: Recursive Experiment
- Step 1: Generate a fontana simulation with 1000 collisions, seed 42, and 10 initial expressions the rest is default settings
- Step 2: Use my base simulation to create 3 recursions (generations) of that experiment
- Step 3: Generate an outgroup using Church numerials and random seed 7
- Step 4: Compare metrics utilizing dashboard tools
  
3) Experiment 3: Extinction (Most Popular Expression)
- Step 1: Generate a fontana simulation with 1000 collisions, seed 42, and 10 initial expressions the rest is default settings
- Step 2: utilizing the extinciton function b utton, delete the most populated expression from the simulation

4) Experiment 4: Invasive Species 
- Step 1: Generate a fontana simulation with 1000 collisions, seed 42, and 10 initial expressions the rest is default settings
- Step 2: Add 50 copies of the identity function /x.x and note changes
- Step 3: Add 100 copies of the same identity expression, note changes
- Step 4: Ad 1000 copies of the same identity expression, note changes
- Step 5: Final multi experiment dendrogram comparison across each checkpoint of adding more copies of the identity expression


## Results
1) Experiment 1: Comparing Different Seeds
- Exp 1 = seed 42
- Exp 2 = Seed 280
- Exp 3 = Seed 3017
- Exp 4 = OUTGROUP
  
<img width="1291" height="692" alt="Screenshot 2026-04-12 215430" src="https://github.com/user-attachments/assets/0cc1b54b-a494-4f12-b681-3c15281a210e" />
- Observations:
- Each simulation has their own cluster of similar expressions shown in the dendrogram
-  exp 1 and exp2 have some shared expressions however, exp 3 does not have any shared expressions with any of the other experiments -> this could suggeste that some expressions are more likely to be created regardless of the seed.
-  One interesting observation was that even very complex and long lambda expressions were shared between seed 42 and seed 280, although a lot of the shared expressions are shorter and simple, this image below tells a different story
  <img width="1136" height="803" alt="Screenshot 2026-04-13 202749" src="https://github.com/user-attachments/assets/a970d7a6-8863-4224-8ac6-fa9bf36585c8" />


2) Experiment 2: Recursive Experiment
   -Exp 4 = OUTGROUP
   - Exp 5 = Parent Generation (Gen 1)
   - Exp 6 = Generation 2
   - Exp 7 = Generation 3
   - Exp 8 = Generation 4
<img width="1548" height="834" alt="Screenshot 2026-04-13 203633" src="https://github.com/user-attachments/assets/4073d28c-d387-4ca3-a07a-f9f5645b2cd6" />
- Observations:
- There are no unique nodes shown for generation 3 and generation 4, they are only present in the shared black nodes, this may inidcate that the system has reached a fixed point
- There are only about 4 shared expresisons, 3 of which are shared across all generations. These expressions are increidbly long and complex as shown in the screenshots 
<img width="406" height="362" alt="Screenshot 2026-04-13 204543" src="https://github.com/user-attachments/assets/0418478c-0af5-4c22-8a68-7a8ec5962d38" />
<img width="452" height="358" alt="Screenshot 2026-04-13 204443" src="https://github.com/user-attachments/assets/f7d47010-7335-4345-a360-8a92da8ff093" />
<img width="486" height="569" alt="Screenshot 2026-04-13 204431" src="https://github.com/user-attachments/assets/1aa6b8ad-0b59-40a3-a760-03fe50ba3e49" />

## Discussion

### Next steps for AlChemy
### Implications for Prebiotic Chemistry experiments
