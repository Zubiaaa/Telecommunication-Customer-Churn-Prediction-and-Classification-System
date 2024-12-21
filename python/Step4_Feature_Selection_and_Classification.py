#!/usr/bin/env python
# coding: utf-8

# In[1]:


# If you do not already have this package install it by uncommenting the below line:
#pip install scikit-learn-intelex


# In[2]:


# Below 2 lines accelerate sklearn algorithms while using the familiar scikit-learn package and getting the same results
from sklearnex import patch_sklearn
patch_sklearn()


# In[3]:


# import relevant libraries:
import random
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score


# In[4]:


# importing Dataset
dataset = pd.read_csv('Customer_Churn_Final_Dataset.csv')
dataset.head()


# In[5]:


# Returning a tuple representing the dimensionality of the Dataset
dataset.shape


# In[6]:


# Returning the column labels of the DataFrame
dataset.columns


# In[7]:


# Separate input features and target:
    
target_variable = dataset["Churn"]
independent_features = dataset.drop(columns="Churn")
independent_features


# In[8]:


# How many independent features do we have?
feature_size = independent_features.shape[1]
feature_size


# In[9]:


# Creating a list of column numbers in independent features dataset:
def total_features():
    features = []
    # including all the independent variables in the list "features" 
    features = list(range(feature_size))
    return features


# In[10]:


# Column Number list:
features_list = total_features()
print(features_list)


# In[11]:


class Customer_Churn:
    
    ############## Initialize variables and lists ##############
    def __init__(self):
        
        self.features = total_features()
        self.chromosome_size = feature_size
        self.population = []
        self.chromosome = []
        self.population_size = 12
        self.number_of_generations = 100
        self.optimal_score = 55
        
    ############## Create Random Chromosome ##############
    def createRandomChromosomes(self):
        randomChromosome = []
        for i in range(self.chromosome_size):
            randomChromosome.append(random.randint(0,1))
            self.chromosome = randomChromosome
        return randomChromosome
    
    ############## Initial Population ##############
    def initialPopulation(self, population_size):
        population = []
        for i in range(0, population_size):
            population.insert(i, self.createRandomChromosomes())
        self.population = population
        return population
    
    ############## Model's Score Calculation Function ##############
    def scoreCalculator(self, X, y):

        # Trying another validation technique, K-fold Cross-validation
        kfold = KFold(n_splits=10)
        
        # Building Gaussion Naive Bayes model
        model = GaussianNB()
        
        # Using f1_score as it is an imbalanced dataset
        scores = cross_val_score(model, X, y, scoring='f1_macro', cv=kfold)
        
        # Returning the score
        return scores.mean()*100.0
    
    ############## Fitness Function ##############
    def fitness(self, chromosome):
        fitness_values = []
        # creating a selected feature subset to feed it in our model (initially taking all independent features)
        features_subset = independent_features
        chromosome_fitness = 0
        
        # starting from column number 0, will then iterate through all
        column_number = 0
        
        # iterating through columns in the datasett containing only idependent features
        for column_name in independent_features:
            # column number should not be greater than the chromosome size
            if column_number <= self.chromosome_size:
                # checking the value in the chromosome (taking column number as the index) if it's equal to 0
                # 0 in chromosome means, that feature is not selected in that chromosome 
                if(chromosome[column_number]==0):
                    # if it is true, then drop that column from the feature subset
                    # creating a feature subset for our supervised learning model
                    features_subset = features_subset.drop(column_name, axis=1)
                    
                column_number += 1

        X = features_subset
        y = target_variable
        # Calculating the score for our selected feature subset using Gaussian Naive Bayes model
        chromosome_fitness = self.scoreCalculator(X, y)
        # return that score
        return chromosome_fitness
        
    ############## Ranking Chromosomes ##############
    def rankChromosomes(self, population):
        fitnessResults = {}
        ranked_population = []
        ranked_score = []
        # looping through a population
        for i in range(0,len(population)):
            # placing every chromosome's fitness/score in the dictionary (taking population index as the key)
            fitnessResults[i] = self.fitness(population[i])
        # a list of sorted scores in descending order:
        ranked_score_list = list(sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True))
        
        # looping through the ranked scores list
        for i in ranked_score_list:
            # converting the tuple into a list
            i = list(i)
            # population's index is in i[0]
            index = i[0]
            # a ranked population:
            ranked_population.append(population[index])
            
        # looping through the ranked scores list
        for i in ranked_score_list:
            # converting the tuple into a list
            i = list(i)
            # chromosome's score is in i[1]
            score = i[1]
            # a ranked score list:
            ranked_score.append(score)
        return ranked_population, ranked_score
    
    ############## Selecting Elite Chromosomes ##############
    def elitism_selection(self, population, elitism_rate  = 0.2):
        selected_elites = []
        fitness_values = []
        # selecting 20% elites to create a robust population
        ranked_population, ranked_score = self.rankChromosomes(population)
        population_size = len(ranked_population)
        # Mathematical calculation for takig out 20% of the population:
        elite = int(population_size * elitism_rate)
        # selecting 20% elites
        selected_elites = ranked_population[:elite]
        # selecting 20% elite's fitness values
        fitness_values = ranked_score[:elite]
        return selected_elites, fitness_values
    
    ############## Tournament Selection Function ##############
    def tournament_selection(self, population):
        winner = []
        parentA, parentB = [], []
        fitness_values = []
        random_index1 = 0,
        random_index2 = 0
        length = len(population)
        # 2 rounds for our tournament, each mutually exclusive tournament will have 2 random selected parents,
        # battling against each other to win
        for i in range(2):
            # 2 random indices
            random_index1 = random.randint(0, (length-2))
            random_index2 = random_index1+1
            
            # picking out 2 random parents from the population
            parentA = population[random_index1]
            parentB = population[random_index2]
            
            # calculating their fitness
            fitness_of_A = self.fitness(parentA)
            fitness_of_B = self.fitness(parentB)
            
            # the one with the greater fitness wins the tournament
            if fitness_of_A >= fitness_of_B:
                winner.append(parentA)
                fitness_values.append(fitness_of_A)
                
            else:
                winner.append(parentB)
                fitness_values.append(fitness_of_A)
        # Shuffling chromosomes in a population, to enable random crossover
        random.shuffle(winner)
        return winner, fitness_values

    ############## Crossover Function ##############
    def crossover(self, selected, crossover_rate = 0.6):
        # extracting parents
        parentA = selected[0]
        parentB = selected[1]
        # copying each parent to their children
        child1, child2 = parentA.copy(), parentB.copy()
        # generating a random number
        r1 = random.uniform(0, 1)
        # performing crossover only if the crossover rate is more than the random number
        if crossover_rate > r1:
            # randomly generating a crossover point that is not at the end of the string
            crossover_point = random.randint(1, len(parentA)-2)
            # crossing over using randomly generated crossover point
            child1 = parentA[:crossover_point] + parentB[crossover_point:]
            child2 = parentB[:crossover_point] + parentA[crossover_point:]
        return child1, child2
    
    ############## Mutation Function ##############
    def mutation(self, child1, child2, mutation_rate = 0.08):
        # looping over the first child
        for i in range(len(child1)):
            # generating a random number
            r1 = random.uniform(0, 1)
            # if the random number is less than the mutation rate, flip the bit in the child at a random index
            if r1 < mutation_rate:
                # flip the bit
                j = random.randint(0, len(child1)-1)
                temp = child1[i]
                child1[i] = child1[j]
                child1[j] = temp
        # looping over the second child
        for i in range(len(child2)):
            # generating a random number
            r1 = random.uniform(0, 1)
            # if the random number is less than the mutation rate, flip the bit in the child at a random index
            if r1 < mutation_rate:
                # flip the bit
                j = random.randint(0, len(child2)-1)
                temp = child2[i]
                child2[i] = child2[j]
                child2[j] = temp
        return child1, child2
    
    ############## Create Next Population ##############
    def create_next_population(self, current_population, population_size):
        # Initializing variables:
        next_population = []
        elite_parents, tournament_winner_parents = [], []
        new_offspring1, new_offspring2, new_offspring3, new_offspring4, = [], [], [], []
        mutated_children1, mutated_children2 = [], []
        fitness_values_elites, fitness_values_tournament_winners = [], []
        max_fitness_value_elite = 0
        max_fitness_value_tournament_winner = 0
        maximum_fitness_values_per_iteration = []
        maximum_fitness_value = 0
        winner = []
        
        # Creating the next population:
        
        # Returning selected 20% elite chromosomes and their score/fitness values: 
        elite_parents, fitness_values_elites = self.elitism_selection(current_population)
        # Crossing over elite chromosomes to create a hybrid in the next generation 
        # To preserve the good properties of elites to next generations, I crossedover the elites and tournament winners separately
        new_offspring1, new_offspring2 = self.crossover(elite_parents)
        # Appending the next population with crossedover chromosomes:
        next_population.append(new_offspring1)
        next_population.append(new_offspring2)
        
        # While adding (crossedover/mutated) chromosomes to the next population,
        # ensuring its length doesn't exceeds the fixed population size
        while len(next_population) < population_size:
            # Returning winner chromosomes of the tournament and their score/fitness values:
            tournament_winner_parents, fitness_values_tournament_winners = self.tournament_selection(current_population)
            
            # Crossing over tournament's winner chromosomes to create a hybrid in the next generation
            new_offspring3, new_offspring4 = self.crossover(tournament_winner_parents)
            
            # Mutating only the tournament's winner parents offsprings and not the elite's parents offsprings to ensure robustness
            mutated_children1, mutated_children2 = self.mutation(new_offspring3, new_offspring4)
            
            # Appending the next population with mutated chromosomes:
            next_population.append(mutated_children1)
            next_population.append(mutated_children2)
            
            # Below code keeps track of the chromosome with the highest fittness among all the chromosomes in a population:
            # 1. Checking the maximum chromosome among elites:
            max_fitness_value_elite = max(fitness_values_elites)
            for i in range(len(fitness_values_elites)):
                if fitness_values_elites[i] == max_fitness_value_elite:
                    elite = list(elite_parents[i])
            
            # 2. Checking the maximum chromosome among Tournament's winners:
            max_fitness_value_tournament_winner = max(fitness_values_tournament_winners)
            for i in range(len(fitness_values_tournament_winners)):
                if fitness_values_tournament_winners[i] == max_fitness_value_tournament_winner:
                    tournament = list(tournament_winner_parents[i])
            
            # 3. Checking the maximum chromosome among the maximum Tournament's winner and maximum elite
            # This is called the population's winner
            if max_fitness_value_elite >= max_fitness_value_tournament_winner:
                maximum_fitness_values_per_iteration.append(max_fitness_value_elite)
                winner = [elite]
            else:
                maximum_fitness_values_per_iteration.append(max_fitness_value_tournament_winner)
                winner = [tournament]
        
        maximum_fitness_value = max(maximum_fitness_values_per_iteration)
        
        # Returning the next population, the chromosome with the highest fittest among all the chromosomes in 1 populaton,
        # and it's score
        return next_population, maximum_fitness_value, winner
        
    ############## Run Main Genetic Algorithm ##############      
    def geneticAlgorithm(self):
        # Initializing variables:
        generation = 1
        found_optima = False
        max_score = 0
        maximum_fitness_value = []
        maximum_fitness_value_per_population = []
        fittest_chromosome = []
        final_maximum_fitness_value = 0
        chromosome_with_max_score = []
        winner = []

        # Creating the initial population (with random bits)
        current_population = self.initialPopulation(self.population_size)
        print("Initial population:", current_population)

        # While adding the new (next) populations to create generations,
        # ensuring its length doesn't exceeds the fixed number of generation's limit
        while generation < self.number_of_generations+1:
            # Creating the Next Population
            current_population, maximum_fitness_value, winner = self.create_next_population(current_population, self.population_size)
            print("Generation",generation,":", current_population)
            
            # Saving all the maximum fitness values (maximum among all the chromosomes in 1 population)
            maximum_fitness_value_per_population.append(maximum_fitness_value)
             # Saving all the chromosomes with the highest fittest (highest among all the chromosomes in 1 populaton)
            fittest_chromosome.append(winner)
            
            # Incrementing the generation
            generation += 1
            
            # If any chromosome in the current population is the optimal solution, stop running any more generations
            for fitness in maximum_fitness_value_per_population:
                # Checking if any chromosome in the next population matches the optimal solution
                if fitness > self.optimal_score:
                    print("Optimal solution found in", generation-1)
                    found_optima = True
                    break
            # If matches, finish creating any more generations and return the solution
            if found_optima == True:
                break
        
            
        # Below code keeps track of the chromosome with the highest fittness among every population:
        # Checking for the maximum chromosome among all the chromosomes in all populations:
        final_maximum_fitness_value = max(maximum_fitness_value_per_population)
        for i in range(len(maximum_fitness_value_per_population)):
            if maximum_fitness_value_per_population[i] == final_maximum_fitness_value:
                winner = fittest_chromosome[i]
        # Printing the number of generation and the winner chromosome with its score/fitness value
        print("Completed ", generation-1, "Generations")
        print("The Winner Chromosome with the Maximum Score ", final_maximum_fitness_value, "is:", winner)
        
        # Returning the winner chromosome for further modelling
        return winner


# In[12]:


c = Customer_Churn()


# In[13]:


winner = c.geneticAlgorithm()


# In[14]:


# Removing double bracket from the list called "winner"
winner = winner[0]
print(winner)


# In[15]:


# Transforming winner chromosome into column numbers:
def transform_chromosome_to_features_list(chromosome):
    independent_features_list = []
    for i in range(len(features_list)):
        if (chromosome[i]==1):
            independent_features_list.append(features_list[i])
    return independent_features_list


# In[16]:


chromosome = winner
independent_features_list = transform_chromosome_to_features_list(chromosome)
print(independent_features_list)
print("'",len(independent_features_list),"'", " Features are selected")


# In[17]:


# Transforming column numbers into a dataframe (our "Features Subset"):
def transform_features_list_to_features(independent_features_list):
    independent_features_subset = pd.DataFrame()
    index = 0
    for column_number, column_name in enumerate(independent_features.columns):
        for i in independent_features_list:
            if (column_number == i):
                independent_features_subset[index] = independent_features[column_name]
                independent_features_subset.rename(columns = {index : column_name}, inplace=True)
                index += 1
    return independent_features_subset


# In[18]:


independent_features_subset = transform_features_list_to_features(independent_features_list)
independent_features_subset.head()


# In[19]:


# The features subset contains less number of columns than in the full dataset (but the same number of rows)
independent_features_subset.shape


# In[20]:


X = independent_features_subset
y = target_variable


# In[21]:


# Trying a validation technique, K-fold Cross-validation
kfold = KFold(n_splits=10)
# Building Gaussion Naive Bayes model
model = GaussianNB()
# Using f1_score as it is an imbalanced dataset
score_kfold = cross_val_score(model, X, y, scoring='f1_macro', cv=kfold)
print("F1 Score: %.3f%%" % (score_kfold.mean()*100.0)) 


# In[22]:


# evaluate model
scores = cross_val_score(model, X, y, scoring='precision', cv=kfold)
# summarize performance
print("Precision: %.3f%%" % (scores.mean()*100.0))


# In[23]:


# evaluate model
scores = cross_val_score(model, X, y, scoring='recall', cv=kfold)
# summarize performance
print("Recall: %.3f%%" % (scores.mean()*100.0))

