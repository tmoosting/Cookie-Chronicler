# Cookie-Chronicler
<sub>Based on an assignment for Masters course: [Computational Creativity](https://studiegids.universiteitleiden.nl/courses/103312/computational-creativity) at Leiden University
</sub>


An evolutionary algorithm cookie recipe generator

### Recipe Cookbook: Examples
[Link](https://drive.google.com/file/d/1AreaEpLrHYjNsJ35Ek2kQ6mYoZnGzMIQ/view?usp=sharing) 

### System Requirements
* python 3.6
* jupyter

### Files
* Logbook.ipynb: report and execution
* cookie_class.py: python code and dataclass `CookieGenerator`
* DataImport - Ingredients.csv: ingredient data sheet
* DataImport - RecipesFull.csv: recipes data sheet
* .jpegs in /pics folder: used for recipe output (linked to environment condition)


### Running the application

Make sure you have all the files above & that your working directory is set where the above files are. 

```
from cookie_class import CookieGenerator
cookie = CookieGenerator(environment_condition = "Cookie Cutter")
cookie.run_genetic_algorithm(1000) 
```
The details of this are outlined in the Logbook as are the various variables that you can pass to the CookieGenerator class:

```
environment_condition: str = "Cookie Cutter" # conditions for cookies
fitness_method:str = "Manhattan"
mutation_chance:float = .20 # probabilty of mutation chance
parent_selection_size: int =  10 # the size of the pool to choose parents from
extinction_event_chance:float = 0.01 #  1 % of recipe population lost over all generations
min_number_generations: int = 100 # minimum number of gnerations to run GA for
stopping_criteria_runs: int = 100 # checking if latest poulation fitness less than or equal all previous number runs 
file_path_recipes = "DataImport - RecipesFull.csv"
file_path_ingredients = "DataImport - Ingredients.csv"
```


