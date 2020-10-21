import pandas as pd
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
import math
from IPython.display import Image, display, Markdown
plt.style.use('ggplot')


#Useful Functions
def make_markdown_table(array):

    """ Input: Python list with rows of table as lists
               First element as header. 
        Output: String to put into a .md file."""


    markdown = "\n" + str("| ")

    for e in array[0]:
        to_add = " " + str(e) + str(" |")
        markdown += to_add
    markdown += "\n"

    markdown += '|'
    for i in range(len(array[0])):
        markdown += str("-------------- | ")
    markdown += "\n"

    for entry in array[1:]:
        markdown += str("| ")
        for e in entry:
            to_add = str(e) + str(" | ")
            markdown += to_add
        markdown += "\n"
        
    return markdown + "\n"

qualifiers = ['rich','creamy','distinctive','salty','chewy','nobake','sweet','nutty','moist','flat','crispy','softcentre','grownup','creamy', 'earth','childhood' 'favorite','tangy','traditional','tart','glistening','tidy', 'fast','robust','homemade','mellow','crumbly','delicate','flavorful','golden','fudgy','decadent','nutritional','nutty','easy','delicate'] 

def create_quantity_converter_dict():
    """Conversion for measurements to grams"""
    quantity_converter_dict = {}
    quantity_converter_dict['oz'] = 30
    quantity_converter_dict['cup'] = 160
    quantity_converter_dict['tbsp'] = 15
    quantity_converter_dict['tsp'] = 3
    quantity_converter_dict['pinch'] = 1
    quantity_converter_dict['piece'] = 50
    quantity_converter_dict['sprinkle'] = 5
    quantity_converter_dict['gram'] = 1
    return quantity_converter_dict 

quantity_converter_dict = create_quantity_converter_dict()



@dataclass
class CookieGenerator:
    environment_condition: str = "Cookie Cutter" # conditions for cookies
    fitness_method:str = "Manhattan"
    mutation_chance:float = .20 # probabilty of mutation chance
    parent_selection_size: int =  10 # the size of the pool to choose parents from
    extinction_event_chance:float = 0.01 #  1 % of recipe population lost over all generations
    min_number_generations: int = 100 # minimum number of gnerations to run GA for
    stopping_criteria_runs: int = 100 # checking if latest poulation fitness less than or equal all previous number runs 
    file_path_recipes = "DataImport - RecipesFull.csv"
    file_path_ingredients = "DataImport - Ingredients.csv"
    recipe_dict:dict = field(default_factory=dict)
    ingredient_purpose_dict: dict = field(default_factory=dict)
    ingredient_occurence:dict = field(default_factory=dict)
    ingredient_association:dict = field(default_factory=dict)
    final_recipe_dict:dict = field(default_factory=dict)
    purpose_average_proportions:dict = field(default_factory=dict)
    track_population_fitness:list = field(default_factory=list)
    track_max_population_fitness:list = field(default_factory=list)
   
    def import_data(self, file_path: str) -> pd.DataFrame:
        """ Imports data from file."""
        return pd.read_csv(file_path)
    
    @staticmethod
    def convert_all_amounts_to_gram(recipes:pd.DataFrame, quantity_converter_dict: dict)-> pd.DataFrame:
        """Converts all recipes' amounts values (cups, oz etc.) to grams."""
        recipes['amount'] =recipes.apply(lambda row: row["amount"]* quantity_converter_dict.get(row['units']), axis =1)
        return recipes
       
    def create_recipe_dict(self,recipes: pd.DataFrame)-> None:
        """ Creates dictionary with names as keys & values list of ingredients."""
        for name in recipes['Name'].unique():
            self.recipe_dict[name]= {}
            self.recipe_dict[name]["ingredients"] = list(recipes[recipes['Name']==name].ingredient)
            self.recipe_dict[name]["quantities"] = list(recipes[recipes['Name']==name].amount)
            self.recipe_dict[name]["purposes"] = list(recipes[recipes['Name']==name].Purpose)
    
    def create_ingredient_purpose_dict(self,ingredients: pd.DataFrame)-> None:
        """ Creates dictionary with ingredients as keys & values their purposes."""
        for ingredient in ingredients['Ingredient'].unique():
            self.ingredient_purpose_dict[ingredient]= ingredients[ingredients.Ingredient == ingredient]['Purpose'].values.item()

            
       
    def create_average_purpose_proportion_dict(self, recipes:pd.DataFrame)-> None:
        """Sum up all proportion values for each purpose type, then average it out into one dict"""
        purpose_proportions = {}
        for purpose in list(recipes.Purpose.unique()):
            purpose_proportions[purpose] = []
        for recipe in list(recipes.Name.unique()):
            df = pd.DataFrame( { "Purpose": recipes[recipes.Name == recipe]['Purpose'], "Quantity": recipes[recipes.Name == recipe]['amount']})
            df = pd.DataFrame(df.groupby('Purpose')['Quantity'].sum()/df['Quantity'].sum()).reset_index()
            for purpose in list(df.Purpose):
                purpose_proportions[purpose].append(df[df.Purpose == purpose]['Quantity'].values.item())
        self.purpose_average_proportions = {k: np.mean(v) for k, v in purpose_proportions.items()}
        # rescale so proportions sum to one
        self.purpose_average_proportions = {k: (v/sum(self.purpose_average_proportions.values()))  for k, v in self.purpose_average_proportions.items()}
        
        # adjust ideal proportions for different environments
        if self.environment_condition == "Sweet Tooth":
            self.purpose_average_proportions["Sugar"] += .15
        elif self.environment_condition == "Health Nut":
            self.purpose_average_proportions["Sugar"] -= .15
            self.purpose_average_proportions["Chunks"] += .15
        elif self.environment_condition == "Toothless":
            self.purpose_average_proportions["Chunks"] -= .15
        
        self.purpose_average_proportions = {k: (v/sum(self.purpose_average_proportions.values()))  for k, v in self.purpose_average_proportions.items()}
                                         
    def create_ingredient_occurence_dict(self, recipes: pd.DataFrame)-> None:
        """Count the occurence of each ingredient in data-set."""
        self.ingredient_occurence = dict(Counter(recipes.ingredient))
       
    def create_ingredient_association_dict(self)-> None:
        """Calculate association between ingredients."""
        for ingredient in list(self.ingredient_occurence.keys()):
            self.ingredient_association[ingredient] = {}
        for recipe in self.recipe_dict.keys():
            for ingredient in self.recipe_dict.get(recipe).get("ingredients"):
                for other_ingredient in self.recipe_dict.get(recipe):
                    if other_ingredient  == ingredient:
                        continue
                    if not other_ingredient in self.ingredient_association[ingredient].keys():
                        self.ingredient_association[ingredient][other_ingredient] = 0
                    else:
                        self.ingredient_association[ingredient][other_ingredient] += 1
    

    def cal_recipe_fitness(self, single_recipe_dict: dict)-> float:
        """Calculate fitness for a recipe based on average purpose proportions."""
        # optimise
        df = pd.DataFrame( { "Purpose": single_recipe_dict[list(single_recipe_dict.keys())[0]]['purposes'], "Quantity": single_recipe_dict[list(single_recipe_dict.keys())[0]]['quantities']})
        df = pd.DataFrame(df.groupby('Purpose')['Quantity'].sum()/df['Quantity'].sum()).reset_index()
        for purpose in self.purpose_average_proportions.keys():
            if purpose not in df["Purpose"].values:
                update_dict = {"Purpose": [purpose], "Quantity": [0]}
                df = pd.concat([df,pd.DataFrame(update_dict)], ignore_index = True)
                df.reset_index()
        if self.fitness_method == "Manhattan":
            diff_list = [abs(self.purpose_average_proportions.get(purpose) - df[df["Purpose"]==purpose]['Quantity'].values.item()) for purpose in list(df.Purpose.unique())]
        elif self.fitness_method == "Euclidean":
            diff_list = [math.sqrt(pow(self.purpose_average_proportions.get(purpose) - df[df["Purpose"]==purpose]['Quantity'].values.item(),2)) for purpose in list(df.Purpose.unique())]
            
        fitness = -sum(diff_list)
        return fitness
   
    
    def cal_pop_fitness(self, recipe_dict: dict) -> dict:
        """Calculate fitness for a dictionary of recipes."""
        population_fitness = []
        for recipe in  recipe_dict.keys():
            single_recipe_dict = {k:v for k,v in recipe_dict.items()  if k == recipe}
            population_fitness.append(self.cal_recipe_fitness(single_recipe_dict))
        return dict(zip(recipe_dict.keys(), population_fitness))
   

    def select_mating_recipes(self,population_fitness_dict:dict, num_parents: int = 2):
        """Selecting the best 10 individuals in the current generation as possible parents for
        producing the offspring of the next generation."""
        possible_parents = sorted(population_fitness_dict, key=population_fitness_dict.get, reverse=True)[:self.parent_selection_size]
        
        return random.sample(possible_parents, num_parents)


    def recipe_crossover(self, parents: list ,offspring_name: str, offspring_size:int =1)-> dict:
        """Generate offspring from parents."""
        # n length shortest parent, k length longer parent then generate m random int 0-n,
        # then take first m from shortest parent and l-m + random,  from longest parent
        
        parent_dict = {parent: self.recipe_dict[parent] for parent in parents}
        
        # caluclate length of parents ingredient lists
        smallest_parent_length = int(np.floor(min([len(parent_dict[parent].get("ingredients")) for parent in parent_dict.keys()])))
        largest_parent_length = int(np.floor(max([len(parent_dict[parent].get("ingredients")) for parent in parent_dict.keys()]))) 
        
        # random cross-over in shortest parents recipe
        crossover_point = random.randint(1,smallest_parent_length -1)
        # inherits from cross-over index to last index with a 50% chance of dropping final index.
        second_crossover_end = largest_parent_length - 1 + random.randint(-1,0) 
        
        # perform cross-over
        offspring_ingredients = parent_dict[list(parent_dict.keys())[0]].get("ingredients") [0:crossover_point] + \
        parent_dict[list(parent_dict.keys())[1]].get("ingredients") [crossover_point:second_crossover_end]
        offspring_purposes= parent_dict[list(parent_dict.keys())[0]].get("purposes") [0:crossover_point] + \
        parent_dict[list(parent_dict.keys())[1]].get("purposes") [crossover_point:second_crossover_end]
        offspring_quantities= parent_dict[list(parent_dict.keys())[0]].get("quantities") [0:crossover_point] + \
        parent_dict[list(parent_dict.keys())[1]].get("quantities") [crossover_point:second_crossover_end]
        offspring_dict = {}
        ## tidy up
        offspring_dict[offspring_name] = {}
        offspring_dict[offspring_name]["ingredients"] = offspring_ingredients
        offspring_dict[offspring_name]["purposes"] = offspring_purposes
        offspring_dict[offspring_name]["quantities"] = offspring_quantities
        
        return offspring_dict
   
    def recipe_mutation(self, offspring_dict:dict, offspring_name:str)-> None:
        """Random mutation in offspring."""
        if random.random() < self.mutation_chance:
            random_ingredient =  random.choice(list(self.ingredient_occurence.keys()))
            random_ingredient_purpose = self.ingredient_purpose_dict.get(random_ingredient)
            random_index = random.randint(0,len(offspring_dict[offspring_name].get("ingredients"))-1)
            offspring_dict[offspring_name].get("ingredients")[random_index] = random_ingredient
            offspring_dict[offspring_name].get("purposes")[random_index] = random_ingredient_purpose
        return offspring_dict
   
    def integrate_offspring(self, population_fitness_dict:dict, offspring_dict:dict,offspring_name)-> None:
        """Integrates offspring into population if fitness higher."""
       
        offspring_fitness = self.cal_recipe_fitness(offspring_dict)
        worst_recipe = sorted(population_fitness_dict, key=population_fitness_dict.get, reverse=False)[:1]
        lowest_fitness_value = population_fitness_dict[worst_recipe[0]]
       
        if (offspring_fitness > lowest_fitness_value):
            # delete crappy pop member
            del self.recipe_dict[worst_recipe[0]]
            self.recipe_dict.update(offspring_dict)
            
    
    def extinction(self, num_generations:int)-> None:
        """Integrates offspring into population if fitness higher."""
        chance_each_generation = (self.extinction_event_chance*100)/num_generations
        if random.random() < chance_each_generation:
            print(emoji.emojize("""There has been an extinction event!! :crying_face: One of the recipes
            has been lost!! """))
            random_recipe_index = random.randint(0, len(self.recipe_dict.keys()) -1)
            del self.recipe_dict[list(self.recipe_dict.keys())[random_recipe_index]]
  
   
    def pick_final_recipe(self)-> dict:
        """Picks from the final population a random recipe from the top 5."""
        final_fitness_dict = self.cal_pop_fitness(self.recipe_dict)
        final_five = self.select_mating_recipes(final_fitness_dict)
       
        random_index = random.randint(0,1)
       
        self.final_dict = {final_five[random_index]: self.recipe_dict[final_five[random_index]]}
     
    def present_recipe(self,single_recipe_dict:dict):
        """Output final recipe."""
        
        # get recipe details
        final_ingredients = single_recipe_dict.get(list(single_recipe_dict.keys())[0]).get('ingredients')
        final_ingredients = [ingredient.capitalize() for ingredient in final_ingredients]
        final_quantities =  single_recipe_dict.get(list(single_recipe_dict.keys())[0]).get('quantities')
        final_quantities = [str(quantity) + ' grams' for quantity in final_quantities]
        final_purposes = single_recipe_dict.get(list(single_recipe_dict.keys())[0]).get('purposes')
        structures, risers, binders, fats, sugars, flavours, chunks, decorators = [], [], [],[],[],[],[],[]
        for index in range(0, len(final_ingredients)):
            if final_purposes[index] == "Structure":
                structures.append(final_ingredients[index])
            elif final_purposes[index] == "Rising":
                risers.append(final_ingredients[index])
            elif final_purposes[index] == "Binding": 
                binders.append(final_ingredients[index]) 
            elif final_purposes[index] == "Fat":
                fats.append(final_ingredients[index]) 
            elif final_purposes[index] == "Sugar": 
                sugars.append(final_ingredients[index]) 
            elif final_purposes[index] == "Flavor":
                flavours.append(final_ingredients[index]) 
            elif final_purposes[index] == "Chunks":
                chunks.append(final_ingredients[index]) 
            elif final_purposes[index] == "Decorate":
                decorators.append(final_ingredients[index]) 
        
        # generate table of ingredients
        table = [["Ingredient","Amount"]]
        for index in range(0, len(final_ingredients)):
            table.append([final_ingredients[index],final_quantities[index]])
        
        # Generate name
        
        recipe_name = random.choice([keyword.capitalize() for keyword in qualifiers])
        if len(flavours) > 0:
             recipe_name+=  f" {flavours[0]}"
        if len(chunks) > 0:
             recipe_name+=  f" {chunks[0]}"
        recipe_name += " Cookies"
                
            
        # display recipe
        display(Markdown("# " + recipe_name))
        display(Markdown(f"## A {self.environment_condition} recipe"))
        display(Image(filename=f'pics/{self.environment_condition}.jpg'))
        display(Markdown("This recipe has been generated by a genetic algorithm!"))
        display(Markdown(emoji.emojize("## Shopping List  :shopping_cart:")))
        display(Markdown(make_markdown_table(table)))
        display(Markdown(emoji.emojize("### Step 1  :bowl_with_spoon:")))
        step_1_list = list(set(structures + sugars + risers))
        display(Markdown(f"Preheat the oven to 180 degrees and place the {', '.join('**'+x+'**' for x in step_1_list)} in a bowl & mix."))
        display(Markdown(emoji.emojize("### Step 2  :spoon:")))
        step_2_list = list(set(binders + fats))
        display(Markdown(f"Add in the {', '.join('**'+x+'**' for x in step_2_list)} and mix until you have an elastic dough that can be shaped in cookies."))
        display(Markdown(emoji.emojize("### Step 3  :rainbow:")))
        step_3_list = list(set(chunks + flavours))
        display(Markdown(f"Fold in {', '.join('**'+x+'**' for x in step_3_list)}."))
        display(Markdown(emoji.emojize("### Step 4  :sparkles:")))
        display(Markdown("Using your hands, roll out the cookie dough and cut into shapes of your choosing."))
        display(Markdown(emoji.emojize("### Step 5  :fire:")))
        if "Egg" in final_ingredients:
            display(Markdown(f"Cook the cookies in the oven for approximately 20 minutes."))
        else:
            display(Markdown(f"These are no-bake cookies, leave to cool in the fridge."))
        display(Markdown(emoji.emojize("### Step 6 :face_savoring_food:")))
        if len(decorators) > 0:
            display(Markdown(f"Decorate with the {', '.join('**'+x+'**' for x in decorators)}  and serve & enjoy!"))
        else:
            display(Markdown(f"Serve & enjoy!"))
        
      
        
    def plot_max_population_fitness(self, final_num_generations):
        """Plots the total fitness against generation number."""
        sns.lineplot(x= list(range(1,final_num_generations + 1)), y = self.track_max_population_fitness, color = "hotpink")
        plt.title("Max Population Fitness over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Max Recipe Fitness of Population")
        plt.show()
    
    def plot_total_population_fitness(self, final_num_generations):
        """Plots the total fitness against generation number."""
        sns.lineplot(x= list(range(1,final_num_generations + 1)), y = self.track_population_fitness, color = "hotpink")
        plt.title("Total Population Fitness over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Total Recipe Fitness of Population")
        plt.show()
        

   
    def run_genetic_algorithm(self, num_generations: int = 1000):
        """Runs the GA for a specicied number of generations."""
        # checks number of generations
        assert num_generations >= self.min_number_generations, emoji.emojize(f"Need to run for at least min_number_generations: {self.min_number_generations} generations :crying_face: !")
        
        print(emoji.emojize(f"Hello Cookie Monster :cookie: ... now running the genetic algorithm for {num_generations} generations!"))
        print(emoji.emojize('Good Luck :thumbs_up:'))
        
        # import recipe information
        recipes = self.import_data(self.file_path_recipes)
        ingredients = self.import_data(self.file_path_ingredients)
        recipes = self.convert_all_amounts_to_gram(recipes, quantity_converter_dict)
        recipes = recipes.merge(ingredients, how = 'left', left_on = recipes.ingredient, right_on = ingredients.Ingredient)
        
        
        
       
        # create required data
        self.create_recipe_dict(recipes)
        self.create_ingredient_purpose_dict(ingredients)
        
        # infer fitness benchmarks from data
        self.create_average_purpose_proportion_dict(recipes)
        self.create_ingredient_occurence_dict(recipes)
        self.create_ingredient_association_dict()

        
        for i in range(0, num_generations):
            
            # calculate population fitness
            population_fitness_dict = self.cal_pop_fitness(self.recipe_dict)
            
           
            # select parents
            parents = self.select_mating_recipes(population_fitness_dict)
           
            # offspring name
            offspring_name = self.environment_condition + "_recipe_offspring_" + str(i)
           
            # cross-over
            offspring_dict= self.recipe_crossover(parents,offspring_name)
           
            # mutation
            offspring_dict= self.recipe_mutation(offspring_dict, offspring_name)
            
            # integration
            self.integrate_offspring(population_fitness_dict, offspring_dict,offspring_name)
            
            # extinction event
            self.extinction(num_generations)
            
            # recalculate population fitness
            population_fitness_dict = self.cal_pop_fitness(self.recipe_dict)
            
            # track the max population fitness
            self.track_population_fitness.append(sum(population_fitness_dict.values()))
            self.track_max_population_fitness.append(max(population_fitness_dict.values()))
            
            # stopping criteria
            if (len(self.track_max_population_fitness) >= self.min_number_generations) & all(self.track_max_population_fitness[-1] <= rest 
                                                                  for rest in self.track_max_population_fitness[-self.stopping_criteria_runs:]):
                final_num_generations = i + 1
                print(f"Stopping after {final_num_generations} generations. The stopping criteria have been met.")
                break    # break here
            else:
                final_num_generations =num_generations
                

        
        # plots to track genetic algorithm
        self.plot_max_population_fitness(final_num_generations)
        self.plot_total_population_fitness(final_num_generations)
        
        # pick the final recipe
        self.pick_final_recipe()
        
        # present the final recipe
        self.present_recipe(self.final_dict)
           
        return 