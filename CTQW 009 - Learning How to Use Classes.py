#Learning how to use Classes

#A Class, like any object, can be described by sets of properties that it may have
#For instance, if we want to describe a person we'll want to know their age, height, name, etc

#So when we want to define an object in a class we'll want to think of all the necessary properties that can define our object
#Therefore, the first step is to consider "class" and then the objects in the class,
#For this example we'll use "Person"
class Person:
    #Now we want to be specific with the following code as this is how we define the characteristics of our objects
    #We first write "def __inti__(self , ...):"
    #At the elipsis, we then list all the properties about our objects that we deem essential to define a class
    #So for us to define our class of people, we will list the following properties:
    #age , weight , height , first_name , last_name
    def __init__(self , age , weight , height , first_name , last_name):
        #Then, to ensure that the objects in the class can be characterised by each property
        #We must write the list of characteristics as follows
        self.age = age
        self.weight = weight
        self.height = height
        self.first_name = first_name
        self.last_name = last_name
#Now we've successfully designed a class!!!

#Consider the following example where we define a user with the following certain characteristics
user = Person(24 , 71 , 168 , "Conall" , "Campbell")
#Here our user is age 24, weight 71, height 168, and is named Conall Campbell
#To then recall certain properties about this member of our class (we want to know more about user)
#Then we must recall "user.property" in the terminal
#This will print the information about our user, which we can therefore use to change in whatever way we see fit




#How does this link with the Genetic Algorithm for the CTQW problem?



#For my genetic algorithm I intend to make an individual class for networks so they're predeterminantly built
#therefore I can simply recall something like ".get_sink_output"
#which will allow me to just throw whatever parameters I'd like into my class and obtain the outputs of my sinks

#I want to have the following classes to perform the following operations:
    # - Evaluator Class: takes population and target network + gives an array of fidelities     ->    get_fidelity_output
    # - Evolution Class: pass in array of networks + fidelities wrt target network and considers
                         #which elements are kept + which elements are to be killed off (i.e. which parents survive)    ->    get_evolution_output
    # - Mating Class: Produces offspring + mutates them, returning the new population       ->      get_mating_output

#Remember the aim of the algorithm is to determine the topology of an unknown quantum network just through the measurements made by the sink
#I first want to confirm that there is a direct link between these two statements, i.e.,
#if I have a population of complex networks and a known complex network, and I run my GA on my population
#does having a high fidelity between my sink populations CORRESPOND to me being able to directly reconstructing the topology of the known network
#from the sink measurements alone?