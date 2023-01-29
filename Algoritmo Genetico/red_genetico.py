import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#------------------Activation Functions----------------
#Radial
def radbas(x):
    a=np.exp(-x**2)
    return a

#-------------------NN arquitectures-------------------
#1st layer:RadBas && 2nd layer: Perceptron
def nn(Input,w1,w2,b1,b2):
    Output=np.zeros(len(Input))
    for i in range(len(Input)):
        dist=np.sqrt((w1-Input[i])**2)
        out1=radbas(dist*b1)
        Output[i]=np.dot(w2,out1)+b2
    return Output

#1st layer:Perceptron && 2nd layer: Perceptron
def nn1(w1,w2,b1,b2):
    Output=np.zeros(len(Input))
    for i in range(len(Input)):
        out1=w1*Input[i]+b1
        out1[out1<0]=0
        Output[i]=np.dot(w2,out1)+b2
    return Output

#-----------------Function to optimizate-----------------------
def mse(weights,Input, Target,s1,s2,R):
    MSE=np.zeros(weights.shape[0])
    for j in range(weights.shape[0]):
        w1=(weights[j,0:s1*R]).T
        b1=(weights[j,s1*R:2*s1*R]).T
        w2=weights[j,2*s1*R:(2*s1*R)+(s1*s2)]
        b2=weights[j,(2*s1*R)+(s1*s2):(2*s1*R)+(s1*s2)+s2]
        Output=nn(Input,w1,w2,b1,b2)
        MSE[j] = mean_squared_error(Output, Target)
    return MSE

#-------------------Fitness function----------------------------
def Fitness(x):
    fitness=1/x;
    return fitness

#--------------------Random choice function----------------------
def roulette(probability):
    r=np.random.rand(); pos=1; proac=0;
    for k in range(len(probability)):
        proac=proac+probability[k]
        if proac>r:
            pos=k
            break
    return pos

#--------------------Crossing function----------------------
def crossing_(parent1,parent2):
    son1=np.zeros(len(parent1))
    son2=np.zeros(len(parent2))
    for k in range(len(parent1)):
        a=np.min([parent1[k],parent2[k]])-2;
        b=np.max([parent1[k],parent2[k]])+2;
        son1[k]=(b-a)*np.random.rand()+a;
        son2[k]=(b-a)*np.random.rand()+a;
    return son1,son2

def crossing_1(parent1,parent2):
    son1=np.zeros(len(parent1))
    son2=np.zeros(len(parent1))
    r=((len(parent1)-3)*np.floor(np.random.rand()))+2
    print('Rand',r)
    son1[0:r]=parent1[0:r]
    son1[r:len(parent2)]=parent2[r:len(parent2)]
    son2[0:r]=parent2[0:r]
    son2[r:len(parent2)]=parent1[r:len(parent1)]
    return son1,son2

#--------------------Mutation function----------------------
def mutation(son,probmut,lim1,lim2):
    sonmutated=son;
    for k in range(len(son)):
        r=np.random.rand()
        if r<probmut:
            if k>=2*s1*R:
                sonmutated[k]=(lim2*2*(np.random.rand()))-lim2;
            else:
                sonmutated[k]=(lim1*2*(np.random.rand()))-lim1;
    return sonmutated

def genetic_algorithm(s1,s2,R,population,generations,Input, Target,probmut,lim1,lim2):
    n=(2*s1*R)+(s1*s2)+s2
    weights =np.zeros([population,n])
    weights[:,0:(2*s1*R)] = (lim1*2*np.random.rand(population,(2*s1*R)))-(1*lim1)
    weights[:,(2*s1*R):(2*s1*R)+(s1*s2)+s2] = (lim2*2*np.random.rand(population,((s1*s2)+s2)))-1*lim2
    for generation in range(generations):
        fitness=Fitness(mse(weights,Input, Target,s1,s2,R))
        probability=fitness/sum(fitness)
        parents=np.zeros([population,n])
        
        #Parents selection
        for i in range(population):
            pos=roulette(probability)
            parents[i,:]=weights[pos,:]
        
        #Crossing
        sons=np.zeros([population,n])
        for j in range(0,population,2):
            parent1=parents[j,:]
            parent2=parents[j+1,:]
            son1,son2=crossing_(parent1,parent2)
            sonmutated1=mutation(son1,probmut,lim1,lim2)
            sonmutated2=mutation(son2,probmut,lim1,lim2)
            sons[j,:]=sonmutated1
            sons[j+1,:]=sonmutated2

        #Selection    
        fitness_final=np.zeros(2*population)
        all_weights=np.zeros([2*population,n])
        all_weights[0:population,:]=sons
        all_weights[population:2*population,:]=parents
        fitness_final=Fitness(mse(all_weights,Input, Target,s1,s2,R))
        organized=np.flip(np.argsort(fitness_final))
        for k in range(population):
            weights[k]=all_weights[organized[k],:]
    fitness=Fitness(mse(weights,Input, Target,s1,s2,R))
    final_weight=weights[np.argmax(fitness),:]
    return final_weight
   
#---------------------Algorithm test --------------------------------------------
"""
#Sin -10 10
Input=np.linspace(-10.0, 10.0, num=100)
Target=np.sin(Input);lim1=10;lim2=1
population=70
R=1;s1=6;s2=1;
generations=500

#Sin -5 5
Input=np.linspace(-5.0, 5.0, num=100)
Target=2*np.cos(Input)+np.sin(3*Input)+5;lim1=5;lim2=8
population=50;
R=1;s1=10;s2=1;
generations=200
"""
#x**2/100
Input=np.linspace(-10.0, 10.0, num=100)
Target=(Input**2)/100;lim1=10;lim2=10
population=50;
R=1;s1=1;s2=1;
generations=200

probmut=np.random.rand()
weight=genetic_algorithm(s1,s2,R,population,generations,Input, Target,probmut,lim1,lim2)
w1=(weight[0:s1*R]).T
b1=(weight[s1*R:2*s1*R]).T
w2=weight[2*s1*R:(2*s1*R)+(s1*s2)]
b2=weight[(2*s1*R)+(s1*s2):(2*s1*R)+(s1*s2)+s2]
print(w1)
print(b1)
print(w2)
print(b2)
Output=nn(Input,w1,w2,b1,b2)
plt.plot(Input,Target)
plt.plot(Input,Output)
plt.show()
