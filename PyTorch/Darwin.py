import torch
import random
import copy
import requests #for sending updates to my phone via telegram
import numpy as np
import pandas as pd
import seaborn as sns

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import Counter
from torch.autograd import Variable
from operator import itemgetter
from tqdm import tqdm, tnrange, tqdm_notebook

from bokeh.plotting import figure 
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool
from bokeh.layouts import gridplot


def random_value(space):
    """Returns random value from space."""
    
    val = None
    
    if 'func' in space: #randomise optimiser or activation function
        val = random.sample(space['func'], 1)[0] 
    
    elif isinstance(space['lb'], int): #randomise number of units or layers
        val = random.randint(space['lb'], space['ub'])
    
    else: #randomise percentages, i.e. dropout rates or weight decay
        val = random.random() * (space['ub'] - space['lb']) + space['lb']
    
    return val


def randomize_network(layer_space, net_space): 
    """Returns a randomised neural network"""
    net = {}
    
    for key in net_space.keys():
        net[key] = random_value(net_space[key])
        
    layers = []
    
    for i in range(net['nb_layers']):
        layer = {}
        for key in layer_space.keys():
            layer[key] = random_value(layer_space[key])
        layers.append(layer)
        net['layers'] = layers
        
    return net

def mutate_net(nnet, layer_space, net_space):
    """Mutates a network hyperparameters as defined 
    in layer_space and net_space
    """
    net = copy.deepcopy(nnet)
    
    
    # mutate optimizer
    for k in ['lr', 'weight_decay', 'optimizer']:
        if random.random() < net_space[k]['mutate']:
            net[k] = random_value(net_space[k])
    
    
    # mutate layers
    for layer in net['layers']:
        for k in layer_space.keys():
            if random.random() < layer_space[k]['mutate']:
                layer[k] = random_value(layer_space[k])
                
                
    # mutate number of layers -- 50% add 50% remove
    if random.random() < net_space['nb_layers']['mutate']:
        if net['nb_layers'] <= net_space['nb_layers']['ub']:
            if random.random()< 0.5 and \
            net['nb_layers'] < net_space['nb_layers']['ub']:
                layer = {}
                for key in layer_space.keys():
                    layer[key] = random_value(layer_space[key])
                net['layers'].append(layer)      
            else:
                if net['nb_layers'] > 1:
                    net['layers'].pop()

                
            # value & id update
            net['nb_layers'] = len(net['layers'])         
            
    return net

class Flatten(nn.Module):
    """Flattens input to vector size (batchsize, 1)
    (for use in NetFromBuildInfo).
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class NetFromBuildInfo(nn.Module):
    def __init__(self, build_info):
        super(NetFromBuildInfo, self).__init__()
        
        self.activation_dict = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU()
            }

        #NETWORK DEFINITION
        
        previous_units = 28 * 28 #MNIST shape
        
        self.model = nn.Sequential()
        self.model.add_module('flatten', Flatten())
         
        for i, layer_info in enumerate(build_info['layers']):
            i = str(i)
            
            self.model.add_module(
                'fc_' + i,
                nn.Linear(previous_units, layer_info['nb_units'])
                )
            
            previous_units = layer_info['nb_units']
            
            self.model.add_module(
                'dropout_' + i,
                nn.Dropout(p=layer_info['dropout_rate'])
                )
            if layer_info['activation'] == 'linear':
                continue #linear activation is identity function
            self.model.add_module(
                layer_info['activation']+ i,
                self.activation_dict[layer_info['activation']])

        self.model.add_module(
            'logits',
            nn.Linear(previous_units, 10) #10 MNIST classes
            )
        self.model.add_module(
            'log_softmax',
            nn.LogSoftmax(dim=1)
        )
        
        
        ##OPTIMIZER

        self.opt_args = {#'params': self.model.parameters(),
                 'weight_decay': build_info['weight_decay'],
                 'lr': build_info['lr']
                 }
        
        self.optimizer_dict = {
            'adam': optim.Adam(self.model.parameters(),**self.opt_args),
            'rmsprop': optim.RMSprop(self.model.parameters(),**self.opt_args),
            'adadelta':optim.Adadelta(self.model.parameters(),**self.opt_args),
            'sgd': optim.SGD(self.model.parameters(), **self.opt_args, momentum=0.9) #momentum to train faster
            }

        self.optimizer = self.optimizer_dict[build_info['optimizer']]
        
        
    def forward(self, x):
        x = self.model(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader, optimizer, epoch):
    
    model.train(True)
    
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward() 
        optimizer.step()
        running_loss += loss.item()

    running_loss /= len(train_loader.dataset)    
    
    if epoch % 100 == 0:
        print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, running_loss ))

def test(model, test_loader, adv_func=None, adversarial=False, eps=0.5):
    """
    Test model

    Args:
    test_loader: a PyTorch dataloader to test on
    adv_func: a function that returns adversarial examples 
    """
    model.train(False)
    
    test_loss = 0
    correct = 0
    
    if adversarial:
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data= adv_func(model, data, target, eps=eps)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).cuda()).sum().item()
            test_loss += F.nll_loss(output, target, size_average=False).item()
        
    else:
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    
    return (test_loss, correct)


class TournamentOptimizer:
    """Define a tournament play selection process."""

    def __init__(self, population_sz, layer_space, net_space, init_fn, mutate_fn, builder_fn,
                 train_fn, test_fn, data_loader, test_loader, adv_func):
        
        self.init_fn = init_fn
        self.layer_space = layer_space
        self.net_space = net_space
        self.mutate_fn = mutate_fn
        self.builder_fn = builder_fn
        self.train = train_fn
        self.test = test_fn
        self.dataloader = data_loader
        self.testloader = test_loader
        self.population_sz = population_sz
        self.adv_func = adv_func
        
        torch.manual_seed(1)
        
        self.genomes = [init_fn(self.layer_space, self.net_space) for i in range(population_sz)]   
        self.population = []
        
        self.test_results = {} 
        self.genome_history = {} 

        self.generation = 0

    def step(self, generations=1, save=True, phone=False):
        """Tournament evolution step."""

        for _ in tnrange(generations, desc='Overall progress'): #tqdm progress bar

            self.generation += 1

            self.genome_history[self.generation] = self.genomes
            self.population = [NetFromBuildInfo(i).cuda() for i in self.genomes]
            self.children = []
            

            self.train_nets(save=save)
            self.evaluate_nets()

            mean = np.mean(self.test_results[self.generation]['correct'])
            best = np.max(self.test_results[self.generation]['correct'])

            tqdm.write('Generation {} Population mean:{} max:{}'
                       .format(self.generation, mean, best))
            
            if phone: #update via telegram
                requests.post("https://api.telegram.org/bot{}/"
                  "sendMessage".format(BOT_TOKEN), 
                  data={'chat_id': '{}'.format(CHANNEL),
                    'text':'Generation {} completed \n'
                        'Population mean: {} max: {}'
                        .format(self.generation, mean, best)})

                
                

            n_elite = 2
            sorted_pop = np.argsort(self.test_results[self.generation]['correct'])[::-1]
            elite = sorted_pop[:n_elite]
            
            # elites always included in the next population
            self.elite = []
            print('\nTop performers:')
            for no, i in enumerate(elite):
                self.elite.append((self.test_results[self.generation]['correct'][i], 
                                   self.population[i]))    

                self.children.append(self.genomes[i])

                tqdm.write("{}: score:{}".format(no,
                            self.test_results[self.generation]['correct'][i]))   




            #https://stackoverflow.com/questions/31933784/tournament-selection-in-genetic-algorithm
            p = 0.85 # winner probability 
            tournament_size = 3
            probs = [p*((1-p)**i) for i in range(tournament_size-1)]
            probs.append(1-np.sum(probs))
            #probs = [0.85, 0.1275, 0.0224]

            while len(self.children) < self.population_sz:
                pop = range(len(self.population))
                sel_k = random.sample(pop, k=tournament_size)
                fitness_k = list(np.array(self.test_results[self.generation]['correct'])[sel_k])
                selected = zip(sel_k, fitness_k)
                rank = sorted(selected, key=itemgetter(1), reverse=True)
                pick = np.random.choice(tournament_size, size=1, p=probs)[0]
                best = rank[pick][0]
                genome = self.mutate_fn(self.genomes[best], self.layer_space, self.net_space)
                self.children.append(genome)
                
            self.genomes = self.children
                

        
        
    def train_nets(self, save=True):
        """trains population of nets"""
         
        for i, net in enumerate(tqdm_notebook(self.population)):
            for epoch in range(1, 2):
                torch.manual_seed(1)
                self.train(net, self.dataloader, net.optimizer, epoch)
                
            if save:
                fp = r"D:\Models\NeuroEvolution/{}-{}".format(self.generation, i)
                torch.save(net.state_dict(), fp)
                
                
    def evaluate_nets(self):
        """evaluate the models."""
        
        losses = []
        corrects = []
        clean_corrects = []
        
        self.test_results[self.generation] = {}
        
        for i in range(len(self.population)):
            net = self.population[i]
            loss, correct = self.test(net, self.testloader, adv_func=self.adv_func,
                                    adversarial=True, eps=0.5) 
            _, clean_correct = self.test(net, self.testloader)
            
            losses.append(loss)
            corrects.append(correct)
            clean_corrects.append(clean_correct)
        
        self.test_results[self.generation]['losses'] = losses
        self.test_results[self.generation]['correct'] = corrects
        self.test_results[self.generation]['clean_correct'] = clean_corrects


def progressplotter(optimizer, clean=False):
    
    if clean:
        dataset = 'clean_correct'
    else:
        dataset = 'correct'
    
    means = []
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    
    gens = range(len(optimizer.test_results))
    popsize = len(optimizer.test_results[1][dataset])
    
    for i in gens:
        ax.scatter([i for j in range(popsize)], optimizer.test_results[i+1][dataset])
        mean = np.mean(optimizer.test_results[i+1][dataset])
        means.append(mean)
        ax.scatter(i, mean, c=1)
        
        if i == 0:
            continue
        plt.plot([i-1, i], [means[i-1], mean], c='black')
        
    ax.set_xticks(np.arange(0, len(means),1))
    ax.set_xlabel('Generation')
    ax.set_ylabel('Correct classifications')
    
    if clean:
        ax.set_title('Accuracy on clean dataset')
    else:
        ax.set_title('Accuracy on adversarial dataset')
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(30)
        
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

def diffplotter(optimizer):
    diff = {}
    for gen in optimizer.test_results:
        diff[gen] = []
        for i in range(len(optimizer.test_results[gen]['clean_correct'])):
            clean = optimizer.test_results[gen]['clean_correct'][i]
            adver = optimizer.test_results[gen]['correct'][i]
            diff[gen].append(clean - adver)
            
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)

    gens = len(optimizer.test_results)
    popsize = len(optimizer.test_results[gen]['clean_correct'])

    for i in range(gens):
        ax.scatter([i for j in range(popsize)], diff[i+1])
        
    ax.set_title('Difference between clean and adversarial accuracy')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Clean accuracy - adversarial accuracy')
    
    ax.set_xticks(np.arange(0, gens,1))
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(30)
        
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

def bestplotter(optimizer, k=0):

    holder = {
        'clean accuracy' : [],
        'adversarial accuracy' : [],
        'number of layers' : [],
        'activation function' : [],
        'dropout rate' : [],
        'optimizer' : [],
        'number of units in layer' : [],
        'learning rate' : [],
    }

    for gen in optimizer.test_results:

        curr = optimizer.test_results[gen]

        best_index = np.argsort(curr['correct'])[::-1][k]

        holder['clean accuracy'].append(curr['clean_correct'][best_index])
        holder['adversarial accuracy'].append(curr['correct'][best_index])

        genome = optimizer.genome_history[gen][best_index]

        holder['number of layers'].append(genome['nb_layers'])
        holder['dropout rate'].append(genome['layers'][0]['dropout_rate'])
        holder['number of units in layer'].append(genome['layers'][0]['nb_units'])
        holder['optimizer'].append(genome['optimizer'])
        holder['activation function'].append(genome['layers'][0]['activation'])
        holder['learning rate'].append(genome['lr'])
        
    gens = len(holder['clean accuracy'])
    
    fig = plt.figure(figsize=(20,20))
    
    for i in range(8):
        ax = fig.add_subplot(4,2, i+1)
        ax.set_ylabel(list(holder.keys())[i])
        ax.set_xlabel('Generation')
        ax.set_xticks(np.arange(0, gens,5))
        ax.yaxis.label.set_fontsize(15)
        
        
        
        for j in range(gens):
            ax.scatter(j, holder[list(holder.keys())[i]][j], c='black')


def avgplotter(optimizer):

    holder = {
        'clean accuracy' : [],
        'adversarial accuracy' : [],
        'number of layers' : [],
        'activation function' : [],
        'dropout rate' : [],
        'optimizer' : [],
        'number of units in layer' : [],
        'learning rate' : [],
    }

    for gen in optimizer.test_results:

        curr = optimizer.test_results[gen]
        genomes = optimizer.genome_history[gen]
        
        templist1 = []
        templist2 = []
        templist3 = []
        templist4 = []
        templist5 = []
        templist6 = []
        for net in genomes:
            templist1.append(net['nb_layers'])
            templist2.append(net['optimizer'])
            templist3.append(net['lr'])
            templist4.append(net['layers'][0]['dropout_rate'])
            templist5.append(net['layers'][0]['nb_units'])
            templist6.append(net['layers'][0]['activation'])
            
            
        holder['number of layers'].append(np.mean(templist1))
        holder['optimizer'].append(Counter(templist2).most_common()[0][0])
        holder['learning rate'].append(np.mean(templist3))
        holder['dropout rate'].append(np.mean(templist4))
        holder['number of units in layer'].append(np.mean(templist5))
        holder['activation function'].append(Counter(templist6).most_common()[0][0])
        

        holder['clean accuracy'].append(np.mean(curr['clean_correct']))
        holder['adversarial accuracy'].append(np.mean(curr['correct']))
            

    gens = len(holder['clean accuracy'])
    
    fig = plt.figure(figsize=(20,20))
    
    for i in range(8):
        ax = fig.add_subplot(4,2, i+1)
        ax.set_ylabel(list(holder.keys())[i])
        ax.set_xlabel('Generation')
        ax.set_xticks(np.arange(0, gens,5))
        ax.yaxis.label.set_fontsize(15)
        
        for j in range(gens):
            ax.scatter(j, holder[list(holder.keys())[i]][j], c='black')


def rebuild_from_save(optimizer, generation, position):
    
    genome = optimizer.genome_history[generation][position]
    
    net = NetFromBuildInfo(genome)
    
    net.load_state_dict(torch.load(r"D:\Models\NeuroEvolution\{}-{}".format(generation, position)))
    
    return net.cuda()

def sanity_check(optimizer, test_loader):
    
    for generation in optimizer.test_results:
        print('generation {}: \n'.format(generation))
        for i, result in enumerate(optimizer.test_results[generation]['correct']):
            
            mod = rebuild_from_save(optimizer, generation, i)
            _, rebuild_result = test(mod, test_loader, adversarial=True, eps=0.5)
            
            if result == rebuild_result:
                print("result = {}, rebuild result = {}. (equal)".format(result, rebuild_result))
            else:
                print("result = {}, rebuild result = {}. (different!!)".format(result, rebuild_result))


def get_best_model(optimizer, adversarial=True):
    current_best = 0
    for i, gen in enumerate(optimizer.test_results):
        if adversarial: 
            generation_correct = optimizer.test_results[gen]['correct']
        else:
            generation_correct = optimizer.test_results[gen]['clean_correct']
        for j, score in enumerate(generation_correct):
            if score > current_best:
                best_gen = gen
                best_pos = j
                current_best = score
    clean_score = optimizer.test_results[best_gen]['clean_correct'][best_pos]
    adv_score = optimizer.test_results[best_gen]['correct'][best_pos]
                
    return [best_gen, clean_score, adv_score, rebuild_from_save(optimizer, best_gen, best_pos)]

def best_printer(optimizer):
    holdict = {}
    holdict['best_clean'] = {}
    holdict['best_adversarial'] = {}
    
    holdict['best_clean']['generation'] , \
    holdict['best_clean']['clean'], \
    holdict['best_clean']['adversarial'], _ = get_best_model(optimizer, adversarial=False)
    
    holdict['best_adversarial']['generation'] , \
    holdict['best_adversarial']['clean'], \
    holdict['best_adversarial']['adversarial'], _ = get_best_model(optimizer, adversarial=True)
    
    return pd.DataFrame(holdict).T

def multi_plot(optimizer, data_loader, adv_func=None, adversarial=False, eps=0.5):
    
    best_gen, best_clean_score, best_adv_score, best_model = get_best_model(optimizer)
    batch = next(iter(data_loader))
    
    print("Showing best model which was found in generation {}\n"
          "Clean accuracy = {}\nadversarial accuracy ={}\n\n"
         "Model: \n\n".format(best_gen, best_clean_score,
                           best_adv_score), best_model, "\n\n",
          "Images below are {}"
          .format('adversarial' if adversarial else 'clean'))

    fig = plt.figure(figsize=(20,20))

    counter=0
    for i in range(len(batch[1])):
        if batch[1][i].item() == counter: #find first digit that is the correct class for plot position (0-9)
            counter+=1
            if counter == 10: break
            ax = fig.add_subplot(3,3, counter)
            if adversarial:
                image = adv_func(best_model, batch[0][i].view(1,1,28,28).cuda(),
                                   batch[1][i].view(1), eps=eps)      
            else:
                image = batch[0][i].cuda()
            softmax = np.exp(best_model(image.view(1,1,28,28)).detach().cpu().numpy())
            prediction = softmax.argmax()
            prediction_pct = softmax.max()
            ax.imshow(image.detach().cpu().numpy().reshape(28,28), cmap='Greys')
            ax.text(x=3, y=31, s="Predicted: {x} ({y:.2f}%)"
                     .format(x=prediction, y=100 *prediction_pct), fontsize=20)
            image=0


def dataframer(optimizer):
    
    number_of_layers = []
    learning_rate = []
    act_func = []
    number_of_units_1 = []
    dropout_rate = []
    genome_hist = []
    generations = []
    clean_scores = []
    adv_scores = []
    
    for generation in optimizer.test_results:

        scores = optimizer.test_results[generation]
        genomes = optimizer.genome_history[generation]

        clean_scores += scores['clean_correct']
        adv_scores += scores['correct']

        for genome in genomes:
            
            generations.append(generation)
            genome_hist.append(genome)
            number_of_layers.append(genome['nb_layers'])
            learning_rate.append(genome['lr'])

            act_func.append(genome['layers'][0]['activation'])
            number_of_units_1.append(genome['layers'][0]['nb_units'])
            dropout_rate.append(genome['layers'][0]['dropout_rate'])

            df = pd.DataFrame([generations, clean_scores, adv_scores, number_of_layers,
                             learning_rate, act_func, 
                             number_of_units_1, dropout_rate, genome_hist]).T

            df.columns = ['Generation', 'Clean','Adversarial','No_layers',
                          'Lr', 'Act_func', 'Nb_units', 'Dropout', 'Genome']
    
    return df


def int_plot(df, x, y, x2, y2, gen):
    if gen == 'all':
        source = ColumnDataSource(df.iloc[:, :-1])#last column contains dicts which causes bokeh to fail
    else: 
        source = ColumnDataSource(df[df['Generation']==gen].iloc[:, :-1]) 
    
    tiplist = [("Accuracy", "@Clean"), 
            ("Adversarial accuracy", "@Adversarial"),
            ("Number of layers", "@No_layers"),
            ("Generation", "@Generation"),
            ("Activation function", "@Act_func"),
              ("Dropout", "@Dropout")]
    
    options = dict(plot_width=400, plot_height=400,
                   tools="pan,wheel_zoom,box_zoom,box_select,lasso_select",
                  active_scroll= 'wheel_zoom')

    p1 = figure(title="{} vs {}".format(y, x), **options)
    p1.scatter(x, y, color="blue", source=source,
               hover_line_color="black")#, radius=0.1)
    p1.xaxis.axis_label = x
    p1.yaxis.axis_label = y
    if y in ['Adversarial', 'Clean']:
        p1.y_range.start = -1000
        p1.y_range.end = 11000
    p1.add_tools(HoverTool(tooltips=tiplist))

    p2 = figure(title="{} vs {}".format(y2, x2), **options)
    p2.scatter(x2, y2, color="green", source=source, 
               hover_line_color="black")#, radius=0.1)
    p2.xaxis.axis_label = x2
    p2.yaxis.axis_label = y2
    if y2 in ['Adversarial', 'Clean']:
        p2.y_range.start = -1000
        p2.y_range.end = 11000
    p2.add_tools(HoverTool(tooltips=tiplist))

    p = gridplot([[ p1, p2]], toolbar_location="left")

    show(p)