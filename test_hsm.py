import theano
import theano.tensor as T
import random
import numpy as np
import hierarchical_softmax
import time
label_count=2000
example_size=500

def create_data(label_count, example_size):
    the_examples_and_labels = []
    for l in range(label_count):
        base_example = []
        for i in range(example_size):
            base_example.append(random.uniform(-1.0,1.0))
        for i in range(100): # batch_size
            the_examples_and_labels.append((np.array(corrupt(base_example)), l))

    return the_examples_and_labels

def corrupt(vector):

    new_vector = []
    for v in vector:
        new_vector.append(v + random.uniform(-0.05, 0.05))
    return new_vector

def load_data():

    data = create_data(label_count, example_size)

    random.shuffle(data)
    mini_example=[]
    mini_labels=[]
    for i in range(0,len(data),100):
        examples = []
        labels = []
        for b in range(i, i+100):
            examples.append(data[b][0])
            labels.append(data[b][1])
        mini_example.append(np.asarray(examples,dtype=theano.config.floatX))
        mini_labels.append(np.asarray(labels,dtype=np.int))
    return np.asarray(mini_example),np.asarray(mini_labels)

def hsm(examples,labels):


    #Hierarchical softmax test
    tree = hierarchical_softmax.build_binary_tree(range(label_count))
    hs = hierarchical_softmax.HierarchicalSoftmax(tree,example_size,label_count)

    train_f = hs.get_training_function()

    hc = []
    now=time.time()
    for i in range(10):
        for ex,lab in zip(examples,labels):
            hc.append(train_f(ex, lab))
        #print np.mean(hc), i
    print "training time:", time.time()-now

    #pf = hs.get_probability_function()
    #print 'Minibatch probabilities after training'
    #print T.exp(pf(examples[0], labels[0])[0]).eval()
'''
    print 'Predictions'
    red = hs.label_tool(minibatches[0][0])
    for o, r in zip(minibatches[0][1],red[1][-1]):
        print o, r
'''
def normal_softmax(examples,labels):
    print ("Normal softmax")
    #Logistic regression with normal softmax-test
    input = T.fmatrix('inputs')
    y = T.ivector('labels')
    rng = np.random.RandomState(1234)

    learning_rate = 0.5
    W = theano.shared(value=np.asarray(rng.uniform(
        low=-np.sqrt(6. / 50),
        high=np.sqrt(6. / 50),size=(example_size, label_count)),
        dtype=theano.config.floatX), name = 'W_soft', borrow = True)

    b = theano.shared(value=np.zeros(label_count))

    p_y_given_x = T.nnet.softmax(T.dot(input, W) + b)
    cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x,y))
    params = [W, b]

    gparams = T.grad(cost,params)
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]
    train_f = theano.function(inputs=[input, y],
                              outputs=[cost],
                              updates=updates)

    #debug_f=theano.function(inputs=[input,y],outputs=cost)


    hc = []
    now=time.time()
    for i in range(10):
        for ex,lab in zip(examples,labels):
            result=train_f(ex,lab)
            hc.append(result)
        #print np.mean(hc), i
    print "training time:", time.time()-now

if __name__=="__main__":
    examples,labels=load_data()
    normal_softmax(examples,labels)
    #hsm(examples,labels)
