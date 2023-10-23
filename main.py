import numpy as np
import matplotlib.pyplot as plt
import random
import math

f = open("data.txt", 'r')

DIMENSION = int(f.readline())
SIZE = int(f.readline())
LEFT_EDGE = float(f.readline())
RIGHT_EDGE = float(f.readline())

ALPHA = 0.00001
LAMBDA = 0.0
BATCH_SIZE = 1
EPOCHS = 500
ADD_ONES = 0

TREE_DEPTH = 9
QUALITY = 0.999

FOREST_SIZE = 7


colors = ["blue", "red", "green", "purple", "yellow", "pink", "black"]


dataset = [[[x[i] for i in range(DIMENSION + ADD_ONES)], x[-1]] for x in [[1]*ADD_ONES + [float(t) for t in str(f.readline()).split()] for i in range(SIZE)]]
ANSWERS = []
ans_set = set(ANSWERS)
for s in dataset: 
    if not(s[1] in ans_set):
        ans_set.add(s[1])
        ANSWERS += [s[1]]    

def data_reg_plot(dataset):
    for s in dataset:
            plt.scatter(s[0][ADD_ONES], s[1], 1, "black")

def data_class_plot(dataset):
    for s in dataset:
        plt.scatter(s[0][0], s[0][1], 1, colors[ANSWERS.index(int(s[1]))])

#Linear regression
def learn_lin_reg(dataset):
    w = np.array([random.random() for i in range(DIMENSION + 1)])
    for e in range(EPOCHS):
        random.shuffle(dataset)
        if(1):
            for b in range(SIZE // BATCH_SIZE):
                X_BATCH = np.array([ dataset[b*BATCH_SIZE + i][0] for i in range(BATCH_SIZE)])
                Y_BATCH = np.array([ dataset[b*BATCH_SIZE + i][1] for i in range(BATCH_SIZE)])
                
                GRAD = np.full((DIMENSION + 1), 0.0)
                f = X_BATCH @ w
                err = f - Y_BATCH
                GRAD = X_BATCH.T @ err / BATCH_SIZE
                #GRAD = 2 * X_BATCH.T.dot(X_BATCH.dot(w) - Y_BATCH)/BATCH_SIZE
                
                w -= ALPHA*GRAD + 2*ALPHA*LAMBDA*np.array([0, w[1]])
        if(0):
            X = np.array([dataset[i][0] for i in range(SIZE)])
            Y = np.array([dataset[i][1] for i in range(SIZE)])
            
            GRAD = 2*X.T @ (X @ w - Y) / SIZE
            
            w -= ALPHA * GRAD
    if(1):
        plot_data_reg(dataset)
        plt.plot([LEFT_EDGE, RIGHT_EDGE], [w[0] + w[1]*LEFT_EDGE, w[0] + w[1]*RIGHT_EDGE])
        plt.show()
    return w
    
#Linear classification
def learn_lin_class(dataset):
    w = np.array([random.random() for i in range(DIMENSION + 1)])
    for e in range(EPOCHS):
        random.shuffle(dataset)
        if(0):
            for b in range(SIZE // BATCH_SIZE):
                X_BATCH = np.array([ dataset[b*BATCH_SIZE + i][0] for i in range(BATCH_SIZE)])
                Y_BATCH = np.array([ dataset[b*BATCH_SIZE + i][1] for i in range(BATCH_SIZE)])
                
                
                GRAD = np.full((DIMENSION + 1), 0.0)
                f = X_BATCH @ w
                dist = Y_BATCH * f
                GRAD = (-1 if dist < 1 else -X_BATCH.T @ Y_BATCH) / BATCH_SIZE
                #GRAD = 2 * X_BATCH.T.dot(X_BATCH.dot(w) - Y_BATCH)/BATCH_SIZE
                
                w -= ALPHA*GRAD + 2*ALPHA*LAMBDA*np.array([0, w[1], w[2]])
        if(1):
            for i in range(SIZE):
                X = np.array(dataset[i][0])
                Y = dataset[i][1]
                
                GRAD = np.full((DIMENSION + 1), 0.0)
                f = X @ w
                dist = Y * f
                GRAD = (-Y*X if dist < 1 else 0) / SIZE
                
                w -= ALPHA * GRAD
    if(1):
        for s in dataset:
            plt.scatter(s[0][1], s[0][2], 1, "blue"*int((s[1]+1)/2) + "red"*int((1 - s[1])/2))
        plt.plot([LEFT_EDGE, RIGHT_EDGE], [(-w[0] - w[1]*LEFT_EDGE)/w[2], -(w[0] + w[1]*RIGHT_EDGE)/w[2]])
        plt.show()
    return w

#Logistic regression (not yet)
def learn_log_reg(dataset):
    w = np.array([random.random() for i in range(DIMENSION + 1)])
    for e in range(EPOCHS):
        random.shuffle(dataset)
        if(0):
            for b in range(SIZE // BATCH_SIZE):
                X_BATCH = np.array([ dataset[b*BATCH_SIZE + i][0] for i in range(BATCH_SIZE)])
                Y_BATCH = np.array([ dataset[b*BATCH_SIZE + i][1] for i in range(BATCH_SIZE)])
                
                
                GRAD = np.full((DIMENSION + 1), 0.0)
                f = X_BATCH @ w
                dist = Y_BATCH * f
                GRAD = (-1 if dist < 1 else -X_BATCH.T @ Y_BATCH) / BATCH_SIZE
                #GRAD = 2 * X_BATCH.T.dot(X_BATCH.dot(w) - Y_BATCH)/BATCH_SIZE
                
                w -= ALPHA*GRAD + 2*ALPHA*LAMBDA*np.array([0, w[1], w[2]])
        if(1):
            for i in range(SIZE):
                X = np.array(dataset[i][0])
                Y = dataset[i][1]
                
                GRAD = np.full((DIMENSION + 1), 0.0)
                f = X @ w
                dist = Y * f
                GRAD = (-Y*X if dist < 1 else 0) / SIZE
                
                w -= ALPHA * GRAD
    if(1):
        for s in dataset:
            plt.scatter(s[0][1], s[0][2], 1, "blue"*int((s[1]+1)/2) + "red"*int((1 - s[1])/2))
        plt.plot([LEFT_EDGE, RIGHT_EDGE], [(-w[0] - w[1]*LEFT_EDGE)/w[2], -(w[0] + w[1]*RIGHT_EDGE)/w[2]])
        plt.show()
    return w

#Decision tree
class tree_node:
    def __init__(self, i, const, isl, ans = -1):
        self.index = i
        self.constant = const
        self.isleaf = isl
        self.ans = ans

def impurity(els):
    if els == []:
        return 0
    avg = 0
    for el in els:
        avg += el[1]
    avg /= len(els)
    h = 0
    for el in els:
        h += (el[1] - avg)*(el[1] - avg)
    h /= len(els)
    return h

def branch_est(els, j, t):
    els_l = []
    els_r = []
    for el in els:
        if el[0][j] < t:
            els_r += [el]
        else: 
            els_l += [el]
    return len(els)*impurity(els) - len(els_l)*impurity(els_l) - len(els_r)*impurity(els_r)

def construct_tree(tree, dataset, els, n = 1, lvl = 1):
    answers_local = []
    ans_set_local = set(answers_local)
    for s in dataset: 
        if not(s[1] in ans_set_local):
            ans_set_local.add(s[1])
            answers_local += [s[1]]    
    
    class_count = [0 for i in range(len(answers_local))]
    i_class_max = 0
    for el in els:
        for i in range(len(answers_local)):
            if el[1] == answers_local[i]:
                class_count[i] += 1
                if class_count[i] > class_count[i_class_max]:
                    i_class_max = i
                break
    if lvl == TREE_DEPTH or class_count[i_class_max] > len(els)*QUALITY:
        tree[n] = tree_node(0, 1, 1, answers_local[i_class_max])
        return
    
    branch_est_max = -1
    i_est_max = -1
    t_est_max = 0
    
    for i in range(DIMENSION):
        for el in els:
            est = branch_est(els, i, el[0][i])
            if est > branch_est_max:
                branch_est_max = est
                t_est_max = el[0][i]
                i_est_max = i
    
    tree[n] = tree_node(i_est_max, t_est_max, 0)
    els_r = []
    els_l = []
    for el in els:
        if el[0][i_est_max] < t_est_max:
            els_r += [el]
        else:
            els_l += [el]
    construct_tree(tree, dataset, els_l, 2*n, lvl + 1)
    construct_tree(tree, dataset, els_r, 2*n + 1, lvl + 1)
    return tree

def tree_class_plot(tree, dataset, n = 1, x1_min = LEFT_EDGE, x1_max = RIGHT_EDGE, x2_min = LEFT_EDGE, x2_max = RIGHT_EDGE):
    answers_local = []
    ans_set_local = set(answers_local)
    for s in dataset: 
        if not(s[1] in ans_set_local):
            ans_set_local.add(s[1])
            answers_local += [s[1]]      
    if(tree[n].isleaf):
        x = np.arange(x1_min, x1_max, 0.01)
        y1 = []
        y2 = []
        for t in x:
            y1 += [x2_max]
            y2 += [x2_min]
        plt.fill_between(x, y1, y2, color = colors[answers_local.index(int(tree[n].ans))], alpha = 0.2)
        return
    else:
        if tree[n].index == 0:
            tree_class_plot(tree, 2*n, tree[n].constant, x1_max, x2_min, x2_max)
            tree_class_plot(tree, 2*n + 1, x1_min, tree[n].constant, x2_min, x2_max)
        else:
            tree_class_plot(tree, 2*n, x1_min, x1_max, tree[n].constant, x2_max)
            tree_class_plot(tree, 2*n + 1, x1_min, x1_max, x2_min, tree[n].constant)

def tree_decision(tree, x):
    if tree[1] == 0: return 0
    n = 1
    while (tree[n].isleaf != 1):
        if (x[tree[n].index] < tree[n].constant):
            n = 2*n + 1
        else:
            n = 2*n
    return tree[n].ans

def tree_reg_plot(tree):
    x = np.arange(LEFT_EDGE, RIGHT_EDGE, 0.01)
    y = [tree_decision(tree, [x0]) for x0 in x]
    plt.plot(x, y)

def tree_out(tree, i = 1, lvl = 1, str = ""):
        if tree[i] == 0: return
        if tree[i].isleaf == 0:
            print("    " * (lvl - 1), str + " Node: x", tree[i].index, " < ", tree[i].constant, sep="")
            tree_out(tree, 2*i, lvl + 1, "No ->")
            tree_out(tree, 2*i + 1, lvl + 1, "Yes ->")
        else:
            print("    " * (lvl - 1), str + " Leaf: Answer =", tree[i].ans)
            
#GBDT

def compute_gbdt(forest, x, pr = 0, lim = FOREST_SIZE):
    ans = []
    for i in range(lim):
        ans += [tree_decision(forest[i], x)]
    if(pr): print(x, ans, sum(ans))
    return sum(ans)
    
def learn_gbdt(dataset):
    forest = []
    X = [s[0] for s in dataset]
    Y = [s[1] for s in dataset]
    cur_Y = [s[1] for s in dataset]

    for k in range(FOREST_SIZE):
        gbdt_reg_plot(forest, k)
        plot_data_reg(dataset)
        plt.show() 
        cur_tree = [0]*(2**TREE_DEPTH)
        construct_tree(cur_tree,  [[X[i], cur_Y[i]] for i in range(SIZE)],  [[X[i], cur_Y[i]] for i in range(SIZE)])
        forest += [cur_tree]

        cur_Y = [Y[i] - compute_gbdt(forest, X[i], 0, k + 1) for i in range(SIZE)]
        print("Tree №", k)
        tree_out(cur_tree)
        for i in range(SIZE): 
            if abs(cur_Y[i]) > 1000:
                print("ПИЗДЕЦ БЛЯТЬ", X[i], Y[i], cur_Y[i], tree_decision(forest[0], X[i]), Y[i] - tree_decision(forest[0], X[i]))
    return forest

def gbdt_reg_plot(forest, lim = FOREST_SIZE):
    x = np.arange(LEFT_EDGE, RIGHT_EDGE, 0.01)
    y = [compute_gbdt(forest, [x0], 0, lim) for x0 in x ]
    plt.plot(x, y)
    
#Neural network

def ELU(x):
    t = np.ones((x.size, 1))
    for i in range(x.size):
        if x[i][0] < 0:
            t[i][0] = math.exp(x[i][0]) - 1
        else:
            t[i][0] = x[i][0]            
    return t

def d_ELU(x):
    t = np.ones((x.size, 1))
    for i in range(x.size):
        if x[i] < 0:
            t[i][0] = math.exp(x[i][0])       
    return t

def sq_err(ans, y):
    return (ans - y)*(ans - y)/2

def d_sq_err(ans, y):
    return ans - y

def rand(x, y = 1):
    return random.random()

LAYER_SIZES = [10, 15, 15, 1]
LAYER_FUNCS = [ELU]*len(LAYER_SIZES)
LAYER_D_FUNCS = [d_ELU]*len(LAYER_SIZES)

def forward(net, x):
    t = [(net[0][0] @ x) + net[1][0]]
    h = [LAYER_FUNCS[0](t[0])]
    for i in range(len(LAYER_SIZES) - 1):
        t += [(net[0][i + 1] @ h[i]) + net[1][i + 1]]
        h += [LAYER_FUNCS[0](t[i + 1])]
    return [t, h, x, h[-1][0][0]]

def backward(net, ans, targ):
    L = len(LAYER_SIZES)
    grad_w = [np.ones((LAYER_SIZES[0], DIMENSION))]
    grad_b = [np.ones((LAYER_SIZES[0], 1))]
    grad_t = [np.ones((LAYER_SIZES[0], 1))]
    grad_h = [np.ones((LAYER_SIZES[0], 1))]
    for i in range(L - 1):
        grad_w += [np.ones((LAYER_SIZES[i + 1], LAYER_SIZES[i]))]
        grad_b += [np.ones((LAYER_SIZES[i + 1], 1))]
        grad_t += [np.ones((LAYER_SIZES[i + 1], 1))]
        grad_h += [np.ones((LAYER_SIZES[i + 1], 1))]
    
    grad_h[-1][0][0] = d_sq_err(ans[1][-1], targ)
    grad_t[-1] = grad_h[-1]*LAYER_D_FUNCS[-1](ans[0][-1])
    grad_b[-1] = grad_t[-1]
    grad_w[-1] = grad_t[-1] @ ((ans[1][-2]).T)
    for i in range(L - 1):
        grad_h[L - 2 - i] = net[0][L - 1 - i].T @ grad_t[L - 1 - i]
        grad_t[L - 2 - i] = grad_h[L - 2 - i]*LAYER_D_FUNCS[L - 2 - i](ans[0][L - 2 - i])
        grad_b[L - 2 - i] = grad_t[L - 2 - i]
        if (i == L - 2):
            grad_w[0] = grad_t[0] @ ((ans[2]).T)
        else:
            grad_w[L - 2 - i] = grad_t[L - 2 - i] @ (ans[1][L - 3 - i]).T
    return [grad_w, grad_b]

def learn_neural_net(dataset):
    data_copy = dataset[:]
    net = [[],[]]
    L = len(LAYER_SIZES)
    net[0] = [np.random.rand(LAYER_SIZES[0], DIMENSION)]
    net[1] = [np.random.rand(LAYER_SIZES[0], 1)]
    print(net[0][0], net[1][0])
    for i in range(L - 1):
        net[0] += [np.random.rand(LAYER_SIZES[i + 1], LAYER_SIZES[i])]
        net[1] += [np.random.rand(LAYER_SIZES[i + 1], 1)]
        print(net[0][i + 1], net[1][i + 1])
    for k in range(EPOCHS):
        random.shuffle(data_copy)
        for i in range(SIZE):
            res = forward(net, np.array([[s] for s in data_copy[i][0]]))
            grad = backward(net, res, data_copy[i][1])
            for i in range(L):
                net[0][i] -= ALPHA*grad[0][i]
                net[1][i] -= ALPHA*grad[1][i]
    return net

def neural_net_reg_plot(net):
    x = np.arange(LEFT_EDGE, RIGHT_EDGE, 0.1)
    y = [(forward(net, np.array([[x[i]]]))[3]) for i in range(len(x))]
    plt.plot(x, y)

print(np.array([[1], [2], [3], [4]]) * np.array([[10, 100, 1000, 10000]]))
net0 = learn_neural_net(dataset)
neural_net_reg_plot(net0)
data_reg_plot(dataset)
plt.show() 

