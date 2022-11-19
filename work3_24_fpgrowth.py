import numpy as np
import pandas as pd
import math
import collections
import itertools


def createFPTree(df, min_support):
    num_itemsets = len(df.index) #считаем строки
    itemsets = df.values
    #проверяем каждый элемент проходит ли он по уровню
    item_support = np.array(np.sum(itemsets, axis=0) / float(num_itemsets))
    item_support = item_support.reshape(-1)
    items = np.nonzero(item_support >= min_support)[0]
    #Сортируем элементы
    indices = item_support[items].argsort()
    rank = {item: i for i, item in enumerate(items[indices])}
    #строим дерево
    tree = FPTree(rank)
    for i in range(num_itemsets):
        nonnull = np.where(itemsets[i, :])[0]
        itemset = [item for item in nonnull if item in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_itemset(itemset)
    return tree


def result(generator, num_itemsets):
    itemsets = []
    supports = []
    for sup, iset in generator:
        itemsets.append(frozenset(iset))
        supports.append(sup / num_itemsets)
    res_df = pd.DataFrame({"support": supports, "itemsets": itemsets})
    return res_df

class FPTree(object):
    def __init__(self, rank=None):#сначала мы заходим сюда
        self.root = FPNode(None)#и создаём пустую голову
        self.nodes = collections.defaultdict(list)
        self.cond_items = []
        self.rank = rank

    def conditional_tree(self, cond_item, minsup):        
        branches = []
        count = collections.defaultdict(int)
        for node in self.nodes[cond_item]:#ищем все пути до айтема от головы
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count

        #повторно сортируем дерево во избежание проблем
        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}
        #делаем проецируемое дерево
        cond_tree = FPTree(rank)
        for idx, branch in enumerate(branches):
            branch = sorted(
                [i for i in branch if i in rank], key=rank.get, reverse=True
            )
            cond_tree.insert_itemset(branch, self.nodes[cond_item][idx].count)
        cond_tree.cond_items = self.cond_items + [cond_item]
        return cond_tree

    def insert_itemset(self, itemset, count=1):
        #Вставляем наш список в дерево
        self.root.count += count
        if len(itemset) == 0:
            return

        #идём по существующему пути в дереве пока можем
        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
                index += 1
            else:
                break

        #вставляем в дерево всё что осталось
        for item in itemset[index:]:
            child_node = FPNode(item, count, node)
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True


class FPNode(object):
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = collections.defaultdict(FPNode)

        if parent is not None:
            parent.children[item] = self

    def itempath_from_root(self):
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path
    
def fpg_step(tree, minsup):
    #Рекурсивно делаем fpgrowth
    count = 0
    items = tree.nodes.keys()
    if tree.is_path():
        #дерево в виде пути случай не наш, но тут отличие в том что рекурсивного вызова функции не происходит
        size_remain = len(items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                support = min([tree.nodes[i][0].count for i in itemset])
                yield support, tree.cond_items + list(itemset)
    else:
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])
            yield support, tree.cond_items + [item]

    #строим проецируемые fp-деревья
    if not tree.is_path():
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)
            for sup, iset in fpg_step(cond_tree, minsup):
                yield sup, iset


data=        ([True,  True,  False, False, True,  False, False],#ABE
              [False, False, False, False, False, True,  True], #FG
              [True,  False, True,  True,  False, False, False],#ACD
              [True,  True,  False, False, True,  False, False],#ABE
              [False, False, True,  True,  False, True,  True], #CDFG
              [True,  False, False, False, True,  False, False],#AE
              [True,  True,  False, False, False, True,  True], #ABFG
              [True,  False, True,  False, False, False, False],#AC
              [False, False, True,  True,  True,  False, False],#CDE
              [False, True,  True,  True,  False, True,  True])  #BCDFG

df=pd.DataFrame(data)
min_support=0.3
tree = createFPTree(df, min_support)
minsup = math.ceil(min_support * len(df.index))#пересчитываем с округлением до большего целого
generator = fpg_step(tree, minsup)
print(result(generator, len(df.index)))


