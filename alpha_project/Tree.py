
# coding: utf-8

# In[188]:


class Node(object):
    
    def __init__(self, value, level):
        """
        Parameters
        --------
        value: str
            Name of city\street\house\block for this node.
        level: int
            Nesting level in tree.
            
        Attributes
        --------
        level: int
            Nesting level in tree.
        parent: Node
            Father for our node. Default value = None.
        child: dict of Node
            dict of children for node. Default value = {}.
        lpu_id: list of int
            Lpu_id list of all suitable lpu for our node. Default value = [].
        value: str
            Name of city\street\house\block for this node.
        
        """
        
        self.level = level
        self.parent = None
        self.child = {}
        self.lpu_id = []
        self.value = value
        
    def print(self, level):
        """ Print our tree. """
        
        if level >= 0:
            print('    ' * level, self.value)

        for i in self.child.keys():
            self.child[i].print(level + 1)


class Tree(object):
    
    def __init__(self):
        """
        Attributes
        --------
        root : Node
            Root of our tree.
        """
        
        self.root = Node(value='0', level=0)

    def add_elem(self, lpu_address, id_):
        """ Function add lpu in tree.
        
        Parameters
        --------
        lpu_address : dict
            For example, {'city': 'Волгоград', 'street': 'им Наумова', 'house': '4', 'block': '3'}.
        id_ : int
            Lpu id.
        
        """
        
        if not lpu_address:
            return self
        
        a = self.root
        a.lpu_id.append(id_)
        
        for i in lpu_address.keys():
            try:
                a.child[lpu_address[i]].parent = a
            except:
                a.child[lpu_address[i]] = Node(lpu_address[i], a.level + 1)
                a.child[lpu_address[i]].parent = a
               
            a = a.child[lpu_address[i]]
            a.lpu_id.append(id_) 
  
        return self
    
    def suitable_nodes(self, lpu_address):
        """ Function find highest node with equal subjects name.
        
        Parameters
        --------
        lpu_address: dict
            For example, {'city': 'Волгоград', 'street': 'им Наумова', 'house': '4', 'block': '3'}.
        
        """
        
        if not lpu_address:
            return self.root
        
        a = self.root
        for i in lpu_address.keys():
            try:
                a = a.child[lpu_address[i]]
            except:
                return a
        
        return a

    def print(self):
        """ Print our tree. """
       
        self.root.print(level=-1)
