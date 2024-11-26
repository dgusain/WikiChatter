'''
@author: Sougata Saha
@modifier: Divyesh Pratap Singh
Institute: University at Buffalo
'''

import math


class Node:

    def __init__(self, value=None, next=None, tf = 0, topic = None):
        """ Class to define the structure of each node in a linked list (postings list).
            Value: document id, Next: Pointer to the next node
            Add more parameters if needed.
            Hint: You may want to define skip pointers & appropriate score calculation here"""
        self.next = next
        self.skip_pointer = None
        self.value = value
        self.tf = tf
        self.tfidf = 0.0
        self.topic = topic

class LinkedList:
    """ Class to define a linked list (postings list). Each element in the linked list is of the type 'Node'
        Each term in the inverted index has an associated linked list object.
        Feel free to add additional functions to this class."""
    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.length, self.n_skips, self.idf = 0, 0, 0.0
        self.skip_length = None

    def traverse_list(self):
        traversal = []
        current = self.start_node
        if current is None:
            return
        else:
            """ Write logic to traverse the linked list.
                To be implemented."""
            while current:
                traversal.append(current.value)
                current = current.next
            return traversal

    def traverse_skips(self):
        traversal = []
        current = self.start_node
        if current is None:
            return traversal
        else:
            """ Write logic to traverse the linked list using skip pointers.
                To be implemented."""
            while current.skip_pointer:
                traversal.append(current.value)
                current = current.skip_pointer
            traversal.append(current.value)
        return traversal
    
    def add_skip_connections(self):
        n_skips = math.floor(math.sqrt(self.length))
        self.skip_length = int(round(math.sqrt(self.length), 0))
        if self.skip_length == 0:
            self.skip_length = 1  # Avoid division by zero
        if n_skips * n_skips == self.length:
            n_skips = n_skips - 1
        """ Write logic to add skip pointers to the linked list. 
            This function does not return anything.
            To be implemented."""

        if not self.start_node:
            return
        slow = self.start_node
        fast = slow
        while n_skips > 0:
            for i in range(self.skip_length):
                if fast.next:
                    fast = fast.next
                else:
                    fast = None
                    break
            if fast:
                slow.skip_pointer = fast
                slow = fast
            n_skips -= 1
            
    def insert_in_order(self, doc_id, topic):
        """ Write logic to add new elements to the linked list.
            Insert the element at an appropriate position, such that elements to the left are lower than the inserted
            element, and elements to the right are greater than the inserted element.
            To be implemented. """
        if not self.start_node:
            self.start_node = Node(value=doc_id, tf=1, topic=topic)
            self.length += 1
            return
        
        current = self.start_node
        prev = None
        
        while current and current.value < doc_id:
            prev = current
            current = current.next

        if current and current.value == doc_id:
            current.tf += 1
        else:
            new_node = Node(value=doc_id, tf=1, topic=topic)
            if prev is None:
                new_node.next = self.start_node
                self.start_node = new_node
            else:
                prev.next = new_node
                new_node.next = current
            self.length += 1

    def get_postings(self):
        """
        Traverse the linked list and return a list of dictionaries with doc_id and tfidf.
        """
        postings = []
        current = self.start_node
        while current:
            postings.append({
                'doc_id': current.value,
                'tfidf': current.tfidf,
                'topic': current.topic
            })
            current = current.next
        return postings